"""
free_running_spot_check.py
===========================
Quick spot check comparing teacher-forced vs free-running (autoregressive)
generation quality for the full SpeakMK1 pipeline (AudioEncoder ->
DirectAudioProjection -> SpeakMK1LLM).

For each (audio, child_text, target_response) triple it:
  1. Encodes the audio through AudioEncoder + projection (same as your
     live-mic inference script)
  2. TEACHER-FORCED pass: feeds the full prompt+target sequence in one
     forward pass and measures next-token top-1 accuracy + perplexity
     against the real target tokens. This is what training-time metrics
     reflect.
  3. FREE-RUNNING pass: greedy-decodes starting from the prompt only, using
     the model's own previous outputs at every step. This is what real
     deployment looks like.
  4. Prints both side by side so you can see how much the numbers diverge.

ASSUMPTIONS (mirroring audio_tester.py's convention of flagging what's
unverified against the real codebase):
  - Manifest is a CSV with columns: wav_path, child_text, target_text
  - target_text is the ideal SLP response text, no special tokens included
  - Prompt template matches the one in your live-mic inference script
  - Greedy decoding (argmax) is used for BOTH passes so the comparison is
    apples-to-apples. The live-mic script uses top-p sampling; that's not
    used here since it would confound teacher-forced vs free-running.
  - Checkpoint paths in load_models() default to the ones in your live-mic
    script -- override with CLI flags if your latest checkpoints differ.
  - encoder.encode_features() and DirectAudioProjection's forward signature
    are unverified beyond what the live-mic script shows.

USAGE
-----
    python free_running_spot_check.py \
        --manifest stage5_val_manifest.csv \
        --num-samples 8

If your manifest's columns are named differently:
    python free_running_spot_check.py \
        --manifest stage5_val_manifest.csv \
        --wav-col audio_path --child-col child_utterance --target-col slp_response
"""

import argparse
import csv
import math
import random

import torch
import torch.nn.functional as F
import librosa
import numpy as np

from audio_encoder import AudioEncoder, AudioEncoderConfig
from audio_trainer import SmallConfig
from speak_mk1_llm import SpeakMK1LLM, SpeakMK1LLMConfig
from transformers import AutoTokenizer
from train_proj import DirectAudioProjection

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

PROMPT_TEMPLATE = (
    "<|system|>You are a warm, expert AI speech-language pathologist helping "
    "a child with articulation errors. Analyze the error and provide "
    "encouraging corrective feedback.<|child|> {child_text}<|slp|>"
)


def load_models(
    encoder_ckpt="../Model_files/Audio_encoder_v1.1/audio_encoder_epoch_5.pt",
    proj_ckpt="../Model_files/audio_proj_training/audio_proj_best.pt",
    llm_ckpt="../Model_files/checkpoints_v2.3.1/stage5_audio/ckpt_final.pt",
):
    enc_cfg = SmallConfig(d_model=512, llm_dim=4096)
    encoder = AudioEncoder(enc_cfg).to(DEVICE)
    enc_state = torch.load(encoder_ckpt, map_location=DEVICE, weights_only=False)
    encoder.load_state_dict(enc_state, strict=False)
    encoder.eval()

    proj = DirectAudioProjection(512, 512).to(DEVICE)
    proj_state = torch.load(proj_ckpt, map_location=DEVICE, weights_only=False)
    proj.load_state_dict(proj_state["audio_proj"])
    proj.eval()

    llm_cfg = SpeakMK1LLMConfig(
        vocab_size=50283,
        d_model=512,
        d_state=64,
        num_blocks=6,
        nheads_ssm=8,
        nheads_attn=8,
        top_k_audio=32,
        num_experts=4,
        top_k_experts=2,
        dropout=0.0,
        aux_loss_weight=1e-2,
    )
    llm = SpeakMK1LLM(llm_cfg).to(DEVICE)
    llm_state = torch.load(llm_ckpt, map_location=DEVICE, weights_only=False)
    llm.load_state_dict(llm_state["model"], strict=True)
    # Same gate-collapse workaround as the live-mic script
    with torch.no_grad():
        for block in llm.blocks:
            if hasattr(block.cross_attn, "gate"):
                block.cross_attn.gate.data.fill_(0.3)
    llm.eval()

    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
    tokenizer.add_special_tokens({"additional_special_tokens": [
        "<|system|>", "<|child|>", "<|slp|>", "<|think|>", "<|endturn|>"
    ]})

    return encoder, proj, llm, tokenizer


def encode_audio(encoder, proj, wav_path, sr=16000):
    y, _ = librosa.load(wav_path, sr=sr)
    mel_np = librosa.feature.melspectrogram(
        y=y, sr=sr, n_fft=400, hop_length=160, n_mels=80, fmin=0.0, fmax=8000.0,
    )
    mel_np = librosa.power_to_db(mel_np, ref=np.max)
    mel = torch.tensor(mel_np.T, dtype=torch.float32).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        audio_feats = encoder.encode_features(mel)
        audio_out = proj(audio_feats)
    return audio_out


def teacher_forced_pass(llm, tokenizer, prompt_ids, target_ids, audio_out):
    """Single forward pass over prompt+target, ground-truth tokens at every
    position -- this is what training loss/accuracy reflects."""
    full_ids = torch.cat([prompt_ids, target_ids], dim=-1)
    with torch.no_grad():
        logits, _, _ = llm(input_ids=full_ids, audio_out=audio_out, audio_padding_mask=None)

    # Only score the target span (positions after the prompt)
    prompt_len = prompt_ids.shape[-1]
    target_logits = logits[0, prompt_len - 1:-1]     # predicts target_ids[0..n-1]
    target_labels = target_ids[0]

    pred_ids = target_logits.argmax(dim=-1)
    correct = (pred_ids == target_labels).sum().item()
    total = target_labels.shape[0]
    acc = correct / total if total else 0.0

    ce = F.cross_entropy(target_logits, target_labels, reduction="mean")
    ppl = math.exp(ce.item())

    return acc, ppl


def free_running_pass(llm, tokenizer, prompt_ids, audio_out, max_new_tokens=60):
    """Greedy autoregressive decode -- the model only ever sees its own
    previous outputs, matching real deployment."""
    generated = prompt_ids.clone()
    with torch.no_grad():
        for _ in range(max_new_tokens):
            logits, _, _ = llm(input_ids=generated, audio_out=audio_out, audio_padding_mask=None)
            next_token = logits[0, -1].argmax(dim=-1, keepdim=True).unsqueeze(0)
            generated = torch.cat([generated, next_token], dim=-1)
            if next_token.item() == tokenizer.eos_token_id:
                break
    gen_text = tokenizer.decode(generated[0, prompt_ids.shape[-1]:], skip_special_tokens=False)
    return gen_text


def token_overlap_ratio(hyp_text, ref_text, tokenizer):
    """Crude free-running quality signal: fraction of reference tokens that
    also appear in the generated text, in any order. Not a real metric --
    just enough to flag gross divergence at a glance tonight."""
    hyp_ids = set(tokenizer(hyp_text).input_ids)
    ref_ids = set(tokenizer(ref_text).input_ids)
    if not ref_ids:
        return 0.0
    return len(hyp_ids & ref_ids) / len(ref_ids)


def load_manifest(path, num_samples, seed):
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)
    rng = random.Random(seed)
    n = min(num_samples, len(rows))
    return rng.sample(rows, n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--wav-col", default="wav_path")
    ap.add_argument("--child-col", default="child_text")
    ap.add_argument("--target-col", default="target_text")
    ap.add_argument("--num-samples", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-new-tokens", type=int, default=60)
    ap.add_argument("--encoder-ckpt", default="../Model_files/Audio_encoder_v1.1/audio_encoder_epoch_5.pt")
    ap.add_argument("--proj-ckpt", default="../Model_files/audio_proj_training/audio_proj_best.pt")
    ap.add_argument("--llm-ckpt", default="../Model_files/checkpoints_v2.3.1/stage5_audio/ckpt_final.pt")
    args = ap.parse_args()

    encoder, proj, llm, tokenizer = load_models(args.encoder_ckpt, args.proj_ckpt, args.llm_ckpt)
    rows = load_manifest(args.manifest, args.num_samples, args.seed)

    tf_accs, tf_ppls, overlaps = [], [], []
    print("=" * 90)
    for i, row in enumerate(rows):
        wav_path = row[args.wav_col]
        child_text = row[args.child_col]
        target_text = row[args.target_col]

        audio_out = encode_audio(encoder, proj, wav_path)

        prompt = PROMPT_TEMPLATE.format(child_text=child_text)
        prompt_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(DEVICE)
        target_ids = tokenizer(target_text, return_tensors="pt").input_ids.to(DEVICE)

        tf_acc, tf_ppl = teacher_forced_pass(llm, tokenizer, prompt_ids, target_ids, audio_out)
        fr_text = free_running_pass(llm, tokenizer, prompt_ids, audio_out, args.max_new_tokens)
        overlap = token_overlap_ratio(fr_text, target_text, tokenizer)

        tf_accs.append(tf_acc)
        tf_ppls.append(tf_ppl)
        overlaps.append(overlap)

        print(f"[{i}] wav={wav_path}")
        print(f"    child utterance : {child_text}")
        print(f"    target response : {target_text}")
        print(f"    TEACHER-FORCED  : acc={tf_acc:.3f}  ppl={tf_ppl:.2f}")
        print(f"    FREE-RUNNING    : \"{fr_text.strip()}\"")
        print(f"    token overlap vs target: {overlap:.3f}")
        print("-" * 90)

    print(f"\nAvg teacher-forced token accuracy : {sum(tf_accs)/len(tf_accs):.3f}")
    print(f"Avg teacher-forced perplexity      : {sum(tf_ppls)/len(tf_ppls):.2f}")
    print(f"Avg free-running token overlap     : {sum(overlaps)/len(overlaps):.3f}")
    print("=" * 90)


if __name__ == "__main__":
    main()