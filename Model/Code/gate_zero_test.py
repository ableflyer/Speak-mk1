"""
Gate isolation test.

Runs the SAME prompt through the LLM twice:
  1. cross_attn.gate forced to 0.0   -> audio conditioning effectively OFF
  2. cross_attn.gate forced to 0.3   -> your current inference setting

If (1) is coherent and (2) is not, the audio-conditioning path is the
corruptor. If both are incoherent, the problem is base LM fluency /
data starvation, independent of audio.

Adjust the import paths at the top to match your actual file layout
(these assume the same layout as your Flask api.py).
"""

import sys
import numpy as np
import torch
import librosa

from audio_encoder import AudioEncoder, AudioEncoderConfig
from audio_trainer import SmallConfig
from speak_mk1_llm import SpeakMK1LLM, SpeakMK1LLMConfig
from train_proj import DirectAudioProjection
from transformers import AutoTokenizer

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SAMPLE_RATE = 16000

ENCODER_PATH = "../Model_files/Audio_encoder_v1.1/audio_encoder_epoch_5.pt"
PROJ_PATH    = "../Model_files/audio_proj_training/audio_proj_best.pt"
LLM_PATH     = "../Model_files/checkpoints_v2.3.1/stage5_audio/ckpt_final.pt"


# ── Load models (trimmed: no Kokoro, don't need TTS for this test) ─────────

def load_models():
    print("Loading encoder...", file=sys.stderr)
    enc_cfg = SmallConfig(d_model=512, llm_dim=4096)
    encoder = AudioEncoder(enc_cfg).to(DEVICE)
    enc_ckpt = torch.load(ENCODER_PATH, map_location=DEVICE, weights_only=False)
    encoder.load_state_dict(enc_ckpt, strict=False)
    encoder.eval()

    print("Loading projection...", file=sys.stderr)
    proj = DirectAudioProjection(512, 512).to(DEVICE)
    proj_ckpt = torch.load(PROJ_PATH, map_location=DEVICE, weights_only=False)
    proj.load_state_dict(proj_ckpt["audio_proj"])
    proj.eval()

    print("Loading LLM...", file=sys.stderr)
    llm_cfg = SpeakMK1LLMConfig(
        vocab_size=50283, d_model=512, d_state=64, num_blocks=6,
        nheads_ssm=8, nheads_attn=8, top_k_audio=32,
        num_experts=4, top_k_experts=2, dropout=0.0, aux_loss_weight=1e-2,
    )
    llm = SpeakMK1LLM(llm_cfg).to(DEVICE)
    llm_ckpt = torch.load(LLM_PATH, map_location=DEVICE, weights_only=False)
    llm.load_state_dict(llm_ckpt["model"], strict=True)
    llm.eval()

    print("Loading tokenizer...", file=sys.stderr)
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
    tokenizer.add_special_tokens({"additional_special_tokens": [
        "<|system|>", "<|child|>", "<|slp|>", "<|think|>", "<|endturn|>"
    ]})

    return encoder, proj, llm, tokenizer


def set_gate(llm, value: float):
    with torch.no_grad():
        for block in llm.blocks:
            if hasattr(block.cross_attn, "gate"):
                block.cross_attn.gate.data.fill_(value)


# ── Generation helpers (copied from your inference code, unchanged) ────────

def apply_repetition_penalty(logits, generated_ids, penalty=1.3):
    for token_id in set(generated_ids[0].tolist()):
        if logits[token_id] > 0:
            logits[token_id] /= penalty
        else:
            logits[token_id] *= penalty
    return logits


def get_banned_ngram_tokens(generated_ids, ngram_size=3):
    generated = generated_ids[0].tolist()
    if len(generated) < ngram_size:
        return set()
    ngrams = {}
    for i in range(len(generated) - ngram_size + 1):
        prefix = tuple(generated[i:i + ngram_size - 1])
        next_tok = generated[i + ngram_size - 1]
        ngrams.setdefault(prefix, set()).add(next_tok)
    current_prefix = tuple(generated[-(ngram_size - 1):])
    return ngrams.get(current_prefix, set())


def generate_text(llm, tokenizer, input_ids, audio_out,
                   max_new_tokens=150, temperature=0.7, top_p=0.9,
                   repetition_penalty=1.2, no_repeat_ngram_size=3):
    generated = input_ids.clone()
    with torch.no_grad():
        for _ in range(max_new_tokens):
            logits, _, _ = llm(input_ids=generated, audio_out=audio_out, audio_padding_mask=None)
            next_logits = logits[0, -1] / temperature
            next_logits = apply_repetition_penalty(next_logits, generated, repetition_penalty)
            banned = get_banned_ngram_tokens(generated, no_repeat_ngram_size)
            for tok in banned:
                next_logits[tok] = -float("inf")
            next_logits = next_logits / temperature

            sorted_logits, sorted_indices = torch.sort(next_logits, descending=True)
            cumprobs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
            mask = cumprobs - torch.softmax(sorted_logits, dim=-1) > top_p
            sorted_logits[mask] = -float("inf")
            probs = torch.softmax(sorted_logits, dim=-1)
            next_token = sorted_indices[torch.multinomial(probs, 1)]

            generated = torch.cat([generated, next_token.unsqueeze(0)], dim=-1)
            if next_token.item() == tokenizer.eos_token_id:
                break
    return tokenizer.decode(generated[0], skip_special_tokens=False)


def build_prompt(transcription, target_phrase, target_sound, accuracy, child_age, focus_area):
    system = (
        "You are a professional pediatric speech-language pathology assistant. "
        "Give brief, clinical, encouraging feedback about the child's pronunciation. "
        "Do not reference physical touch, body parts other than mouth/tongue placement "
        "in a clinical/anatomical way, and do not use secretive or personal language. "
        "Keep responses under 3 sentences."
    )
    context = (
        f"Target phrase: '{target_phrase}'. Target sound: /{target_sound}/. "
        f"Accuracy score: {accuracy:.0f}%. Child age: {child_age}. Focus area: {focus_area}."
    )
    child_turn = f"{context} Child said: \"{transcription}\""
    return f"<|system|>{system}<|child|>{child_turn}<|slp|>"


def make_silent_mel():
    silent = np.zeros(SAMPLE_RATE * 1, dtype=np.float32)
    mel_np = librosa.feature.melspectrogram(
        y=silent, sr=SAMPLE_RATE, n_fft=400, hop_length=160, n_mels=80,
    )
    mel_np = librosa.power_to_db(mel_np, ref=np.max)
    return torch.tensor(mel_np.T, dtype=torch.float32).unsqueeze(0).to(DEVICE)


# ── Main test ────────────────────────────────────────────────────────────

def main():
    encoder, proj, llm, tokenizer = load_models()

    # Same test case as your last output, for direct comparison
    prompt = build_prompt(
        transcription="I wanna pway wif my fwiends.",
        target_phrase="I want to play with my friends.",
        target_sound="l",
        accuracy=62,
        child_age=6,
        focus_area="Liquid consonants",
    )
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(DEVICE)

    # Use silent mel so audio conditioning is a known-constant input across
    # both runs -- isolates the GATE VALUE as the only variable, not the audio content.
    mel = make_silent_mel()
    with torch.no_grad():
        audio_feats = encoder.encode_features(mel)
        audio_out = proj(audio_feats)

    print("\n" + "=" * 70)
    print("PROMPT")
    print("=" * 70)
    print(prompt)

    for gate_value in [0.0, 0.3]:
        set_gate(llm, gate_value)
        torch.manual_seed(0)  # same sampling seed across runs for fair comparison
        full_output = generate_text(llm, tokenizer, input_ids, audio_out)
        _, _, slp_response = full_output.partition("<|slp|>")
        slp_response = slp_response.replace("<|endturn|>", "").strip()

        print("\n" + "=" * 70)
        print(f"GATE = {gate_value}")
        print("=" * 70)
        print(slp_response)

    print("\n" + "=" * 70)
    print("Also worth running: text-only sanity check with NO audio at all")
    print("(pass audio_out=None if your model's forward supports it) to see")
    print("if the LLM can produce coherent output on this prompt when the")
    print("cross-attention pathway is bypassed entirely, not just zeroed.")
    print("=" * 70)


if __name__ == "__main__":
    main()