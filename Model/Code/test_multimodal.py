"""
test_speakmk1.py
================
Two-section test for SpeakMK1Multimodal:

    Section 1 — LibriSpeech-test evaluation
        Loads wav/txt pairs from librispeech-test-clean, runs inference,
        prints transcription alongside model response, computes avg loss.

    Section 2 — Real-time microphone chat
        Records from mic, encodes audio, generates a response, prints it.
        Loops until you type 'quit'.

Usage:
    # Section 1 only
    python test_speakmk1.py --mode eval

    # Section 2 only
    python test_speakmk1.py --mode chat

    # Both
    python test_speakmk1.py --mode both
"""

import argparse
import os
import sys
import time
import glob
import queue
import threading

import torch
import torch.nn.functional as F
import librosa
import numpy as np
from transformers import AutoTokenizer

# ── Local imports ─────────────────────────────────────────────────────────
from audio_encoder import AudioEncoder, AudioEncoderConfig
from speak_mk1_llm import SpeakMK1LLM, SpeakMK1LLMConfig
from speak_mk1_combined import SpeakMK1Multimodal, DirectAudioProjection
from audio_trainer import SmallConfig


# ════════════════════════════════════════════════════════════════════════════
# CONFIG — edit these paths before running
# ════════════════════════════════════════════════════════════════════════════

class TestConfig:
    # ── Checkpoint ────────────────────────────────────────────────────────
    # Point to the best Phase 2 checkpoint from multimodal training
    multimodal_ckpt = "../Model_files/multimodal_training_v2.1/phase2_epoch1.pt"

    # Fallback: load components separately if combined ckpt not found
    audio_encoder_ckpt = "../Model_files/Audio_encoder_v1.1/audio_encoder_epoch_2.pt"
    llm_ckpt           = "../Model_files/checkpoints_v2.3.1/stage4/ckpt_final.pt"
    audio_proj_ckpt    = "../Model_files/multimodal_training_v2.1/phase2_epoch1.pt"

    # ── LibriSpeech test data ─────────────────────────────────────────────
    test_data_root = "../Data/PhonemeDatasets/LibriSpeech/LibriSpeech/test-clean"
    max_eval_samples = 20      # how many utterances to evaluate

    # ── Tokeniser ─────────────────────────────────────────────────────────
    tokenizer_name = "EleutherAI/gpt-neox-20b"
    pad_token_id   = 1
    max_text_len   = 256

    # ── Audio ─────────────────────────────────────────────────────────────
    sample_rate  = 16_000
    n_mels       = 80
    n_fft        = 400
    hop_length   = 160
    max_frames   = 512

    # ── Model dims — must match training ──────────────────────────────────
    audio_d_model = 512
    llm_d_model   = 512
    vocab_size    = 50285    # always override
    num_experts   = 4        # always override

    # ── Generation ────────────────────────────────────────────────────────
    max_new_tokens = 80
    temperature    = 0.7
    top_p          = 0.9

    # ── Microphone ────────────────────────────────────────────────────────
    mic_record_seconds = 5     # seconds to record per turn
    mic_sample_rate    = 16_000


# ════════════════════════════════════════════════════════════════════════════
# MODEL LOADER
# ════════════════════════════════════════════════════════════════════════════

def load_model(cfg: TestConfig, device: torch.device) -> SpeakMK1Multimodal:
    """
    Loads SpeakMK1Multimodal from checkpoint.

    Tries the combined multimodal checkpoint first.
    Falls back to loading components separately if not found.
    """
    audio_cfg = SmallConfig(d_model=cfg.audio_d_model, llm_dim=cfg.llm_d_model)

    llm_cfg = SpeakMK1LLMConfig(
        vocab_size    = cfg.vocab_size,
        d_model       = cfg.llm_d_model,
        d_state       = 64,
        num_blocks    = 6,
        nheads_ssm    = 8,
        nheads_attn   = 8,
        top_k_audio   = 32,
        num_experts   = cfg.num_experts,
        top_k_experts = 2,
        dropout       = 0.0,
        aux_loss_weight = 1e-2,
    )

    print("Building model architecture...")
    model = SpeakMK1Multimodal(
        audio_cfg       = audio_cfg,
        llm_cfg         = llm_cfg,
        freeze_backbone = False,   # inference — no freezing needed
    ).to(device)

    # ── Try loading combined checkpoint ───────────────────────────────────
    if os.path.exists(cfg.multimodal_ckpt):
        print(f"  Loading combined checkpoint: {cfg.multimodal_ckpt}")
        ckpt = torch.load(cfg.multimodal_ckpt, map_location=device, weights_only=False)
        state = ckpt["model"] if "model" in ckpt else ckpt
        missing, unexpected = model.load_state_dict(state, strict=False)
        if missing:
            print(f"  WARNING — missing keys: {len(missing)}")
        print("  Combined checkpoint loaded.")

    # ── Fallback: load components separately ──────────────────────────────
    else:
        print(f"  Combined ckpt not found, loading components separately...")

        # Audio encoder
        if os.path.exists(cfg.audio_encoder_ckpt):
            model.load_audio_encoder(cfg.audio_encoder_ckpt, device)
        else:
            print(f"  WARNING — audio encoder ckpt not found: {cfg.audio_encoder_ckpt}")

        # LLM
        if os.path.exists(cfg.llm_ckpt):
            model.load_llm(cfg.llm_ckpt, device)
        else:
            print(f"  WARNING — LLM ckpt not found: {cfg.llm_ckpt}")

        # Audio projection (from phase1 or phase2 checkpoint)
        if os.path.exists(cfg.audio_proj_ckpt):
            proj_ckpt = torch.load(cfg.audio_proj_ckpt, map_location=device, weights_only=False)
            proj_state = proj_ckpt.get("audio_proj", proj_ckpt)
            model.audio_proj.load_state_dict(proj_state, strict=True)
            print(f"  Audio projection loaded: {cfg.audio_proj_ckpt}")

    model.eval()
    total = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"  Total parameters: {total:.1f}M")
    return model


# ════════════════════════════════════════════════════════════════════════════
# AUDIO UTILITIES
# ════════════════════════════════════════════════════════════════════════════

def wav_to_mel(
    wav_path: str,
    cfg: TestConfig,
    device: torch.device,
) -> torch.Tensor:
    """
    Loads a wav file and converts to a padded log-mel tensor.

    Returns
    -------
    mel : (1, max_frames, n_mels)  — batch dim=1 for single utterance
    """
    import torchaudio

    y, _ = librosa.load(wav_path, sr=cfg.sample_rate)
    waveform = torch.from_numpy(y).unsqueeze(0)   # (1, samples)

    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate = cfg.sample_rate,
        n_fft       = cfg.n_fft,
        hop_length  = cfg.hop_length,
        n_mels      = cfg.n_mels,
    )
    db_transform = torchaudio.transforms.AmplitudeToDB()

    mel = mel_transform(waveform)           # (1, n_mels, T)
    mel = db_transform(mel).squeeze(0).T   # (T, n_mels)

    T = mel.shape[0]
    if T >= cfg.max_frames:
        mel = mel[:cfg.max_frames]
    else:
        mel = F.pad(mel, (0, 0, 0, cfg.max_frames - T))

    return mel.unsqueeze(0).to(device)     # (1, max_frames, n_mels)


def numpy_to_mel(
    y: np.ndarray,
    cfg: TestConfig,
    device: torch.device,
) -> torch.Tensor:
    """
    Converts a raw numpy waveform array to a padded mel tensor.
    Used by the microphone section.

    Returns
    -------
    mel : (1, max_frames, n_mels)
    """
    import torchaudio

    waveform = torch.from_numpy(y.astype(np.float32)).unsqueeze(0)

    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate = cfg.sample_rate,
        n_fft       = cfg.n_fft,
        hop_length  = cfg.hop_length,
        n_mels      = cfg.n_mels,
    )
    db_transform = torchaudio.transforms.AmplitudeToDB()

    mel = mel_transform(waveform)
    mel = db_transform(mel).squeeze(0).T   # (T, n_mels)

    T = mel.shape[0]
    if T >= cfg.max_frames:
        mel = mel[:cfg.max_frames]
    else:
        mel = F.pad(mel, (0, 0, 0, cfg.max_frames - T))

    return mel.unsqueeze(0).to(device)


# ════════════════════════════════════════════════════════════════════════════
# GENERATION HELPER
# ════════════════════════════════════════════════════════════════════════════

def generate_response(
    model:     SpeakMK1Multimodal,
    mel:       torch.Tensor,
    tokenizer,
    cfg:       TestConfig,
    device:    torch.device,
    prompt:    str = "",
) -> str:
    with torch.no_grad():
        if prompt:
            prompt_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
        else:
            bos_id = tokenizer.bos_token_id or 1
            prompt_ids = torch.tensor([[bos_id]], device=device)

        print(f"Prompt decoded: {tokenizer.decode(prompt_ids[0])}")
        print(f"Prompt token count: {prompt_ids.shape[1]}")

        generated = model.generate(
            mel            = mel,
            prompt_ids     = prompt_ids,
            max_new_tokens = cfg.max_new_tokens,
            temperature    = cfg.temperature,
            top_p          = cfg.top_p,
        )

        new_tokens = generated[0, prompt_ids.shape[1]:]

        # Stop at first special token
        stop_ids = {
            tokenizer.pad_token_id,
            tokenizer.eos_token_id,
            tokenizer.convert_tokens_to_ids("<|endturn|>"),
            tokenizer.convert_tokens_to_ids("<|endoftext|>"),
        }
        filtered = []
        for tid in new_tokens.tolist():
            if tid in stop_ids:
                break
            filtered.append(tid)

        response = tokenizer.decode(filtered, skip_special_tokens=False)

    return response.strip()


# ════════════════════════════════════════════════════════════════════════════
# SECTION 1 — LibriSpeech-test Evaluation
# ════════════════════════════════════════════════════════════════════════════

def run_librispeech_eval(model: SpeakMK1Multimodal, tokenizer, cfg: TestConfig, device: torch.device):
    """
    Loads up to cfg.max_eval_samples utterances from librispeech-test-clean,
    runs inference on each, and prints:
        - Ground truth transcript
        - Model response
        - Per-sample LM loss

    Also prints average loss at the end.
    """
    print("\n" + "=" * 70)
    print("SECTION 1 — LibriSpeech-test Evaluation")
    print("=" * 70)

    # ── Find test wav/txt pairs ───────────────────────────────────────────
    wav_files = sorted(glob.glob(
        os.path.join(cfg.test_data_root, "**/*.wav"), recursive=True
    ))
    if not wav_files:
        print(f"  ERROR — no wav files found in: {cfg.test_data_root}")
        print("  Make sure test-clean is downloaded and the path is correct.")
        return

    # Limit to max_eval_samples
    wav_files = wav_files[:cfg.max_eval_samples]
    print(f"  Found {len(wav_files)} test utterances (max={cfg.max_eval_samples})\n")

    total_loss  = 0.0
    valid_count = 0

    for i, wav_path in enumerate(wav_files):
        txt_path = wav_path.replace(".wav", ".txt")

        # ── Load transcript ────────────────────────────────────────────────
        if not os.path.exists(txt_path):
            print(f"  [{i+1}] SKIP — no transcript: {txt_path}")
            continue

        with open(txt_path, "r", encoding="utf-8") as f:
            ground_truth = f.read().strip()

        if not ground_truth:
            print(f"  [{i+1}] SKIP — empty transcript")
            continue

        # ── Load audio → mel ───────────────────────────────────────────────
        try:
            mel = wav_to_mel(wav_path, cfg, device)
        except Exception as e:
            print(f"  [{i+1}] SKIP — audio load error: {e}")
            continue

        # ── Compute LM loss on ground truth ───────────────────────────────
        with torch.no_grad():
            token_ids = tokenizer(
                ground_truth,
                max_length     = cfg.max_text_len,
                padding        = "max_length",
                truncation     = True,
                return_tensors = "pt",
            )["input_ids"].to(device)

            labels = token_ids.clone()
            labels[labels == cfg.pad_token_id] = -100
            labels[labels == tokenizer.pad_token_id] = -100

            _, lm_loss, _ = model(
                mel       = mel,
                input_ids = token_ids,
                labels    = labels,
            )
            loss_val = lm_loss.item() if lm_loss is not None else float("nan")

        # ── Generate open response ─────────────────────────────────────────
        prompt = (
            "<|system|> You are SpeakMK1, a warm and clinically expert AI "
            "speech-language pathologist for children. "
            "The child is 6 years old and presents with Stopping (Stopping of /s/ and /z/ (s/z and t/d)). "
            "Clinical goal: Establish /s/ using airflow awareness. "
            "Primary strategy: Tactile Cueing (touching alveolar ridge awareness). "
            "Always analyze errors clinically before responding. Be encouraging, patient, and use scaffolding. "
            "<|endturn|>"
            "<|child|> I went to the tore yeterday <|endturn|>"
            "<|think|>"
        )
        response = generate_response(model, mel, tokenizer, cfg, device, prompt=prompt)

        # ── Print results ──────────────────────────────────────────────────
        fname = os.path.basename(wav_path)
        print(f"[{i+1:02d}] {fname}")
        print(f"  Ground truth : {ground_truth}")
        print(f"  Model output : {response}")
        print(f"  LM loss      : {loss_val:.4f}")
        print()

        if not np.isnan(loss_val):
            total_loss  += loss_val
            valid_count += 1

    # ── Summary ───────────────────────────────────────────────────────────
    if valid_count > 0:
        avg_loss = total_loss / valid_count
        print(f"{'=' * 70}")
        print(f"Evaluation complete — {valid_count}/{len(wav_files)} samples")
        print(f"Average LM loss : {avg_loss:.4f}")
        print(f"{'=' * 70}\n")
    else:
        print("No valid samples evaluated.")


# ════════════════════════════════════════════════════════════════════════════
# SECTION 2 — Real-time Microphone Chat
# ════════════════════════════════════════════════════════════════════════════

def record_mic(seconds: int, sample_rate: int) -> np.ndarray:
    """
    Records audio from the default microphone for `seconds` seconds.

    Requires: pip install sounddevice
    Returns: numpy float32 array at sample_rate
    """
    try:
        import sounddevice as sd
    except ImportError:
        print("  ERROR — sounddevice not installed.")
        print("  Install it with: pip install sounddevice")
        sys.exit(1)

    print(f"  Recording for {seconds}s... (speak now)")
    audio = sd.rec(
        int(seconds * sample_rate),
        samplerate = sample_rate,
        channels   = 1,
        dtype      = "float32",
    )
    sd.wait()
    print("  Recording done.")
    return audio.squeeze()   # (samples,)


def run_chat(model: SpeakMK1Multimodal, tokenizer, cfg: TestConfig, device: torch.device):
    """
    Real-time microphone chat loop.

    Each turn:
        1. Records cfg.mic_record_seconds seconds of audio
        2. Converts to mel
        3. Runs multimodal generation
        4. Prints the response

    Type 'quit' at the prompt to exit.
    Type 'save <path>' to save the last recording as a wav file.
    """
    print("\n" + "=" * 70)
    print("SECTION 2 — Real-time Microphone Chat")
    print("=" * 70)
    print(f"  Record duration : {cfg.mic_record_seconds}s per turn")
    print(f"  Max new tokens  : {cfg.max_new_tokens}")
    print(f"  Temperature     : {cfg.temperature}")
    print()
    print("  Commands:")
    print("    Enter          — start recording")
    print("    quit           — exit")
    print("    seconds <N>    — change recording duration to N seconds")
    print("    temp <F>       — change temperature (e.g. temp 0.5)")
    print("=" * 70 + "\n")

    last_audio = None
    record_secs = cfg.mic_record_seconds
    temperature = cfg.temperature

    while True:
        try:
            cmd = input("Press Enter to record (or type a command): ").strip().lower()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting chat.")
            break

        # ── Commands ──────────────────────────────────────────────────────
        if cmd == "quit":
            print("Goodbye.")
            break

        elif cmd.startswith("seconds "):
            try:
                record_secs = int(cmd.split()[1])
                print(f"  Recording duration set to {record_secs}s")
            except (IndexError, ValueError):
                print("  Usage: seconds <N>")
            continue

        elif cmd.startswith("temp "):
            try:
                temperature = float(cmd.split()[1])
                cfg.temperature = temperature
                print(f"  Temperature set to {temperature}")
            except (IndexError, ValueError):
                print("  Usage: temp <float>")
            continue

        elif cmd.startswith("save "):
            if last_audio is not None:
                try:
                    import soundfile as sf
                    path = cmd.split(" ", 1)[1]
                    sf.write(path, last_audio, cfg.mic_sample_rate)
                    print(f"  Saved to {path}")
                except Exception as e:
                    print(f"  Save failed: {e}")
            else:
                print("  No recording yet.")
            continue

        # ── Record ────────────────────────────────────────────────────────
        try:
            audio = record_mic(record_secs, cfg.mic_sample_rate)
            last_audio = audio
        except Exception as e:
            print(f"  Recording error: {e}")
            continue

        # ── Convert to mel ─────────────────────────────────────────────────
        try:
            mel = numpy_to_mel(audio, cfg, device)
        except Exception as e:
            print(f"  Mel conversion error: {e}")
            continue

        # ── Check audio is not silence ────────────────────────────────────
        rms = np.sqrt(np.mean(audio ** 2))
        if rms < 1e-4:
            print("  WARNING — audio appears to be silence (RMS too low).")
            print("  Check your microphone input.\n")

        # ── Generate response ─────────────────────────────────────────────
        print("  Generating response...")
        t0 = time.time()
        prompt = (
            "<|system|> You are SpeakMK1, a warm and clinically expert AI "
            "speech-language pathologist for children. "
            "The child is 6 years old and presents with Stopping (Stopping of /s/ and /z/ (s/z and t/d)). "
            "Clinical goal: Establish /s/ using airflow awareness. "
            "Primary strategy: Tactile Cueing (touching alveolar ridge awareness). "
            "Always analyze errors clinically before responding. Be encouraging, patient, and use scaffolding. "
            "<|endturn|>"
            "<|child|> I went to the tore yeterday <|endturn|>"
            "<|think|>"
        )
        try:
            response = generate_response(
                model, mel, tokenizer, cfg, device, prompt=prompt
            )
        except Exception as e:
            print(f"  Generation error: {e}")
            continue

        elapsed = time.time() - t0
        print(f"\n  SpeakMK1: {response}")
        print(f"  (generated in {elapsed:.1f}s)\n")


# ════════════════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="SpeakMK1 Multimodal Test")
    parser.add_argument(
        "--mode",
        choices=["eval", "chat", "both"],
        default="both",
        help="Which section to run: eval (LibriSpeech), chat (microphone), or both",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default=None,
        help="Override checkpoint path (combined multimodal .pt file)",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=None,
        help="Override max eval samples",
    )
    args = parser.parse_args()

    cfg    = TestConfig()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.ckpt:
        cfg.multimodal_ckpt = args.ckpt
    if args.samples:
        cfg.max_eval_samples = args.samples

    print("=" * 70)
    print("SpeakMK1 Multimodal — Test")
    print(f"  device  : {device}")
    print(f"  mode    : {args.mode}")
    print(f"  ckpt    : {cfg.multimodal_ckpt}")
    print("=" * 70)

    # ── Load tokeniser ────────────────────────────────────────────────────
    print(f"\nLoading tokeniser: {cfg.tokenizer_name}")
    tokenizer = AutoTokenizer.from_pretrained(cfg.tokenizer_name)
    tokenizer.add_special_tokens({"pad_token": "<|pad|>"})
    tokenizer.add_special_tokens({
        "additional_special_tokens": [
            "<|system|>", "<|child|>", "<|think|>", "<|slp|>",
            "<|endturn|>", "<|endseq|>", "<|ctc_blank|>"
        ]
    })
    special_tokens = ["<|system|>", "<|child|>", "<|think|>", "<|slp|>", "<|endturn|>", "<|endseq|>"]
    for tok in special_tokens:
        tid = tokenizer.convert_tokens_to_ids(tok)
        print(f"{tok} -> {tid}  (unk={tokenizer.unk_token_id})")
        assert tid != tokenizer.unk_token_id, f"{tok} not recognized!"
    native = ["<|endoftext|>", "<|padding|>"]
    for tok in native:
        tid = tokenizer.convert_tokens_to_ids(tok)
        print(f"{tok} -> {tid} (native)")
    print(f"len(tokenizer) = {len(tokenizer)}")
    print(f"  vocab_size : {tokenizer.vocab_size}")

    # ── Load model ────────────────────────────────────────────────────────
    print("\nLoading model...")
    model = load_model(cfg, device)
    print("Audio encoder param sample:", 
      model.audio_encoder.stem[0].weight[0, :5])
    print("Audio encoder stem weight norm:", 
        model.audio_encoder.stem[0].weight.norm().item())
    print(f"model embedding size: {model.llm.embedding.weight.shape}")
    print(f"model lm_head size: {model.llm.lm_head.weight.shape}")
    print(f"len(tokenizer): {len(tokenizer)}")
    base_norm = model.llm.embedding.weight[:10].norm(dim=1).mean().item()
    spec_norm = model.llm.embedding.weight[50278:50285].norm(dim=1).mean().item()
    print(f"Base token embed norm: {base_norm:.3f}")
    print(f"Special token embed norm: {spec_norm:.3f}")

    if spec_norm < base_norm * 0.5:
        print("WARNING: Special token embeddings look random — checkpoint may not include them!")

    # ── Run sections ──────────────────────────────────────────────────────
    if args.mode in ("eval", "both"):
        run_librispeech_eval(model, tokenizer, cfg, device)

    if args.mode in ("chat", "both"):
        run_chat(model, tokenizer, cfg, device)


if __name__ == "__main__":
    main()