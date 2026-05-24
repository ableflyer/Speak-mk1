"""
precompute_audio_feats.py
=========================
Precomputes audio encoder features and GPT-NeoX token IDs for training.
Run this once before train_audio_proj.py.

Output: ../Data/audio_feats_cache_neox/batch_XXXXX.pt
Each file contains:
    feats              : (B, T_audio, 512)   — encoder features
    token_ids          : (B, max_text_len)   — GPT-NeoX token IDs
    audio_padding_mask : (B, T_audio)        — True = padded frame
"""

import os
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from audio_encoder import AudioEncoder
from audio_trainer import SmallConfig
from librispeech_byte_dataset import LibriSpeechByteDataset, collate_fn

# ════════════════════════════════════════════════════════════════════════════
# CONFIG  — edit these to match your setup
# ════════════════════════════════════════════════════════════════════════════
LIBRISPEECH_ROOT = "../Data/PhonemeDatasets/LibriSpeech/LibriSpeech/train-clean-100"
AUDIO_CKPT       = "../Model_files/Audio_encoder_v1.1/audio_encoder_epoch_5.pt"
CACHE_DIR        = "../Data/audio_feats_cache_neox"
TOKENIZER_NAME   = "EleutherAI/gpt-neox-20b"
MAX_FRAMES       = 512
MAX_TEXT_LEN     = 512
BATCH_SIZE       = 4
NUM_WORKERS      = 2
# ════════════════════════════════════════════════════════════════════════════


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(CACHE_DIR, exist_ok=True)

    # ── Tokeniser ────────────────────────────────────────────────────────
    print(f"Loading tokeniser: {TOKENIZER_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
    tokenizer.pad_token = tokenizer.eos_token
    pad_id = tokenizer.pad_token_id
    print(f"  vocab_size : {tokenizer.vocab_size}")
    print(f"  pad_token  : '{tokenizer.pad_token}'  (id={pad_id})")

    # ── Audio encoder ─────────────────────────────────────────────────────
    print(f"\nLoading audio encoder: {AUDIO_CKPT}")
    cfg_audio = SmallConfig(d_model=512, llm_dim=4096)
    encoder   = AudioEncoder(cfg_audio).to(device).eval()
    state     = torch.load(AUDIO_CKPT, map_location=device)
    state     = {k: v for k, v in state.items() if not k.startswith("qformer")}
    missing, unexpected = encoder.load_state_dict(state, strict=False)
    non_qf = [k for k in missing if "qformer" not in k]
    if non_qf:
        print(f"  WARNING — non-qformer missing keys: {non_qf}")
    print(f"  missing={len(missing)}  unexpected={len(unexpected)}")

    # ── Dataset ───────────────────────────────────────────────────────────
    # We use LibriSpeechByteDataset only for mel + transcript path lookup.
    # Token IDs are re-encoded here with GPT-NeoX, ignoring the byte_ids
    # that the dataset returns.
    print(f"\nIndexing dataset: {LIBRISPEECH_ROOT}")
    dataset = LibriSpeechByteDataset(
        librispeech_root = LIBRISPEECH_ROOT,
        max_frames       = MAX_FRAMES,
        max_text_len     = MAX_TEXT_LEN,
    )
    # collate_fn returns (mels, byte_ids) — we replace byte_ids below
    loader = DataLoader(
        dataset,
        batch_size  = BATCH_SIZE,
        shuffle     = False,
        collate_fn  = collate_fn,
        num_workers = NUM_WORKERS,
        pin_memory  = True,
    )
    print(f"  {len(dataset)} samples  →  {len(loader)} batches of {BATCH_SIZE}")

    # ── Precompute loop ───────────────────────────────────────────────────
    print(f"\nPrecomputing features → {CACHE_DIR}\n")
    sample_ptr = 0   # tracks which transcript corresponds to each mel

    with torch.no_grad():
        for i, (mel, token_ids) in enumerate(loader):
            B   = mel.shape[0]
            mel = mel.to(device)
            feats = encoder.encode_features(mel)
            audio_padding_mask = (mel.sum(dim=-1) == 0)

            torch.save({
                "feats"             : feats.cpu(),
                "token_ids"         : token_ids.cpu(),
                "audio_padding_mask": audio_padding_mask.cpu(),
            }, os.path.join(CACHE_DIR, f"batch_{i:05d}.pt"))
            
            if i == 0:
                print(f"  first batch token_ids max={token_ids.max().item()}")
                assert token_ids.max().item() < 50283, "Token ID exceeds GPT-NeoX vocab size — check TOKENIZER_NAME and dataset tokenisation!"
            
            if i % 100 == 0:
                frac_pad = (token_ids == pad_id).float().mean().item()
                print(f"  [{i:>5}/{len(loader)}]  "
                      f"feats={tuple(feats.shape)}  "
                      f"token_ids={tuple(token_ids.shape)}  "
                      f"max_id={token_ids.max().item()}  "
                      f"pad_frac={frac_pad:.3f}")

    print(f"\nDone. {len(loader)} batches written to {CACHE_DIR}")
    print("Update ProjTrainConfig.cache_dir to point to this directory.")


if __name__ == "__main__":
    main()