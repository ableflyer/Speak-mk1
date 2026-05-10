"""
train_stage5_audio.py
=====================
Stage 5: Audio-conditioned fine-tuning.
Takes stage 4 checkpoint, keeps LLM trainable, injects random cached
audio features so the model learns to be stable with audio in the loop.
Gates are forced open and allowed to train.

Run:
    python train_stage5_audio.py --resume ../Model_files/checkpoints_v2.3.1/stage4/ckpt_final.pt
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

try:
    from bitsandbytes.optim import AdamW8bit
    HAS_BNB = True
except ImportError:
    HAS_BNB = False

from speak_mk1_llm import SpeakMK1LLM, SpeakMK1LLMConfig
from train_proj import DirectAudioProjection
from audio_encoder import AudioEncoder
from audio_trainer import SmallConfig
import LatentMoE

# ════════════════════════════════════════════════════════════════════════════
# CONFIG
# ════════════════════════════════════════════════════════════════════════════

CKPT_DIR      = Path("../Model_files/checkpoints_v2.3.1/stage5_audio")
DATA_DIR      = Path("../Data/LLM_Data_updated")
AUDIO_CACHE   = "../Data/audio_feats_cache_neox"
PROJ_CKPT     = "../Model_files/audio_proj_training/audio_proj_best.pt"
ENC_CKPT      = "../Model_files/Audio_encoder_v1.1/audio_encoder_epoch_5.pt"

TRAIN_STEPS   = 3_000
LR            = 5e-6
WARMUP_STEPS  = 150
GRAD_ACCUM    = 4
BATCH_SIZE    = 4      # smaller — audio eats VRAM
SEQ_LEN       = 512
MAX_GRAD_NORM = 0.01
LOG_EVERY     = 50
SAVE_EVERY    = 1_000

SPECIAL_TOKENS = {
    "system_start": "<|system|>",
    "child_start":  "<|child|>",
    "think_start":  "<|think|>",
    "slp_start":    "<|slp|>",
    "turn_end":     "<|endturn|>",
    "seq_end":      "<|endseq|>",
}

SYSTEM_PROMPT_TEMPLATE = (
    "You are SpeakMK1, a warm and clinically expert AI speech-language pathologist for children. "
    "The child is {age} years old and presents with {disorder} ({specific_error}). "
    "Clinical goal: {clinical_goal}. "
    "Primary strategy: {strategy}. "
    "Always analyze errors clinically before responding. Be encouraging, patient, and use scaffolding."
)


# ════════════════════════════════════════════════════════════════════════════
# AUDIO CACHE POOL
# ════════════════════════════════════════════════════════════════════════════

class AudioCachePool:
    """
    Loads all cached audio feature files into a pool.
    At training time, randomly samples a batch of audio features
    to pair with each text batch. This is intentionally mismatched —
    we just want the model to be stable with audio in the loop.
    """
    def __init__(self, cache_dir: str, device: torch.device):
        files = sorted(glob.glob(os.path.join(cache_dir, "batch_*.pt")))
        if not files:
            raise RuntimeError(f"No cached audio files found in {cache_dir}")
        self.files  = files
        self.device = device
        print(f"  AudioCachePool: {len(files)} cached batches")

    def sample(self, batch_size: int):
        """Returns (feats, mask) with shape (batch_size, T, 512)."""
        f = random.choice(self.files)
        d = torch.load(f, map_location="cpu", weights_only=False)
        feats = d["feats"]       # (B_file, T, 512)
        mask  = d["audio_padding_mask"]  # (B_file, T)

        # Subsample or repeat to match batch_size
        idx   = torch.randint(0, feats.shape[0], (batch_size,))
        feats = feats[idx].to(self.device)
        mask  = mask[idx].to(self.device)
        return feats, mask


# ════════════════════════════════════════════════════════════════════════════
# DATASET  (same SLPDataset as stage 4)
# ════════════════════════════════════════════════════════════════════════════

class SLPDataset(Dataset):
    def __init__(self, seq_len: int = 512, repeat: int = 30):
        ids_path    = DATA_DIR / "stage4_slp_ids.bin"
        labels_path = DATA_DIR / "stage4_slp_labels.bin"
        if not ids_path.exists():
            raise RuntimeError(
                f"SLP data not found at {ids_path}.\n"
                "Run: python train_speakmk1_llm.py tokenize_slp --path <your.jsonl> --stage 4"
            )
        self.ids     = np.memmap(str(ids_path),    dtype=np.uint16, mode="r")
        self.labels  = np.memmap(str(labels_path), dtype=np.int32,  mode="r")
        self.seq_len = seq_len
        self.n_seqs  = (len(self.ids) - 1) // seq_len
        self.repeat  = repeat
        print(f"  SLPDataset: {len(self.ids):,} tokens | {self.n_seqs:,} seqs | repeat={repeat}")

    def __len__(self):
        return self.n_seqs * self.repeat

    def __getitem__(self, idx):
        idx   = idx % self.n_seqs
        start = idx * self.seq_len
        x = torch.from_numpy(self.ids  [start:start + self.seq_len    ].astype(np.int64))
        y = torch.from_numpy(self.labels[start + 1:start + self.seq_len + 1].astype(np.int64))
        return x, y


# ════════════════════════════════════════════════════════════════════════════
# LR SCHEDULE
# ════════════════════════════════════════════════════════════════════════════

def get_lr(step: int, warmup: int, total: int, max_lr: float) -> float:
    if step < warmup:
        return max_lr * (step + 1) / warmup
    if step >= total:
        return max_lr * 0.01
    progress = (step - warmup) / (total - warmup)
    return max_lr * 0.01 + (max_lr - max_lr * 0.01) * 0.5 * (1 + math.cos(math.pi * progress))


# ════════════════════════════════════════════════════════════════════════════
# MAIN TRAINING
# ════════════════════════════════════════════════════════════════════════════

def train(resume: str):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    CKPT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  Stage 5: Audio-conditioned fine-tuning")
    print(f"  Device : {device}")
    print(f"{'='*60}\n")

    # ── Load LLM ──────────────────────────────────────────────────────────
    print("Loading LLM...")
    model_cfg = SpeakMK1LLMConfig(
        vocab_size=50283, d_model=512, d_state=64, num_blocks=6,
        nheads_ssm=8, nheads_attn=8, top_k_audio=32,
        num_experts=4, top_k_experts=2, dropout=0.0, aux_loss_weight=1e-2,
    )
    llm = SpeakMK1LLM(model_cfg).to(device)
    ckpt = torch.load(resume, map_location=device, weights_only=False)
    llm.load_state_dict(ckpt["model"], strict=True)
    
    # freeze everything first
    for name, param in llm.named_parameters():
        param.requires_grad = False
        
    # Unfreeze cross_attn entirely
    for name, param in llm.named_parameters():
        if 'cross_attn' in name:
            param.requires_grad = True

    # Unfreeze LLM body EXCEPT routers
    for name, param in llm.named_parameters():
        if 'cross_attn' not in name and 'router' not in name:
            param.requires_grad = True
    print(f"  Loaded from {resume}  (stage={ckpt.get('stage')}, step={ckpt.get('step')})")

    # Force all cross-attention gates open and keep them trainable
    with torch.no_grad():
        for block in llm.blocks:
            if hasattr(block.cross_attn, 'gate'):
                block.cross_attn.gate.data.fill_(0.01)
    n_gates = sum(1 for b in llm.blocks if hasattr(b.cross_attn, 'gate'))
    print(f"  Opened {n_gates} cross-attention gates")
    
    trainable = [(n, p) for n, p in llm.named_parameters() if p.requires_grad]
    print(f"  Trainable params: {len(trainable)}")
    for n, p in trainable:
        print(f"    {n}: {p.shape}")

    # ── Load projection (frozen) ───────────────────────────────────────────
    print("Loading audio projection (frozen)...")
    proj = DirectAudioProjection(512, 512).to(device)
    proj_ckpt = torch.load(PROJ_CKPT, map_location=device, weights_only=False)
    proj.load_state_dict(proj_ckpt["audio_proj"])
    for p in proj.parameters():
        p.requires_grad = False
    proj.eval()
    print(f"  Projection frozen.")

    # ── Data ──────────────────────────────────────────────────────────────
    dataset    = SLPDataset(seq_len=SEQ_LEN, repeat=30)
    loader     = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True,
                            num_workers=2, pin_memory=True, drop_last=True)
    loader_iter = iter(loader)
    audio_pool  = AudioCachePool(AUDIO_CACHE, device)

    # ── Optimizer — LLM trainable, proj frozen ────────────────────────────
    trainable = [p for p in llm.parameters() if p.requires_grad]
    print(f"  Trainable params: {sum(p.numel() for p in trainable)/1e6:.2f}M")

    if HAS_BNB:
        optimizer = AdamW8bit(trainable, lr=LR, betas=(0.9, 0.95),
                              weight_decay=0.1, eps=1e-8)
        print("  Optimizer: AdamW8bit")
    else:
        optimizer = torch.optim.AdamW(trainable, lr=LR, betas=(0.9, 0.95),
                                      weight_decay=0.1, eps=1e-8)
        print("  Optimizer: AdamW")

    # ── Training loop ─────────────────────────────────────────────────────
    print(f"\n  Training for {TRAIN_STEPS:,} steps\n")
    llm.train()
    optimizer.zero_grad()

    best_loss  = float("inf")
    loss_acc   = 0.0
    aux_acc    = 0.0
    t0         = time.time()

    for step in range(TRAIN_STEPS):
        lr = get_lr(step, WARMUP_STEPS, TRAIN_STEPS, LR)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        for micro in range(GRAD_ACCUM):
            try:
                x, y = next(loader_iter)
            except StopIteration:
                loader_iter = iter(loader)
                x, y = next(loader_iter)

            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            # Sample random audio from cache — mismatched is fine,
            # we just need the model to handle audio-conditioned forward passes
            feats, audio_mask = audio_pool.sample(x.shape[0])

            with torch.no_grad():
                audio_out = proj(feats)  # (B, T_audio, 512), no grad through proj

            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                logits, ce_loss, aux_loss = llm(
                    input_ids          = x,
                    labels             = y,
                    audio_out          = audio_out,
                    audio_padding_mask = audio_mask,
                )
                total_loss = ce_loss / GRAD_ACCUM

            total_loss.backward()
            loss_acc += ce_loss.item()
            aux_acc  += aux_loss.item()

        grad_norm = nn.utils.clip_grad_norm_(trainable, MAX_GRAD_NORM)
        optimizer.step()
        optimizer.zero_grad()

        if step % LOG_EVERY == 0:
            n = LOG_EVERY if step > 0 else 1
            avg_loss = loss_acc / (n * GRAD_ACCUM)
            avg_aux  = aux_acc  / (n * GRAD_ACCUM)
            elapsed  = time.time() - t0
            print(f"  step {step:>5} | loss {avg_loss:.4f} | aux {avg_aux:.5f} | "
                  f"lr {lr:.2e} | grad {grad_norm:.3f} | {elapsed:.0f}s")
            loss_acc = 0.0
            aux_acc  = 0.0
            t0 = time.time()

            if step > 0 and avg_loss < best_loss:
                best_loss = avg_loss
                _save(llm, optimizer, step, best_loss, CKPT_DIR / "ckpt_best.pt")

        if step % SAVE_EVERY == 0 and step > 0:
            _save(llm, optimizer, step, best_loss, CKPT_DIR / f"ckpt_step{step}.pt")
            print(f"  Saved step {step}")

    _save(llm, optimizer, TRAIN_STEPS, best_loss, CKPT_DIR / "ckpt_final.pt")
    print(f"\n  Done. Final checkpoint → {CKPT_DIR / 'ckpt_final.pt'}")
    print(f"  Best loss: {best_loss:.4f}")


def _save(model, optimizer, step, best_loss, path):
    torch.save({
        "model":     model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "step":      step,
        "best_loss": best_loss,
        "stage":     5,
    }, path)


# ════════════════════════════════════════════════════════════════════════════
# ENTRYPOINT
# ════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", type=str,
                        default="../Model_files/checkpoints_v2.3.1/stage4/ckpt_final.pt",
                        help="Stage 4 checkpoint to start from")
    args = parser.parse_args()
    train(args.resume)