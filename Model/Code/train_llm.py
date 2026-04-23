"""
train_speakmk1_llm.py
=====================
SpeakMK1 LLM — Training script + CLI chatbot.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
import gc
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
import torch.serialization
import threading
import copy
import torch._dynamo
from torch.utils.data import ConcatDataset as TorchConcat

# ── Try to import bitsandbytes for 8-bit Adam ────────────────────────────────
try:
    from bitsandbytes.optim import Adam8bit, AdamW8bit
    HAS_BNB = True
except ImportError:
    HAS_BNB = False

# ── Your model ───────────────────────────────────────────────────────────────
from speak_mk1_llm import SpeakMK1LLM, SpeakMK1LLMConfig, count_parameters
import LatentMoE

# ════════════════════════════════════════════════════════════════════════════
# 1.  PATHS AND STAGE CONFIG
# ════════════════════════════════════════════════════════════════════════════

DATA_DIR  = Path("./../Data/LLM_Data_updated")
CKPT_DIR  = Path("./../Model_files/checkpoints_v2")
LOG_DIR   = Path("./../Data/logs_v2")

STAGE_CONFIG = {
    1: {
        "name": "general_pretraining",
        "datasets": [
            {
                "id":      "tinystories",
                "hf_path": "roneneldan/TinyStories",
                "hf_name": None,
                "split":   "train",
                "text_key": "text",
                "max_docs": 2_000_000,
                "streaming": False,
            }
        ],
        "train_steps":  50_000,
        "lr":           6e-4,
        "warmup_steps": 2_000,
        "seq_len":      512,
        "batch_size":   4,
        "grad_accum":   8,
    },
    2: {
        "name": "child_directed_adapt",
        "datasets": [
            # {
            #     "id":      "tinystories",
            #     "hf_path": "roneneldan/TinyStories",
            #     "hf_name": None,
            #     "split":   "train",
            #     "text_key": "text",
            #     "max_docs": 500_000,
            #     "streaming": False,
            #     "weight":  0.8,
            # },
            {
                "id":      "childes",
                "local":   True,
            }
        ],
        "train_steps":  30_000,
        "lr":           1e-4,
        "warmup_steps": 500,
        "seq_len":      512,
        "batch_size":   4,
        "grad_accum":   8,
    },
    3: {
        "name": "clinical_injection",
        "datasets": [],
        "train_steps":  5_000,
        "lr":           5e-5,
        "warmup_steps": 200,
        "seq_len":      512,
        "batch_size":   4,
        "grad_accum":   4,
    },
    4: {
        "name": "instruction_tuning",
        "datasets": [
            {
                "id":      "alpaca",
                "hf_path": "tatsu-lab/alpaca",
                "hf_name": None,
                "split":   "train",
                "text_key": "text",
                "max_docs": 52_000,
                "streaming": False,
            },
        ],
        "train_steps":  3_000,
        "lr":           2e-5,
        "warmup_steps": 100,
        "seq_len":      512,
        "batch_size":   4,
        "grad_accum":   4,
    },
}


# ════════════════════════════════════════════════════════════════════════════
# 2.  TOKENIZER
# ════════════════════════════════════════════════════════════════════════════

def get_tokenizer():
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


# ════════════════════════════════════════════════════════════════════════════
# 3.  TOKENIZATION
# ════════════════════════════════════════════════════════════════════════════

def tokenize_childes(childes_root: str, stage: int = 2):
    import re
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    bin_path = DATA_DIR / f"stage{stage}_childes.bin"

    if bin_path.exists():
        size_mb = bin_path.stat().st_size / 1e6
        print(f"  [childes] already exists ({size_mb:.1f} MB) — skipping.")
        return

    tok = get_tokenizer()
    eos = tok.eos_token_id
    root = Path(childes_root)

    cha_files = list(root.rglob("*.cha"))
    print(f"  [childes] Found {len(cha_files)} .cha files in {childes_root}")

    if not cha_files:
        print(f"  [childes] WARNING: No .cha files found. Check your path.")
        return

    tokens_done = 0
    docs_done   = 0
    t0 = time.time()

    with open(bin_path, "wb") as fout:
        for cha_path in cha_files:
            try:
                lines = cha_path.read_text(encoding="utf-8", errors="ignore").splitlines()
            except Exception:
                continue

            utterances = []
            for line in lines:
                if line.startswith("*"):
                    text = re.sub(r"^\*[A-Z]+:\t", "", line)
                    text = re.sub(r"\[.*?\]", "", text)
                    text = re.sub(r"<.*?>", "", text)
                    text = re.sub(r"[+/\\]", "", text)
                    text = re.sub(r"[&@]\w+", "", text)
                    text = text.strip()
                    if text and len(text) > 3:
                        utterances.append(text)

            if not utterances:
                continue

            document = " ".join(utterances)
            if not document.strip():
                continue

            ids = tok.encode(document, add_special_tokens=False)
            ids.append(eos)
            fout.write(np.array(ids, dtype=np.uint16).tobytes())
            tokens_done += len(ids)
            docs_done   += 1

    elapsed = time.time() - t0
    print(f"  [childes] Done. {docs_done} files | "
          f"{tokens_done/1e6:.2f}M tokens | "
          f"{bin_path.stat().st_size/1e6:.1f} MB | "
          f"{elapsed:.0f}s")


def tokenize_stage(stage: int):
    from datasets import load_dataset as hf_load

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    cfg = STAGE_CONFIG[stage]
    tok = get_tokenizer()
    eos = tok.eos_token_id

    print(f"\n{'='*60}")
    print(f"  Tokenizing Stage {stage}: {cfg['name']}")
    print(f"{'='*60}\n")

    for ds_cfg in cfg["datasets"]:
        ds_id   = ds_cfg["id"]
        bin_path = DATA_DIR / f"stage{stage}_{ds_id}.bin"

        if bin_path.exists():
            size_mb = bin_path.stat().st_size / 1e6
            print(f"  [{ds_id}] already exists ({size_mb:.1f} MB) — skipping.")
            continue

        # Skip local-only datasets (no hf_path)
        if ds_cfg.get("local", False):
            print(f"  [{ds_id}] Local dataset — tokenize separately with tokenize_childes() or tokenize_local_jsonl()")
            continue

        print(f"  [{ds_id}] Loading from HuggingFace …")
        ds = hf_load(
            ds_cfg["hf_path"],
            name=ds_cfg["hf_name"],
            split=ds_cfg["split"],
            streaming=ds_cfg["streaming"],
            trust_remote_code=True,
        )

        text_key = ds_cfg["text_key"]
        max_docs = ds_cfg.get("max_docs", None)

        if ds_cfg["streaming"] and max_docs:
            ds = ds.take(max_docs)

        print(f"  [{ds_id}] Tokenizing → {bin_path} …")
        t0 = time.time()

        with open(bin_path, "wb") as f:
            docs_done   = 0
            tokens_done = 0

            for example in ds:
                if max_docs and not ds_cfg["streaming"] and docs_done >= max_docs:
                    break

                text = example.get(text_key, "")
                if not text or not text.strip():
                    continue

                text = text[:10_000]
                ids = tok.encode(text, add_special_tokens=False)
                ids.append(eos)

                arr = np.array(ids, dtype=np.uint16)
                n_tokens = len(ids)
                f.write(arr.tobytes())
                del arr, ids

                docs_done   += 1
                tokens_done += n_tokens

                if docs_done % 10_000 == 0:
                    gc.collect()

                if docs_done % 100_000 == 0:
                    elapsed = time.time() - t0
                    size_gb = (tokens_done * 2) / 1e9
                    print(f"    {docs_done:>8,} docs | "
                        f"{tokens_done/1e6:>7.1f}M tokens | "
                        f"{size_gb:.2f} GB | "
                        f"{elapsed:.0f}s elapsed")


def tokenize_local_jsonl(jsonl_path: str, stage: int, dataset_id: str):
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    tok      = get_tokenizer()
    eos      = tok.eos_token_id
    bin_path = DATA_DIR / f"stage{stage}_{dataset_id}.bin"

    if bin_path.exists():
        print(f"  [{dataset_id}] already exists — skipping.")
        return

    print(f"  Tokenizing {jsonl_path} → {bin_path} …")
    tokens_done = 0

    with open(jsonl_path, "r") as fin, open(bin_path, "wb") as fout:
        for line in fin:
            obj  = json.loads(line)
            text = obj.get("text", "")
            if not text.strip():
                continue
            ids = tok.encode(text, add_special_tokens=False)
            ids.append(eos)
            fout.write(np.array(ids, dtype=np.uint16).tobytes())
            tokens_done += len(ids)

    print(f"  Done. {tokens_done/1e6:.2f}M tokens | "
          f"{bin_path.stat().st_size/1e6:.1f} MB")


# ════════════════════════════════════════════════════════════════════════════
# 4.  DATASET
# ════════════════════════════════════════════════════════════════════════════

class PackedTokenDataset(Dataset):
    def __init__(self, bin_path: str, seq_len: int):
        self.seq_len = seq_len
        self.data    = np.memmap(bin_path, dtype=np.uint16, mode="r")
        self.n_seqs  = (len(self.data) - 1) // seq_len
        print(f"  Loaded {bin_path}")
        print(f"    Tokens : {len(self.data):,}")
        print(f"    Seqs   : {self.n_seqs:,}  (seq_len={seq_len})")

    def __len__(self):
        return self.n_seqs

    def __getitem__(self, idx: int):
        start = idx * self.seq_len
        chunk = self.data[start : start + self.seq_len + 1].astype(np.int64)
        x = torch.from_numpy(chunk[:-1])
        y = torch.from_numpy(chunk[1:])
        return x, y


class ConcatDataset(Dataset):
    def __init__(self, datasets: List[PackedTokenDataset]):
        self.datasets = datasets
        self.lengths  = [len(d) for d in datasets]
        self.total    = sum(self.lengths)
        self.cumsum   = np.cumsum([0] + self.lengths)

    def __len__(self):
        return self.total

    def __getitem__(self, idx: int):
        ds_idx = np.searchsorted(self.cumsum[1:], idx, side='right')
        local  = idx - self.cumsum[ds_idx]
        return self.datasets[ds_idx][local]


def build_dataloader(stage: int, seq_len: int, batch_size: int) -> DataLoader:
    stage_datasets = []
    weights = []

    for ds_cfg in STAGE_CONFIG[stage]["datasets"]:
        ds_id = ds_cfg["id"]
        bin_path = DATA_DIR / f"stage{stage}_{ds_id}.bin"
        
        if not bin_path.exists():
            if ds_cfg.get("local", False):
                print(f"  WARNING: {bin_path} not found — skipping local dataset {ds_id}")
                continue
            print(f"  WARNING: {bin_path} not found — skipping."
                  f" Run: python train_speakmk1_llm.py tokenize --stage {stage}")
            continue
        
        ds = PackedTokenDataset(str(bin_path), seq_len)
        stage_datasets.append(ds)
        weights.append(ds_cfg.get("weight", 1.0))
        print(f"  [{ds_id}] {len(ds):,} sequences (weight={weights[-1]})")

    # Pick up manually added local files
    for extra in DATA_DIR.glob(f"stage{stage}_*.bin"):
        name = extra.stem.replace(f"stage{stage}_", "")
        if not any(d["id"] == name for d in STAGE_CONFIG[stage]["datasets"]):
            print(f"  Found extra dataset: {extra.name}")
            ds = PackedTokenDataset(str(extra), seq_len)
            stage_datasets.append(ds)
            weights.append(1.0)

    if not stage_datasets:
        raise RuntimeError(
            f"No data found for stage {stage}. "
            f"Run: python train_speakmk1_llm.py tokenize --stage {stage}"
        )

    combined = ConcatDataset(stage_datasets)
    print(f"\n  Total sequences for stage {stage}: {len(combined):,}")

    sampler = None
    if len(stage_datasets) > 1 and any(w != 1.0 for w in weights):
        # Build per-sample weights for WeightedRandomSampler
        sample_weights = []
        for ds, w in zip(stage_datasets, weights):
            sample_weights.extend([w] * len(ds))
        sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)
        print(f"  Using weighted sampling with weights: {weights}")

    return DataLoader(
        combined,
        batch_size=batch_size,
        shuffle=(sampler is None),
        sampler=sampler,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
        persistent_workers=True,
        prefetch_factor=2,
    )


# ════════════════════════════════════════════════════════════════════════════
# 5.  LEARNING RATE SCHEDULE
# ════════════════════════════════════════════════════════════════════════════

def get_lr(step: int, warmup_steps: int, total_steps: int, max_lr: float,
           min_lr_ratio: float = 0.01) -> float:
    min_lr = max_lr * min_lr_ratio

    if step < warmup_steps:
        return max_lr * (step + 1) / warmup_steps

    if step >= total_steps:
        return min_lr

    progress = (step - warmup_steps) / (total_steps - warmup_steps)
    cosine   = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_lr + (max_lr - min_lr) * cosine


# ════════════════════════════════════════════════════════════════════════════
# 6.  TRAINING LOOP
# ════════════════════════════════════════════════════════════════════════════

def train(stage: int, resume: Optional[str] = None):
    cfg      = STAGE_CONFIG[stage]
    ckpt_dir = CKPT_DIR / f"stage{stage}"
    log_dir  = LOG_DIR  / f"stage{stage}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = True
    torch.backends.cudnn.benchmark = True
    print(f"\n{'='*60}")
    print(f"  Stage {stage}: {cfg['name']}")
    print(f"  Device: {device}")
    if device.type == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")
    print(f"{'='*60}\n")

    # ── Model ─────────────────────────────────────────────────────────────
    model_cfg = SpeakMK1LLMConfig(
        vocab_size=50277,
        d_model=512,
        d_state=64,
        # FIXED: Use num_blocks, not num_outer_blocks/num_inner_repeats
        num_blocks=6,
        nheads_ssm=8,
        nheads_attn=8,
        top_k_audio=32,
        num_experts=4,
        top_k_experts=2,
        dropout=0.0,
        aux_loss_weight=1e-2,
    )
    model = SpeakMK1LLM(model_cfg).to(device)
    
    # Optional: torch.compile only on CUDA
    # if device.type == "cuda":
    #     model = torch.compile(model, mode="default", fullgraph=False)
    #     torch._dynamo.config.cache_size_limit = 64
    #     torch._dynamo.config.suppress_errors = True

    params = count_parameters(model)
    print(f"  Model parameters: {params['total_M']}M total | {params['trainable_M']}M trainable")

    # ── Optimizer ─────────────────────────────────────────────────────────
    # FIXED: Router LR multiplier reduced from 10x to 1.5x for stability
    # MoE routers are sensitive to aggressive LR scaling [^2^]
    ROUTER_LR_MULT = 5.0
    
    if HAS_BNB:
        print(f"  Optimizer: AdamW8bit (bitsandbytes) — saves ~2 GB VRAM")
        print(f"  Router LR multiplier: {ROUTER_LR_MULT}x")
        optimizer = AdamW8bit([
            {"params": [p for n, p in model.named_parameters() 
                        if "router" not in n], "lr": cfg["lr"]},
            {"params": [p for n, p in model.named_parameters() 
                        if "router" in n], "lr": cfg["lr"] * ROUTER_LR_MULT},
        ], betas=(0.9, 0.95), weight_decay=0.1)
    else:
        print("  Optimizer: AdamW (install bitsandbytes for 8-bit Adam)")
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=cfg["lr"],
            betas=(0.9, 0.95),
            weight_decay=0.1,
            eps=1e-8,
            fused=True,
        )

    # ── Timing test ───────────────────────────────────────────────────────
    if device.type == "cuda":
        print("\n  Running timing test …")
        model.train()
        x_test = torch.randint(0, 50277, (4, 512), device=device)
        y_test = torch.randint(0, 50277, (4, 512), device=device)

        # Warmup
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            logits, ce_loss, aux_loss = model(x_test, labels=y_test)
            total = (ce_loss + model_cfg.aux_loss_weight * aux_loss) / cfg["grad_accum"]
        total.backward()
        optimizer.zero_grad()
        torch.cuda.synchronize()

        # Time 5 passes
        t0 = time.time()
        for _ in range(5):
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                logits, ce_loss, aux_loss = model(x_test, labels=y_test)
                total = (ce_loss + model_cfg.aux_loss_weight * aux_loss) / cfg["grad_accum"]
            total.backward()
            optimizer.zero_grad()
        torch.cuda.synchronize()
        t1 = time.time()

        ms = (t1 - t0) / 5 * 1000
        print(f"  Forward+backward : {ms:.0f}ms per micro-step")
        print(f"  Per full step    : {ms * cfg['grad_accum'] / 1000:.1f}s")
        print(f"  Projected speed  : {86400 / (ms * cfg['grad_accum'] / 1000):.0f} steps/day")

    # ── Resume from checkpoint ─────────────────────────────────────────────
    start_step   = 0
    best_loss    = float("inf")

    if resume:
        print(f"  Resuming from {resume} …")
        torch.serialization.add_safe_globals([SpeakMK1LLMConfig])
        ckpt = torch.load(resume, map_location=device)
        
        model.load_state_dict(ckpt["model"])
        
        saved_stage = ckpt.get("stage", None)
        saved_step = ckpt["step"]
        
        if saved_stage is None:
            if saved_step >= cfg["train_steps"]:
                print(f"  Detected checkpoint from previous stage (step {saved_step} > {cfg['train_steps']})")
                print(f"  Starting Stage {stage} training from step 0 with transferred weights")
                start_step = 0
            else:
                start_step = saved_step
                optimizer.load_state_dict(ckpt["optimizer"])
                best_loss = ckpt.get("best_loss", float("inf"))
                
        elif saved_stage != stage:
            print(f"  Transferring from Stage {saved_stage} to Stage {stage}")
            start_step = 0
            print(f"  Fresh optimizer for Stage {stage} (LR={cfg['lr']})")
            
        else:
            start_step = saved_step
            optimizer.load_state_dict(ckpt["optimizer"])
            best_loss = ckpt.get("best_loss", float("inf"))
            
        print(f"  Starting at step {start_step}")

    # ── Dataloader ─────────────────────────────────────────────────────────
    loader = build_dataloader(stage, cfg["seq_len"], cfg["batch_size"])
    loader_iter = iter(loader)

    # ── Log file ──────────────────────────────────────────────────────────
    log_path = log_dir / "train_log.jsonl"
    log_file = open(log_path, "a")

    # ── Training loop ─────────────────────────────────────────────────────
    total_steps   = cfg["train_steps"]
    warmup_steps  = cfg["warmup_steps"]
    grad_accum    = cfg["grad_accum"]
    max_grad_norm = 1.0

    model.train()
    optimizer.zero_grad()

    t_start   = time.time()
    loss_acc  = 0.0
    aux_acc   = 0.0
    tokens_acc = 0  # Track tokens for accurate throughput

    print(f"\n  Training for {total_steps:,} steps "
          f"(effective batch = {cfg['batch_size'] * grad_accum})\n")

    for step in range(start_step, total_steps):
        # ── Learning rate update ───────────────────────────────────────────
        lr = get_lr(step, warmup_steps, total_steps, cfg["lr"])
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        # ── Gradient accumulation ──────────────────────────────────────────
        for micro_step in range(grad_accum):
            try:
                x, y = next(loader_iter)
            except StopIteration:
                loader_iter = iter(loader)
                x, y = next(loader_iter)

            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            # FIXED: Only set current_step if LatentMoE supports it
            for module in model.modules():
                if isinstance(module, LatentMoE.LatentMoE):
                    if hasattr(module, 'current_step'):
                        module.current_step = step
                    # If not, LatentMoE doesn't need it — don't crash

            with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=(device.type == "cuda")):
                logits, ce_loss, aux_loss = model(x, labels=y)
                
                # FIXED: effective_aux_weight defined HERE, in scope for both train and log
                effective_aux_weight = model_cfg.aux_loss_weight  # 0.01
                
                total_loss = ce_loss + effective_aux_weight * aux_loss
                total_loss = total_loss / grad_accum

            total_loss.backward()
            
            # FIXED: Accumulate RAW losses (not divided by grad_accum)
            loss_acc += ce_loss.item()
            aux_acc  += aux_loss.item()
            tokens_acc += x.numel()

        # ── Gradient clip + optimizer step ────────────────────────────────
        grad_norm = nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()
        optimizer.zero_grad()

        # ── Logging ───────────────────────────────────────────────────────
        if step % 100 == 0:
            # FIXED: Divide by grad_accum here for averaging, not during accumulation
            steps_since_log = 100 if step > 0 else 1
            avg_loss = loss_acc / steps_since_log
            avg_aux  = aux_acc  / steps_since_log
            
            elapsed  = time.time() - t_start
            tokens_seen = step * cfg["batch_size"] * grad_accum * cfg["seq_len"]
            
            log_entry = {
                "step":        step,
                "loss":        round(avg_loss, 4),
                "aux_loss":    round(avg_aux,  6),
                "lr":          round(lr, 8),
                "grad_norm":   round(grad_norm.item(), 4),
                "tokens_M":    round(tokens_seen / 1e6, 1),
                "elapsed_s":   round(elapsed, 1),
            }
            log_file.write(json.dumps(log_entry) + "\n")
            log_file.flush()
            
            print(f"  step {step:>6,} | "
                  f"loss {avg_loss:.4f} | "
                  f"aux {avg_aux:.5f} | "
                  f"lr {lr:.2e} | "
                  f"grad_norm {grad_norm.item():.3f} | " 
                  f"tokens {tokens_seen/1e6:.0f}M | "
                  f"{elapsed:.0f}s", flush=True)
            
            # FIXED: Debug print uses the in-scope variable
            print(f"DEBUG: step={step}, effective_aux_weight={effective_aux_weight}, "
                  f"aux_loss={aux_loss.item():.5f}, ce_loss={ce_loss.item():.4f}")
            
            loss_acc = 0.0
            aux_acc  = 0.0

            if step > 0 and avg_loss < best_loss:
                best_loss = avg_loss
                best_path = ckpt_dir / "ckpt_best.pt"
                _save_checkpoint(model, optimizer, step, best_loss, best_path, stage)

        # ── Checkpointing ─────────────────────────────────────────────────
        if step % 2_000 == 0 and step > 0:
            ckpt_path = ckpt_dir / f"ckpt_step{step}.pt"
            _save_checkpoint(model, optimizer, step, best_loss, ckpt_path, stage)
            print(f"  Checkpoint saved → {ckpt_path}")

    # ── Final checkpoint ──────────────────────────────────────────────────
    final_path = ckpt_dir / "ckpt_final.pt"
    _save_checkpoint(model, optimizer, total_steps, best_loss, final_path, stage)
    log_file.close()
    print(f"\n  Training complete. Final checkpoint → {final_path}")


def _save_checkpoint(model, optimizer, step, best_loss, path, stage):
    torch.save({
        "model":      model.state_dict(),
        "optimizer":  optimizer.state_dict(),
        "step":       step,
        "best_loss":  best_loss,
        "model_cfg":  model.config,
        "stage":      stage,
    }, path)


def _save_checkpoint_async(model, optimizer, step, best_loss, path, stage):
    """
    Asynchronous checkpoint saving.
    FIXED: Clone state dicts immediately to avoid race conditions.
    """
    raw_model = model._orig_mod if hasattr(model, "_orig_mod") else model
    
    # FIXED: Clone immediately in main thread, write in background
    model_state = {k: v.cpu().clone() for k, v in raw_model.state_dict().items()}
    opt_state = copy.deepcopy(optimizer.state_dict())
    model_cfg = raw_model.config
    
    def _write():
        try:
            state = {
                "model":     model_state,
                "optimizer": opt_state,
                "step":      step,
                "best_loss": best_loss,
                "model_cfg": model_cfg,
                "stage":     stage,
            }
            torch.save(state, path)
            print(f"  [Async] Saved checkpoint to {path}")
        except Exception as e:
            print(f"  [Async ERROR] Failed to save {path}: {e}")

    threading.Thread(target=_write, daemon=True).start()


# ════════════════════════════════════════════════════════════════════════════
# 7.  CLI CHATBOT
# ════════════════════════════════════════════════════════════════════════════

def chat(checkpoint_path: str):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = True
    torch.backends.cudnn.benchmark = True
    print(f"\n{'='*60}")
    print(f"  SpeakMK1 LLM — CLI Chat")
    print(f"  Device   : {device}")
    print(f"  Checkpoint: {checkpoint_path}")
    print(f"{'='*60}")

    print("\n  Loading model …")
    torch.serialization.add_safe_globals([SpeakMK1LLMConfig])
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    saved_cfg = ckpt.get("model_cfg", None)
    if saved_cfg is not None:
        model_cfg = saved_cfg
    else:
        model_cfg = SpeakMK1LLMConfig()

    model = SpeakMK1LLM(model_cfg).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    params = count_parameters(model)
    print(f"  Parameters: {params['total_M']}M")
    print(f"  Loaded from step: {ckpt.get('step', 'unknown')}")
    print(f"  Best loss: {ckpt.get('best_loss', 'unknown')}")

    tok = get_tokenizer()

    temperature = 0.8
    top_p       = 0.9
    max_new     = 200
    history     = []

    print(f"\n  Ready. Type a prompt and press Enter.")
    print(f"  Commands: /help /temp /top_p /maxlen /reset /quit")
    print(f"  Temperature: {temperature} | top_p: {top_p} | max_new: {max_new}\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n  Bye.")
            break

        if not user_input:
            continue

        if user_input.startswith("/"):
            parts = user_input.split()
            cmd   = parts[0].lower()

            if cmd == "/quit":
                print("  Bye.")
                break

            elif cmd == "/help":
                print("  Commands:")
                print("    /temp <float>   — generation temperature (default 0.8)")
                print("    /top_p <float>  — nucleus sampling threshold (default 0.9)")
                print("    /maxlen <int>   — max new tokens (default 200)")
                print("    /reset          — clear conversation history")
                print("    /quit           — exit")
                print(f"  Current: temp={temperature} top_p={top_p} maxlen={max_new}")

            elif cmd == "/temp" and len(parts) > 1:
                temperature = float(parts[1])
                print(f"  Temperature → {temperature}")

            elif cmd == "/top_p" and len(parts) > 1:
                top_p = float(parts[1])
                print(f"  top_p → {top_p}")

            elif cmd == "/maxlen" and len(parts) > 1:
                max_new = int(parts[1])
                print(f"  max_new_tokens → {max_new}")

            elif cmd == "/reset":
                history = []
                print("  History cleared.")

            else:
                print(f"  Unknown command: {cmd}. Type /help for commands.")
            continue

        completion_prompt = f"The following is a continuation of text:\n\n{user_input}"
        new_ids = tok.encode(completion_prompt, add_special_tokens=False)

        history.extend(new_ids)
        if len(history) > 512:
            history = history[-512:]

        input_ids = torch.tensor([history], dtype=torch.long, device=device)

        print("Model: ", end="", flush=True)

        with torch.no_grad():
            output_ids = _generate_streaming(
                model=model,
                input_ids=input_ids,
                max_new_tokens=max_new,
                temperature=temperature,
                top_p=top_p,
                eos_token_id=tok.eos_token_id,
                tokenizer=tok,
            )

        generated_ids = output_ids[0, input_ids.shape[1]:].tolist()
        history.extend(generated_ids)
        if len(history) > 512:
            history = history[-512:]

        print()


def _generate_streaming(
    model:         SpeakMK1LLM,
    input_ids:     torch.Tensor,
    max_new_tokens: int,
    temperature:   float,
    top_p:         float,
    eos_token_id:  int,
    tokenizer,
) -> torch.Tensor:
    generated = input_ids
    partial   = []

    for _ in range(max_new_tokens):
        with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=(input_ids.device.type == "cuda")):
            logits, _, _ = model(generated)

        next_logits = logits[:, -1, :].float()

        if temperature != 1.0:
            next_logits = next_logits / temperature

        next_logits = _top_p_filter(next_logits, top_p)

        probs      = torch.softmax(next_logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)

        token_id = next_token.item()

        if token_id == eos_token_id:
            break

        generated = torch.cat([generated, next_token], dim=1)
        partial.append(token_id)

        try:
            text = tokenizer.decode(partial, skip_special_tokens=True)
            if text.endswith(" ") or len(partial) > 6:
                print(text, end="", flush=True)
                partial = []
        except Exception:
            pass

    if partial:
        try:
            print(tokenizer.decode(partial, skip_special_tokens=True),
                  end="", flush=True)
        except Exception:
            pass

    return generated


def _top_p_filter(logits: torch.Tensor, top_p: float) -> torch.Tensor:
    sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
    cumprobs = torch.softmax(sorted_logits, dim=-1).cumsum(dim=-1)
    
    sorted_remove = cumprobs - torch.softmax(sorted_logits, dim=-1) > top_p
    sorted_logits = sorted_logits.masked_fill(sorted_remove, float("-inf"))
    
    logits_filtered = torch.zeros_like(logits).scatter(1, sorted_indices, sorted_logits)
    return logits_filtered


# ════════════════════════════════════════════════════════════════════════════
# 8.  CLI ENTRYPOINT
# ════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="SpeakMK1 LLM — train or chat",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python train_speakmk1_llm.py tokenize --stage 1
  python train_speakmk1_llm.py train    --stage 1
  python train_speakmk1_llm.py train    --stage 1 --resume ./checkpoints/stage1/ckpt_step10000.pt
  python train_speakmk1_llm.py chat     --checkpoint ./checkpoints/stage1/ckpt_best.pt
        """,
    )

    subparsers = parser.add_subparsers(dest="command")

    p_tok = subparsers.add_parser("tokenize", help="Tokenize datasets for a stage")
    p_tok.add_argument("--stage", type=int, required=True, choices=[1, 2, 3, 4])

    p_childes = subparsers.add_parser("tokenize_childes", help="Tokenize local CHILDES .cha files")
    p_childes.add_argument("--path", type=str, required=True, help="Root directory of CHILDES .cha files")
    p_childes.add_argument("--stage", type=int, default=2)

    p_train = subparsers.add_parser("train", help="Train the model")
    p_train.add_argument("--stage",  type=int, required=True, choices=[1, 2, 3, 4])
    p_train.add_argument("--resume", type=str, default=None,
                         help="Path to checkpoint to resume from")

    p_chat = subparsers.add_parser("chat", help="Interactive CLI chat")
    p_chat.add_argument("--checkpoint", type=str, required=True,
                        help="Path to model checkpoint .pt file")

    args = parser.parse_args()

    if args.command == "tokenize":
        tokenize_stage(args.stage)

    elif args.command == "train":
        train(args.stage, resume=args.resume)

    elif args.command == "chat":
        chat(args.checkpoint)
    elif args.command == "tokenize_childes":
        tokenize_childes(args.path, stage=args.stage)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()