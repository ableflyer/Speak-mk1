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
from torch.utils.tensorboard import SummaryWriter

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
CKPT_DIR  = Path("./../Model_files/checkpoints_v2.3.1")
LOG_DIR   = Path("./../Data/logs_v2.3.1")

# add near top with other constants
SPECIAL_TOKENS = {
    "system_start":  "<|system|>",
    "child_start":   "<|child|>",
    "think_start":   "<|think|>",
    "slp_start":     "<|slp|>",
    "turn_end":      "<|endturn|>",
    "seq_end":       "<|endseq|>",
}

SYSTEM_PROMPT_TEMPLATE = (
    "You are SpeakMK1, a warm and clinically expert AI speech-language pathologist for children. "
    "The child is {age} years old and presents with {disorder} ({specific_error}). "
    "Clinical goal: {clinical_goal}. "
    "Primary strategy: {strategy}. "
    "Always analyze errors clinically before responding. Be encouraging, patient, and use scaffolding."
)

STAGE_CONFIG = {
    1: {
        "name": "general_pretraining",
        "datasets": [
            {
                "id":      "tinystories",
                "hf_path": "karpathy/tinystories-gpt4-clean",
                "hf_name": None,
                "split":   "train",
                "text_key": "text",
                "max_docs": 2_800_000,
                "streaming": False,
                "repeat": 3,
            }
        ],
        "train_steps":  75_000,
        "lr":           3e-4,
        "warmup_steps": 3_000,
        "seq_len":      512,
        "batch_size":   8,
        "grad_accum":   4,
    },
    2: {
        "name": "child_directed_adapt",
        "datasets": [
            {
                "id":      "tinystories",
                "hf_path": "karpathy/tinystories-gpt4-clean",
                "hf_name": None,
                "split":   "train",
                "text_key": "text",
                "max_docs": 500_000,
                "streaming": False,
                "weight":  1.0,
                "repeat":  1,
            },
            {
                "id":      "childes",
                "local":   True,
                "weight": 2.0,
                "repeat": 3,  # repeat childes 3x to give it more weight during training
            }
        ],
        "train_steps":  5_000,
        "lr":           1e-4,
        "warmup_steps": 250,
        "seq_len":      512,
        "batch_size":   8,
        "grad_accum":   4,
    },
    3: {
        "name": "clinical_knowledge",
        "datasets": [
            {
                "id":      "pubmed",
                "local":   True,
                "weight":  1.0,
                "repeat":  1,
            }
        ],
        "train_steps":  3_000,   # pubmed is background, don't overfit
        "lr":           5e-5,    # low — you're fine-tuning, not pretraining
        "warmup_steps": 150,
        "seq_len":      512,
        "batch_size":   8,
        "grad_accum":   4,
    },
    4: {
        "name": "clinical_injection",
        "datasets": [],
        "train_steps":  5_000,
        "lr":           5e-5,
        "warmup_steps": 200,
        "seq_len":      512,
        "batch_size":   8,
        "grad_accum":   4,
    },
    5: {
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
        "batch_size":   8,
        "grad_accum":   4,
    },
}

# ════════════════════════════════════════════════════════════════════════════
# Slp dataset
# ════════════════════════════════════════════════════════════════════════════

class SLPDataset(Dataset):
    def __init__(self, stage: int = 4, seq_len: int = 512, repeat: int = 1):
        ids_path    = DATA_DIR / f"stage{stage}_slp_ids.bin"
        labels_path = DATA_DIR / f"stage{stage}_slp_labels.bin"
        
        self.ids    = np.memmap(str(ids_path),    dtype=np.uint16, mode="r")
        self.labels = np.memmap(str(labels_path), dtype=np.int32,  mode="r")
        self.seq_len = seq_len
        self.n_seqs  = (len(self.ids) - 1) // seq_len
        self.repeat  = repeat
        
        print(f"  [SLPDataset] {len(self.ids):,} tokens | {self.n_seqs:,} seqs")

    def __len__(self):
        return self.n_seqs * self.repeat

    def __getitem__(self, idx):
        idx = idx % self.n_seqs
        start = idx * self.seq_len
        x = torch.from_numpy(
            self.ids[start : start + self.seq_len].astype(np.int64)
        )
        y = torch.from_numpy(
            self.labels[start + 1 : start + self.seq_len + 1].astype(np.int64)
        )
        return x, y

# ════════════════════════════════════════════════════════════════════════════
# 2.  TOKENIZER
# ════════════════════════════════════════════════════════════════════════════

def get_tokenizer():
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.add_special_tokens({
        "additional_special_tokens": list(SPECIAL_TOKENS.values())
    })
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
                    text = re.sub(r"^\*[A-Z]+:[\t ]+", "", line)  # speaker ID
                    text = re.sub(r"\[.*?\]", "", text)        # [brackets]
                    text = re.sub(r"<.*?>", "", text)          # <tags>
                    text = re.sub(r"\(+\.*\)+", "", text)      # () (..) (...)
                    text = re.sub(r"\.[A-Za-z]+", "", text)    # .PAR .ee .es etc
                    text = re.sub(r"[+/\\]", "", text)         # operators
                    text = re.sub(r"[&@]\w+", "", text)        # &word @word
                    text = re.sub(r"\*\d+", "", text)          # *4 *2 etc
                    text = re.sub(r"\d+", "", text)            # stray numbers
                    text = re.sub(r"_", " ", text)             # underscores
                    text = re.sub(r"\s+", " ", text)           # collapse spaces
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

def _format_slp_dialogue(entry: dict, tokenizer) -> tuple:
    meta = entry["metadata"]
    system_text = (
        SPECIAL_TOKENS["system_start"] + " " +
        SYSTEM_PROMPT_TEMPLATE.format(
            age=meta["age"],
            disorder=meta["disorder"],
            specific_error=meta.get("specific_error", ""),
            clinical_goal=meta.get("clinical_goal", ""),
            strategy=meta.get("primary_strategy", ""),
        ) + " " + SPECIAL_TOKENS["turn_end"]
    )

    all_ids    = []
    all_labels = []

    sys_ids = tokenizer.encode(system_text, add_special_tokens=False)
    all_ids    += sys_ids
    all_labels += [-100] * len(sys_ids)

    for turn in entry["dialogue"]:
        child_ids = tokenizer.encode(
            SPECIAL_TOKENS["child_start"] + " " + turn["child_input"] + " " + SPECIAL_TOKENS["turn_end"],
            add_special_tokens=False
        )
        think_ids = tokenizer.encode(
            SPECIAL_TOKENS["think_start"] + " " + turn["model_thought"] + " " + SPECIAL_TOKENS["turn_end"],
            add_special_tokens=False
        )
        slp_ids = tokenizer.encode(
            SPECIAL_TOKENS["slp_start"] + " " + turn["slp_response"] + " " + SPECIAL_TOKENS["turn_end"],
            add_special_tokens=False
        )

        all_ids    += child_ids + think_ids
        all_labels += [-100] * (len(child_ids) + len(think_ids))

        all_ids    += slp_ids
        all_labels += slp_ids   # train on SLP output only

    eos_ids = tokenizer.encode(SPECIAL_TOKENS["seq_end"], add_special_tokens=False)
    all_ids    += eos_ids
    all_labels += eos_ids

    return all_ids, all_labels

def tokenize_slp_jsonl(jsonl_path: str, stage: int = 3):
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    ids_path    = DATA_DIR / f"stage{stage}_slp_ids.bin"
    labels_path = DATA_DIR / f"stage{stage}_slp_labels.bin"
    
    if ids_path.exists() and labels_path.exists():
        print(f"  [slp] already exists — skipping.")
        return

    # load tokenizer with special tokens
    tok = get_tokenizer()
    tok.add_special_tokens({
        "additional_special_tokens": list(SPECIAL_TOKENS.values())
    })

    all_ids    = []
    all_labels = []
    n_seqs = 0

    print(f"  [slp] Tokenizing {jsonl_path} ...")

    with open(jsonl_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue

            ids, labels = _format_slp_dialogue(entry, tok)

            # truncate to 2048
            if len(ids) > 2048:
                ids    = ids[:2048]
                labels = labels[:2048]

            all_ids    += ids
            all_labels += labels
            n_seqs += 1

    ids_arr    = np.array(all_ids,    dtype=np.uint16)
    labels_arr = np.array(all_labels, dtype=np.int32)

    ids_arr.tofile(str(ids_path))
    labels_arr.tofile(str(labels_path))

    n_slp = sum(1 for l in all_labels if l != -100)
    print(f"  [slp] Done. {n_seqs} dialogues | "
          f"{len(all_ids):,} tokens | "
          f"{n_slp:,} trainable tokens ({100*n_slp/len(all_ids):.1f}%)")
    
def tokenize_pubmed_jsonl(jsonl_path: str, stage: int = 2):
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    bin_path = DATA_DIR / f"stage{stage}_pubmed.bin"
    
    if bin_path.exists():
        print(f"  [pubmed] exists — skipping.")
        return

    tok = get_tokenizer()
    eos = tok.eos_token_id
    tokens_done = 0

    with open(bin_path, "wb") as fout, open(jsonl_path) as fin:
        for line in fin:
            obj = json.loads(line.strip())
            # Use title + abstract + body (skip full_text, it's redundant)
            text = f"{obj.get('title','')} {obj.get('abstract','')} {obj.get('body','')}".strip()
            if len(text) < 300:
                continue
            text = text[:20_000]  # cap long review papers
            ids = tok.encode(text, add_special_tokens=False)
            ids.append(eos)
            fout.write(np.array(ids, dtype=np.uint16).tobytes())
            tokens_done += len(ids)

    print(f"  [pubmed] {tokens_done/1e6:.2f}M tokens → {bin_path}")

# ════════════════════════════════════════════════════════════════════════════
# 4.  DATASET
# ════════════════════════════════════════════════════════════════════════════

class PackedTokenDataset(Dataset):
    def __init__(self, bin_path: str, seq_len: int, repeat: int = 1):
        self.seq_len = seq_len
        self.data    = np.memmap(bin_path, dtype=np.uint16, mode="r")
        self.n_seqs  = (len(self.data) - 1) // seq_len
        print(f"  Loaded {bin_path}")
        print(f"    Tokens : {len(self.data):,}")
        print(f"    Seqs   : {self.n_seqs:,}  (seq_len={seq_len})")
        self.repeat  = repeat

    def __len__(self):
        return self.n_seqs * self.repeat

    def __getitem__(self, idx: int):
        idx = idx % self.n_seqs
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
    
    if stage == 4:  # For stage 4, use the SLP dataset only
        slp_ds = SLPDataset(stage=stage, seq_len=seq_len, repeat=20)
        print(f"  [slp] {len(slp_ds):,} sequences (repeat=20)")
        return DataLoader(
            slp_ds,
            batch_size=batch_size,
            shuffle=True,
            num_workers=2,
            pin_memory=True,
            drop_last=True,
        )

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
        repeat = ds_cfg.get("repeat", 1)
        ds = PackedTokenDataset(str(bin_path), seq_len, repeat=repeat)
        stage_datasets.append(ds)
        weights.append(ds_cfg.get("weight", 1.0))
        print(f"  [{ds_id}] {len(ds):,} sequences (weight={weights[-1]})")

    # Pick up manually added local files
    for extra in DATA_DIR.glob(f"stage{stage}_*.bin"):
        name = extra.stem.replace(f"stage{stage}_", "")
        if not any(d["id"] == name for d in STAGE_CONFIG[stage]["datasets"]):
            print(f"  Found extra dataset: {extra.name}")
            ds = PackedTokenDataset(str(extra), seq_len, repeat=4)
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
    writer = SummaryWriter(log_dir=str(log_dir / "tensorboard"))

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
        vocab_size=50283,
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
    
    start_step = 0
    best_loss = float("inf")
    resume_optimizer_state = None
    
    tok = get_tokenizer()
    
    if resume:
        print(f"  Resuming from {resume} …")
        torch.serialization.add_safe_globals([SpeakMK1LLMConfig])
        ckpt = torch.load(resume, map_location=device)
        # first_key = list(ckpt["model"].keys())[0]
        # print(f"First key: {first_key}")
        # print(f"Value: mean={ckpt['model'][first_key].mean():.4f}, std={ckpt['model'][first_key].std():.4f}")
        # # check if weights actually have real values
        # for name, param in model.named_parameters():
        #     print(f"{name}: mean={param.data.mean():.4f}, std={param.data.std():.4f}")
        #     break  # just first layer
        
        state = ckpt["model"]

        # Extend output projection + embedding if checkpoint was vocab_size=50277
        for key in list(state.keys()):
            if state[key].shape != model.state_dict()[key].shape:
                old = state[key]
                new = model.state_dict()[key].clone()
                # Copy old weights into the new (larger) tensor
                slices = tuple(slice(0, s) for s in old.shape)
                new[slices] = old
                state[key] = new
                print(f"  Extended {key}: {old.shape} -> {new.shape}")
        
        model.load_state_dict(ckpt["model"])
        
        with torch.no_grad():
            ids = tok.encode("The little dog ran", return_tensors="pt").to(device)
            model.eval()
            _, ce, _ = model(ids, labels=ids)
            model.train()
            print(f"POST LOAD SANITY: {ce.item():.4f}")

        # print(f"train mode ce: {ce_train.item():.4f}")
        # print(f"eval mode ce:  {ce_eval.item():.4f}")
        print(f"Checkpoint stage: {ckpt.get('stage')}")
        print(f"Checkpoint step: {ckpt.get('step')}")
        print(f"Checkpoint best_loss: {ckpt.get('best_loss')}")

        # # quick sanity forward
        # model.eval()
        # with torch.no_grad():
        #     x_test = torch.randint(0, 50277, (1, 32), device=device)
        #     logits, ce, aux = model(x_test, labels=x_test)
        #     print(f"Sanity ce_loss after load: {ce.item():.4f}")
        # model.train()
        
        saved_stage = ckpt.get("stage")
        saved_step = ckpt["step"]
        print(f"  Checkpoint stage={saved_stage}, step={saved_step}, best_loss={ckpt.get('best_loss')}")
        # if saved_stage is None:
        #     if saved_step >= cfg["train_steps"]:
        #         print(f"  Detected checkpoint from previous stage (step {saved_step} > {cfg['train_steps']})")
        #         start_step = 0
        #         print(f"  Starting Stage {stage} training from step 0 with transferred weights")
        #     else:
        #         start_step = saved_step
        #         optimizer.load_state_dict(ckpt["optimizer"])
        #         best_loss = ckpt.get("best_loss", float("inf"))
                
        # elif saved_stage != stage:
        #     print(f"  Transferring from Stage {saved_stage} to Stage {stage}")
        #     start_step = 0
        #     print(f"  Fresh optimizer for Stage {stage} (LR={cfg['lr']})")
            
        # else:
        #     start_step = saved_step
        #     optimizer.load_state_dict(ckpt["optimizer"])
        #     best_loss = ckpt.get("best_loss", float("inf"))
        if saved_stage != stage:
            print(f"  Transferring from Stage {saved_stage} to Stage {stage}")
            start_step = 0
            print(f"  Fresh optimizer for Stage {stage}")
        else:
            start_step = saved_step
            resume_optimizer_state = ckpt["optimizer"]
            best_loss = ckpt.get("best_loss", float("inf"))
        
        print(f"  Starting at step {start_step}")

    params = count_parameters(model)
    print(f"  Model parameters: {params['total_M']}M total | {params['trainable_M']}M trainable")

    # ── Optimizer ─────────────────────────────────────────────────────────
    # FIXED: Router LR multiplier reduced from 10x to 1.5x for stability
    # MoE routers are sensitive to aggressive LR scaling [^2^]
    
    if HAS_BNB:
        print(f"  Optimizer: AdamW8bit (bitsandbytes) — saves ~2 GB VRAM")
        optimizer = AdamW8bit(
            model.parameters(),
            lr=cfg["lr"],
            betas=(0.9, 0.95),
            weight_decay=0.1,
            eps=1e-8
        )
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
    # ── resume ───────────────────────────────────────────────────────
    if resume_optimizer_state is not None:
        optimizer.load_state_dict(resume_optimizer_state)
        print(f"  Optimizer state restored")

    # ── Timing test ───────────────────────────────────────────────────────
    # if device.type == "cuda":
    #     print("\n  Running timing test …")
    #     model.train()
    #     x_test = torch.randint(0, 50277, (4, 512), device=device)
    #     y_test = torch.randint(0, 50277, (4, 512), device=device)

    #     # Warmup
    #     with torch.amp.autocast("cuda", dtype=torch.bfloat16):
    #         logits, ce_loss, aux_loss = model(x_test, labels=y_test)
    #         total = (ce_loss + model_cfg.aux_loss_weight * aux_loss) / cfg["grad_accum"]
    #     total.backward()
    #     optimizer.zero_grad()
    #     torch.cuda.synchronize()

    #     # Time 5 passes
    #     t0 = time.time()
    #     for _ in range(5):
    #         with torch.amp.autocast("cuda", dtype=torch.bfloat16):
    #             logits, ce_loss, aux_loss = model(x_test, labels=y_test)
    #             total = (ce_loss + model_cfg.aux_loss_weight * aux_loss) / cfg["grad_accum"]
    #         total.backward()
    #         optimizer.zero_grad()
    #     torch.cuda.synchronize()
    #     t1 = time.time()

    #     ms = (t1 - t0) / 5 * 1000
    #     print(f"  Forward+backward : {ms:.0f}ms per micro-step")
    #     print(f"  Per full step    : {ms * cfg['grad_accum'] / 1000:.1f}s")
    #     print(f"  Projected speed  : {86400 / (ms * cfg['grad_accum'] / 1000):.0f} steps/day")
    
    # model.load_state_dict(ckpt["model"], strict=False)
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
            avg_loss = loss_acc / (steps_since_log * grad_accum)
            avg_aux  = aux_acc  / (steps_since_log * grad_accum)
            
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

            writer.add_scalar("Loss/train",    avg_loss,            step)
            writer.add_scalar("Loss/aux",      avg_aux,             step)
            writer.add_scalar("LR",            lr,                  step)
            writer.add_scalar("Grad/norm",     grad_norm.item(),    step)
            writer.add_scalar("Tokens/M",      tokens_seen / 1e6,   step)
            writer.add_scalar("ce_loss",       ce_loss.item(),      step)
            
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
    writer.close()
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

def _extract_slp_response(raw_text: str) -> str:
    import re
    # Strip think block
    text = re.sub(r"<\|think\|>.*?<\|endturn\|>", "", raw_text, flags=re.DOTALL)
    # Strip remaining special tokens
    for token in SPECIAL_TOKENS.values():
        text = text.replace(token, "")
    return text.strip()

def chat(checkpoint_path: str):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = True
    torch.backends.cudnn.benchmark = True

    print(f"\n{'='*60}")
    print(f"  SpeakMK1 LLM -- CLI Chat")
    print(f"  Device    : {device}")
    print(f"  Checkpoint: {checkpoint_path}")
    print(f"{'='*60}")

    print("\n  Loading model ...")
    torch.serialization.add_safe_globals([SpeakMK1LLMConfig])
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

    saved_cfg  = ckpt.get("model_cfg", None)
    model_cfg  = saved_cfg if saved_cfg is not None else SpeakMK1LLMConfig()
    model      = SpeakMK1LLM(model_cfg).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    params = count_parameters(model)
    print(f"  Parameters  : {params['total_M']}M")
    print(f"  Loaded step : {ckpt.get('step', 'unknown')}")
    print(f"  Best loss   : {ckpt.get('best_loss', 'unknown')}")

    tok = get_tokenizer()  # now includes special tokens

    # Pre-compute stop token ids
    eos_id     = tok.eos_token_id
    endturn_id = tok.encode(SPECIAL_TOKENS["turn_end"], add_special_tokens=False)[0]

    # ── Session state ──────────────────────────────────────────────────────
    temperature  = 0.8
    top_p        = 0.9
    max_new      = 200
    history: list[int] = []
    system_injected  = False

    # Default clinical context — can be changed via /context command
    clinical_context = dict(
        age            = "6",
        disorder       = "Stopping",
        specific_error = "Stopping of /s/ and /z/ (s/z → t/d)",
        clinical_goal  = "Establish /s/ using airflow awareness",
        strategy       = "Tactile Cueing (touching alveolar ridge awareness)",
    )

    def inject_system():
        nonlocal system_injected
        system_text = (
            SPECIAL_TOKENS["system_start"] + " " +
            SYSTEM_PROMPT_TEMPLATE.format(**clinical_context) + " " +
            SPECIAL_TOKENS["turn_end"]
        )
        sys_ids = tok.encode(system_text, add_special_tokens=False)
        history.extend(sys_ids)
        system_injected = True

    print(f"\n  Ready. Type a prompt and press Enter.")
    print(f"  Commands: /help /temp /top_p /maxlen /context /reset /quit")
    print(f"  Temperature: {temperature}  top_p: {top_p}  max_new: {max_new}\n")

    while True:
        try:
            user_input = input("Child: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n  Bye.")
            break

        if not user_input:
            continue

        # ── Commands ───────────────────────────────────────────────────────
        if user_input.startswith("/"):
            parts = user_input.split()
            cmd   = parts[0].lower()

            if cmd == "/quit":
                print("  Bye.")
                break

            elif cmd == "/help":
                print("  Commands:")
                print("    /temp <float>         -- generation temperature (default 0.8)")
                print("    /top_p <float>        -- nucleus sampling (default 0.9)")
                print("    /maxlen <int>         -- max new tokens (default 200)")
                print("    /context              -- show current clinical context")
                print("    /context <key> <val>  -- update clinical context field")
                print("      keys: age, disorder, specific_error, clinical_goal, strategy")
                print("    /reset                -- clear conversation history")
                print("    /quit                 -- exit")
                print(f"  Current: temp={temperature} top_p={top_p} maxlen={max_new}")

            elif cmd == "/temp" and len(parts) > 1:
                temperature = float(parts[1])
                print(f"  Temperature set to {temperature}")

            elif cmd == "/top_p" and len(parts) > 1:
                top_p = float(parts[1])
                print(f"  top_p set to {top_p}")

            elif cmd == "/maxlen" and len(parts) > 1:
                max_new = int(parts[1])
                print(f"  max_new_tokens set to {max_new}")

            elif cmd == "/context":
                if len(parts) == 1:
                    # Print current context
                    for k, v in clinical_context.items():
                        print(f"    {k}: {v}")
                elif len(parts) >= 3:
                    key = parts[1]
                    val = " ".join(parts[2:])
                    if key in clinical_context:
                        clinical_context[key] = val
                        print(f"  {key} set to: {val}")
                        print(f"  Use /reset to apply new context to conversation.")
                    else:
                        print(f"  Unknown key '{key}'. Valid keys: {list(clinical_context.keys())}")

            elif cmd == "/reset":
                history.clear()
                system_injected = False
                print("  History cleared.")

            else:
                print(f"  Unknown command: {cmd}. Type /help.")
            continue

        # ── Inject system prompt on first real turn ────────────────────────
        if not system_injected:
            inject_system()

        # ── Encode child utterance with special tokens ─────────────────────
        child_text = (
            SPECIAL_TOKENS["child_start"] + " " +
            user_input + " " +
            SPECIAL_TOKENS["turn_end"] +
            SPECIAL_TOKENS["think_start"]   # prime model to think before responding
        )
        new_ids = tok.encode(child_text, add_special_tokens=False)
        history.extend(new_ids)

        # Trim history to last 512 tokens to avoid OOM
        if len(history) > 512:
            # Always keep system prompt (first chunk up to turn_end)
            # Find end of system block
            system_end_marker = tok.encode(
                SPECIAL_TOKENS["turn_end"], add_special_tokens=False
            )[0]
            try:
                sys_end_idx = history.index(system_end_marker) + 1
            except ValueError:
                sys_end_idx = 0
            system_ids  = history[:sys_end_idx]
            recent_ids  = history[sys_end_idx:][-512 + sys_end_idx:]
            history     = system_ids + recent_ids

        input_ids = torch.tensor([history], dtype=torch.long, device=device)

        print("SLP: ", end="", flush=True)

        with torch.no_grad():
            output_ids = _generate_streaming(
                model          = model,
                input_ids      = input_ids,
                max_new_tokens = max_new,
                temperature    = temperature,
                top_p          = top_p,
                eos_token_id   = eos_id,
                endturn_id     = endturn_id,
                tokenizer      = tok,
            )

        # Decode full generated portion for history + display cleanup
        generated_ids   = output_ids[0, input_ids.shape[1]:].tolist()
        raw_generated   = tok.decode(generated_ids, skip_special_tokens=False)
        clean_response  = _extract_slp_response(raw_generated)

        # Add SLP response to history (with proper tokens so model sees full context)
        slp_text = (
            SPECIAL_TOKENS["slp_start"] + " " +
            clean_response + " " +
            SPECIAL_TOKENS["turn_end"]
        )
        history.extend(tok.encode(slp_text, add_special_tokens=False))

        print()



def _generate_streaming(
    model:          SpeakMK1LLM,
    input_ids:      torch.Tensor,
    max_new_tokens: int,
    temperature:    float,
    top_p:          float,
    eos_token_id:   int,
    endturn_id:     int,
    tokenizer,
) -> torch.Tensor:
    generated = input_ids
    partial   = []

    for _ in range(max_new_tokens):
        with torch.amp.autocast("cuda", dtype=torch.bfloat16,
                                enabled=(input_ids.device.type == "cuda")):
            logits, _, _ = model(generated)

        next_logits = logits[:, -1, :].float()

        if temperature != 1.0:
            next_logits = next_logits / temperature

        next_logits = _top_p_filter(next_logits, top_p)
        probs       = torch.softmax(next_logits, dim=-1)
        next_token  = torch.multinomial(probs, num_samples=1)
        token_id    = next_token.item()

        # Stop at EOS or end-of-turn
        if token_id in (eos_token_id, endturn_id):
            break

        generated = torch.cat([generated, next_token], dim=1)
        partial.append(token_id)

        # Stream partial decoded text
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
    p_tok.add_argument("--stage", type=int, required=True, choices=[1, 2, 3, 4, 5])

    p_childes = subparsers.add_parser("tokenize_childes", help="Tokenize local CHILDES .cha files")
    p_childes.add_argument("--path", type=str, required=True, help="Root directory of CHILDES .cha files")
    p_childes.add_argument("--stage", type=int, default=2)
    
    p_slp = subparsers.add_parser("tokenize_slp", help="Tokenize SLP JSONL for stage 3")
    p_slp.add_argument("--path",  required=True, help="Path to SLP JSONL file")
    p_slp.add_argument("--stage", type=int, default=3)
    
    p_pubmed = subparsers.add_parser("tokenize_pubmed", help="Tokenize PubMed JSONL for stage 2")
    p_pubmed.add_argument("--path",  required=True, help="Path to PubMed JSONL file")
    p_pubmed.add_argument("--stage", type=int, default=3)

    p_train = subparsers.add_parser("train", help="Train the model")
    p_train.add_argument("--stage",  type=int, required=True, choices=[1, 2, 3, 4, 5])
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
    elif args.command == "tokenize_slp":
        tokenize_slp_jsonl(args.path, stage=args.stage)
    elif args.command == "tokenize_pubmed":
        tokenize_pubmed_jsonl(args.path, stage=args.stage)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()