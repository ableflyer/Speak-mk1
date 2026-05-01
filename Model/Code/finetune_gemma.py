"""
finetune_gemma_slp.py
=====================
QLoRA fine-tuning of Gemma 3 4B on SLP clinical dialogue data.
Reuses your existing tokenize_slp_jsonl pipeline and SLPDataset.

Usage:
    python finetune_gemma_slp.py tokenize --path ./data/slp_dialogues.jsonl
    python finetune_gemma_slp.py train
    python finetune_gemma_slp.py train --resume ./checkpoints_gemma/ckpt_best.pt
    python finetune_gemma_slp.py chat   --checkpoint ./checkpoints_gemma/ckpt_best
"""

from __future__ import annotations

import argparse
import json
import os
import time
import gc
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

# ── QLoRA deps ───────────────────────────────────────────────────────────────
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    get_cosine_schedule_with_warmup,
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    PeftModel,
)

# ════════════════════════════════════════════════════════════════════════════
# CONFIG
# ════════════════════════════════════════════════════════════════════════════

MODEL_ID   = "google/gemma-3-4b-it"   # instruction-tuned base
DATA_DIR   = Path("./../Data/LLM_finetune")
CKPT_DIR   = Path("./../Model_files/checkpoints_gemma")

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
    "Always analyse errors clinically before responding. Be encouraging, patient, and use scaffolding."
)

@dataclass
class TrainConfig:
    # LoRA
    lora_r:           int   = 8
    lora_alpha:       int   = 16
    lora_dropout:     float = 0.05
    # Training
    train_steps:      int   = 3_000
    warmup_steps:     int   = 150
    lr:               float = 2e-4
    batch_size:       int   = 1       # fits RTX 4060 8GB comfortably
    grad_accum:       int   = 16      # effective batch = 16
    seq_len:          int   = 256
    max_grad_norm:    float = 1.0
    # Logging / saving
    log_every:        int   = 50
    save_every:       int   = 500


CFG = TrainConfig()


# ════════════════════════════════════════════════════════════════════════════
# TOKENIZER
# ════════════════════════════════════════════════════════════════════════════

def get_tokenizer():
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.add_special_tokens({
        "additional_special_tokens": list(SPECIAL_TOKENS.values())
    })
    return tok


# ════════════════════════════════════════════════════════════════════════════
# TOKENIZATION  (same logic as your original _format_slp_dialogue)
# ════════════════════════════════════════════════════════════════════════════

def _format_slp_dialogue(entry: dict, tokenizer) -> tuple[list[int], list[int]]:
    meta = entry["metadata"]
    system_text = (
        SPECIAL_TOKENS["system_start"] + " " +
        SYSTEM_PROMPT_TEMPLATE.format(
            age            = meta["age"],
            disorder       = meta["disorder"],
            specific_error = meta.get("specific_error", ""),
            clinical_goal  = meta.get("clinical_goal", ""),
            strategy       = meta.get("primary_strategy", ""),
        ) + " " + SPECIAL_TOKENS["turn_end"]
    )

    all_ids, all_labels = [], []

    sys_ids = tokenizer.encode(system_text, add_special_tokens=False)
    all_ids    += sys_ids
    all_labels += [-100] * len(sys_ids)   # don't train on system prompt

    for turn in entry["dialogue"]:
        child_ids = tokenizer.encode(
            SPECIAL_TOKENS["child_start"] + " " + turn["child_input"] + " " + SPECIAL_TOKENS["turn_end"],
            add_special_tokens=False,
        )
        think_ids = tokenizer.encode(
            SPECIAL_TOKENS["think_start"] + " " + turn["model_thought"] + " " + SPECIAL_TOKENS["turn_end"],
            add_special_tokens=False,
        )
        slp_ids = tokenizer.encode(
            SPECIAL_TOKENS["slp_start"] + " " + turn["slp_response"] + " " + SPECIAL_TOKENS["turn_end"],
            add_special_tokens=False,
        )

        # mask child + think, train only on SLP response
        all_ids    += child_ids + think_ids + slp_ids
        all_labels += [-100] * (len(child_ids) + len(think_ids)) + slp_ids

    eos_ids = tokenizer.encode(SPECIAL_TOKENS["seq_end"], add_special_tokens=False)
    all_ids    += eos_ids
    all_labels += eos_ids

    return all_ids, all_labels


def tokenize_slp_jsonl(jsonl_path: str):
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    ids_path    = DATA_DIR / "slp_ids.bin"
    labels_path = DATA_DIR / "slp_labels.bin"

    if ids_path.exists() and labels_path.exists():
        print("  [slp] already tokenised — skipping.")
        return

    tok = get_tokenizer()
    all_ids, all_labels = [], []
    n_seqs = 0

    print(f"  Tokenising {jsonl_path} ...")
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
            if len(ids) > 2048:
                ids    = ids[:2048]
                labels = labels[:2048]

            all_ids    += ids
            all_labels += labels
            n_seqs += 1

    np.array(all_ids,    dtype=np.int32).tofile(str(ids_path))
    np.array(all_labels, dtype=np.int32 ).tofile(str(labels_path))

    n_train = sum(1 for l in all_labels if l != -100)
    print(f"  Done. {n_seqs} dialogues | {len(all_ids):,} tokens | "
          f"{n_train:,} trainable ({100*n_train/len(all_ids):.1f}%)")


# ════════════════════════════════════════════════════════════════════════════
# DATASET
# ════════════════════════════════════════════════════════════════════════════

class SLPDataset(Dataset):
    def __init__(self, seq_len: int = 512):
        ids_path    = DATA_DIR / "slp_ids.bin"
        labels_path = DATA_DIR / "slp_labels.bin"

        self.ids = np.memmap(str(ids_path), dtype=np.int32, mode="r")
        self.labels = np.memmap(str(labels_path), dtype=np.int32,  mode="r")
        self.seq_len = seq_len
        self.n_seqs  = (len(self.ids) - 1) // seq_len

        print(f"  [SLPDataset] {len(self.ids):,} tokens | {self.n_seqs:,} seqs")

    def __len__(self):
        return self.n_seqs

    def __getitem__(self, idx):
        start  = idx * self.seq_len
        x      = torch.from_numpy(self.ids   [start     : start + self.seq_len    ].astype(np.int64))
        labels = torch.from_numpy(self.labels[start + 1 : start + self.seq_len + 1].astype(np.int64))
        return x, labels


# ════════════════════════════════════════════════════════════════════════════
# MODEL
# ════════════════════════════════════════════════════════════════════════════

def load_model_for_training(device):
    print("  Loading Gemma 3 4B in 4-bit ...")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit              = True,
        bnb_4bit_quant_type       = "nf4",
        bnb_4bit_compute_dtype    = torch.bfloat16,
        bnb_4bit_use_double_quant = True,   # saves ~0.4 bits/param extra
        llm_int8_enable_fp32_cpu_offload = True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config = bnb_config,
        device_map          = "auto",
        torch_dtype         = torch.bfloat16,
        trust_remote_code   = True,
        max_memory          = {0: "5.5GiB", "cpu": "8GiB"},  # add this
    )

    model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        r              = CFG.lora_r,
        lora_alpha     = CFG.lora_alpha,
        lora_dropout   = CFG.lora_dropout,
        bias           = "none",
        task_type      = "CAUSAL_LM",
        # Gemma attention projection names
        target_modules = ["q_proj", "v_proj"],
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model


# ════════════════════════════════════════════════════════════════════════════
# TRAINING
# ════════════════════════════════════════════════════════════════════════════

def train(resume: Optional[str] = None):
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"\n{'='*60}")
    print(f"  SpeakMK1 QLoRA Fine-tune — Gemma 3 4B")
    print(f"  Device : {device}")
    if device.type == "cuda":
        print(f"  GPU    : {torch.cuda.get_device_name(0)}")
        print(f"  VRAM   : {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")
    print(f"{'='*60}\n")

    # ── Model ─────────────────────────────────────────────────────────────
    model = load_model_for_training(device)

    start_step = 0
    best_loss  = float("inf")

    if resume:
        print(f"  Resuming LoRA adapter from {resume} ...")
        # For LoRA, we save/load the adapter only
        model.load_adapter(resume, adapter_name="default")
        meta_path = Path(resume) / "train_meta.json"
        if meta_path.exists():
            meta       = json.loads(meta_path.read_text())
            start_step = meta.get("step", 0)
            best_loss  = meta.get("best_loss", float("inf"))
        print(f"  Resumed from step {start_step}")

    # ── Tokenizer (resize embeddings if new special tokens added) ─────────
    tok = get_tokenizer()
    model.resize_token_embeddings(len(tok))

    # ── Optimiser — only LoRA params ──────────────────────────────────────
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr           = CFG.lr,
        betas        = (0.9, 0.95),
        weight_decay = 0.01,
        eps          = 1e-8,
    )

    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps   = CFG.warmup_steps,
        num_training_steps = CFG.train_steps,
    )
    if start_step > 0:
        # Fast-forward scheduler to resume step
        for _ in range(start_step):
            scheduler.step()

    # ── Data ──────────────────────────────────────────────────────────────
    dataset = SLPDataset(seq_len=CFG.seq_len)
    loader  = DataLoader(
        dataset,
        batch_size      = CFG.batch_size,
        shuffle         = True,
        num_workers     = 2,
        pin_memory      = True,
        drop_last       = True,
    )
    loader_iter = iter(loader)

    # ── Log ───────────────────────────────────────────────────────────────
    log_path = CKPT_DIR / "train_log.jsonl"
    log_file = open(log_path, "a")

    # ── Loop ──────────────────────────────────────────────────────────────
    model.train()
    optimizer.zero_grad()

    loss_acc = 0.0
    t_start  = time.time()

    print(f"  Training for {CFG.train_steps:,} steps "
          f"(effective batch = {CFG.batch_size * CFG.grad_accum})\n")

    for step in range(start_step, CFG.train_steps):
        for micro in range(CFG.grad_accum):
            try:
                x, labels = next(loader_iter)
            except StopIteration:
                loader_iter = iter(loader)
                x, labels   = next(loader_iter)

            x      = x.to(device,      non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            with torch.amp.autocast("cuda", dtype=torch.bfloat16,
                                    enabled=(device.type == "cuda")):
                out  = model(input_ids=x, labels=labels, token_type_ids = torch.zeros_like(x),)
                loss = out.loss / CFG.grad_accum

            loss.backward()
            loss_acc += out.loss.item()

        grad_norm = nn.utils.clip_grad_norm_(
            filter(lambda p: p.requires_grad, model.parameters()),
            CFG.max_grad_norm,
        )
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

        # ── Logging ───────────────────────────────────────────────────────
        if step % CFG.log_every == 0:
            avg_loss = loss_acc / (CFG.log_every * CFG.grad_accum) if step > 0 else loss_acc / CFG.grad_accum
            elapsed  = time.time() - t_start
            lr_now   = scheduler.get_last_lr()[0]

            entry = {
                "step":      step,
                "loss":      round(avg_loss, 4),
                "lr":        round(lr_now, 8),
                "grad_norm": round(grad_norm.item(), 4),
                "elapsed_s": round(elapsed, 1),
            }
            log_file.write(json.dumps(entry) + "\n")
            log_file.flush()

            print(f"  step {step:>5,} | loss {avg_loss:.4f} | "
                  f"lr {lr_now:.2e} | grad_norm {grad_norm.item():.3f} | "
                  f"{elapsed:.0f}s", flush=True)
            loss_acc = 0.0

            if step > 0 and avg_loss < best_loss:
                best_loss  = avg_loss
                best_path  = CKPT_DIR / "ckpt_best"
                _save(model, optimizer, scheduler, step, best_loss, best_path)
                print(f"  ✓ Best checkpoint saved (loss={best_loss:.4f})")

        # ── Periodic checkpoint ───────────────────────────────────────────
        if step % CFG.save_every == 0 and step > 0:
            ckpt_path = CKPT_DIR / f"ckpt_step{step}"
            _save(model, optimizer, scheduler, step, best_loss, ckpt_path)
            print(f"  Checkpoint → {ckpt_path}")

    # ── Final ─────────────────────────────────────────────────────────────
    final_path = CKPT_DIR / "ckpt_final"
    _save(model, optimizer, scheduler, CFG.train_steps, best_loss, final_path)
    log_file.close()
    print(f"\n  Done. Final adapter → {final_path}")


def _save(model, optimizer, scheduler, step, best_loss, path: Path):
    """Save LoRA adapter + training meta."""
    path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(path))          # saves adapter_config.json + weights
    meta = {
        "step":      step,
        "best_loss": best_loss,
        "model_id":  MODEL_ID,
    }
    (path / "train_meta.json").write_text(json.dumps(meta, indent=2))


# ════════════════════════════════════════════════════════════════════════════
# CHAT
# ════════════════════════════════════════════════════════════════════════════

def _extract_slp_response(text: str) -> str:
    import re
    text = re.sub(r"<\|think\|>.*?<\|endturn\|>", "", text, flags=re.DOTALL)
    for token in SPECIAL_TOKENS.values():
        text = text.replace(token, "")
    return text.strip()


def chat(checkpoint: str):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"\n{'='*60}")
    print(f"  SpeakMK1 (Gemma 3 4B + SLP LoRA) — CLI Chat")
    print(f"  Checkpoint : {checkpoint}")
    print(f"{'='*60}\n")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit           = True,
        bnb_4bit_quant_type    = "nf4",
        bnb_4bit_compute_dtype = torch.bfloat16,
    )

    base = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config = bnb_config,
        device_map          = "auto",
        torch_dtype         = torch.bfloat16,
        trust_remote_code   = True,
    )

    model = PeftModel.from_pretrained(base, checkpoint)
    model.eval()

    tok = get_tokenizer()
    eos_id     = tok.eos_token_id
    endturn_id = tok.encode(SPECIAL_TOKENS["turn_end"], add_special_tokens=False)[0]

    temperature = 0.8
    top_p       = 0.9
    max_new     = 200
    history: list[int] = []
    system_injected = False

    clinical_context = dict(
        age            = "6",
        disorder       = "articulation disorder",
        specific_error = "rhotacism",
        clinical_goal  = "correct /r/ production",
        strategy       = "minimal pairs and modelling",
    )

    def inject_system():
        nonlocal system_injected
        text    = (SPECIAL_TOKENS["system_start"] + " " +
                   SYSTEM_PROMPT_TEMPLATE.format(**clinical_context) + " " +
                   SPECIAL_TOKENS["turn_end"])
        history.extend(tok.encode(text, add_special_tokens=False))
        system_injected = True

    print("  Ready. /help for commands.\n")

    while True:
        try:
            user_input = input("Child: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n  Bye.")
            break

        if not user_input:
            continue

        if user_input.startswith("/"):
            parts = user_input.split()
            cmd   = parts[0].lower()
            if cmd == "/quit":
                break
            elif cmd == "/help":
                print("  /temp <f>  /top_p <f>  /maxlen <i>  /reset  /quit")
            elif cmd == "/temp"   and len(parts) > 1: temperature = float(parts[1])
            elif cmd == "/top_p"  and len(parts) > 1: top_p       = float(parts[1])
            elif cmd == "/maxlen" and len(parts) > 1: max_new     = int(parts[1])
            elif cmd == "/reset":
                history.clear()
                system_injected = False
                print("  History cleared.")
            continue

        if not system_injected:
            inject_system()

        child_text = (SPECIAL_TOKENS["child_start"] + " " + user_input + " " +
                      SPECIAL_TOKENS["turn_end"] + SPECIAL_TOKENS["think_start"])
        history.extend(tok.encode(child_text, add_special_tokens=False))

        # Trim to last 512 tokens, keeping system prompt
        if len(history) > 512:
            try:
                sys_end = history.index(
                    tok.encode(SPECIAL_TOKENS["turn_end"], add_special_tokens=False)[0]
                ) + 1
            except ValueError:
                sys_end = 0
            history = history[:sys_end] + history[sys_end:][-512 + sys_end:]

        input_ids = torch.tensor([history], dtype=torch.long, device=device)

        print("SLP: ", end="", flush=True)

        with torch.no_grad():
            output = model.generate(
                input_ids,
                max_new_tokens  = max_new,
                do_sample       = True,
                temperature     = temperature,
                top_p           = top_p,
                eos_token_id    = [eos_id, endturn_id],
                pad_token_id    = tok.pad_token_id,
            )

        generated_ids  = output[0, input_ids.shape[1]:].tolist()
        raw            = tok.decode(generated_ids, skip_special_tokens=False)
        clean          = _extract_slp_response(raw)
        print(clean)

        slp_text = (SPECIAL_TOKENS["slp_start"] + " " + clean + " " +
                    SPECIAL_TOKENS["turn_end"])
        history.extend(tok.encode(slp_text, add_special_tokens=False))


# ════════════════════════════════════════════════════════════════════════════
# CLI
# ════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="SpeakMK1 Gemma QLoRA fine-tune")
    sub    = parser.add_subparsers(dest="command")

    p_tok = sub.add_parser("tokenize", help="Tokenise SLP JSONL")
    p_tok.add_argument("--path", required=True, help="Path to SLP JSONL file")

    p_tr = sub.add_parser("train", help="Fine-tune with QLoRA")
    p_tr.add_argument("--resume", type=str, default=None,
                      help="Path to LoRA adapter dir to resume from")

    p_ch = sub.add_parser("chat", help="Interactive CLI chat")
    p_ch.add_argument("--checkpoint", required=True,
                      help="Path to saved LoRA adapter directory")

    args = parser.parse_args()

    if args.command == "tokenize":
        tokenize_slp_jsonl(args.path)
    elif args.command == "train":
        train(resume=args.resume)
    elif args.command == "chat":
        chat(args.checkpoint)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()