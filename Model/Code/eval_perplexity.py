"""
SpeakMK1 Per-Stage Perplexity Evaluator
----------------------------------------
Runs each checkpoint against its corresponding held-out data split
and produces a perplexity table for the report.

Usage:
    python eval_perplexity.py

Edit the STAGES config below to match your actual checkpoint paths
and held-out data memmap paths.
"""

import torch
import numpy as np
import math
from pathlib import Path

# ─────────────────────────────────────────────
# EDIT THESE TO MATCH YOUR SETUP
# ─────────────────────────────────────────────

STAGES = [
    {
        "name": "Stage 1 – TinyStories General Pretraining",
        "checkpoint": "../Model_files/checkpoints_v2.3.1/stage1/ckpt_final.pt",
        "data_paths": ["../Data/LLM_Data_updated/stage1_tinystories.bin"],
    },
    {
        "name": "Stage 2 – CHILDES Child-Directed Adaptation",
        "checkpoint": "../Model_files/checkpoints_v2.3.1/stage2/ckpt_final.pt",
        "data_paths": [
            "../Data/LLM_Data_updated/stage2_tinystories.bin",
            "../Data/LLM_Data_updated/stage2_childes.bin",
        ],
    },
    {
        "name": "Stage 3 – PubMed Central Clinical Knowledge",
        "checkpoint": "../Model_files/checkpoints_v2.3.1/stage3/ckpt_final.pt",
        "data_paths": ["../Data/LLM_Data_updated/stage3_pubmed.bin"],
    },
    {
        "name": "Stage 4 – SLP Dialogue Fine-Tuning",
        "checkpoint": "../Model_files/checkpoints_v2.3.1/stage4/ckpt_final.pt",
        "data_paths": [
            "../Data/LLM_Data_updated/stage4_slp_ids.bin",
        ],
    },
    {
        "name": "Stage 5 – Multimodal Reintegration",
        "checkpoint": "../Model_files/checkpoints_v2.3.1/stage5_audio/ckpt_final.pt",
        "data_paths": [
            "../Data/LLM_Data_updated/stage4_slp_ids.bin",
        ],
    },
]

DEVICE        = "cuda" if torch.cuda.is_available() else "cpu"
SEQ_LEN       = 512      # must match your training seq length
BATCH_SIZE    = 4        # reduce to 2 if you hit OOM
NUM_BATCHES   = 50       # 50 batches x 4 x 512 = ~100k tokens per stage, enough
VOCAB_SIZE    = 50283
PAD_TOKEN_ID  = 1

# ─────────────────────────────────────────────
# Model imports -- adjust to your actual module paths
# ─────────────────────────────────────────────
import sys
sys.path.insert(0, "../")  # EDIT THIS

from speak_mk1_llm import SpeakMK1LLM, SpeakMK1LLMConfig  # EDIT THIS


def load_model(checkpoint_path: str) -> torch.nn.Module:
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    config = SpeakMK1LLMConfig(
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

    model = SpeakMK1LLM(config)

    # handle both raw state_dict and wrapped checkpoint formats
    state = ckpt if not isinstance(ckpt, dict) else ckpt.get("model_state_dict", ckpt.get("model", ckpt))
    model.load_state_dict(state, strict=False)
    model.to(DEVICE)
    model.eval()
    return model


def load_data_batches(data_paths: list, num_batches: int):
    """
    Loads token sequences from one or more uint16 numpy memmap files.
    When multiple bins are provided, concatenates them before sampling.
    Returns list of (input_ids, labels) tensor pairs.
    """
    arrays = [np.memmap(p, dtype=np.uint16, mode="r") for p in data_paths]
    data = np.concatenate(arrays) if len(arrays) > 1 else arrays[0]
    total_tokens = len(data)
    stride = SEQ_LEN + 1  # +1 so we can shift for labels

    batches = []
    indices = list(range(0, total_tokens - stride, stride))
    np.random.shuffle(indices)

    batch_inputs, batch_labels = [], []

    for idx in indices:
        chunk = data[idx : idx + stride].astype(np.int64)
        if len(chunk) < stride:
            continue

        batch_inputs.append(chunk[:-1])   # input: tokens 0..N-1
        batch_labels.append(chunk[1:])    # labels: tokens 1..N

        if len(batch_inputs) == BATCH_SIZE:
            inp = torch.tensor(np.stack(batch_inputs), dtype=torch.long).to(DEVICE)
            lbl = torch.tensor(np.stack(batch_labels), dtype=torch.long).to(DEVICE)
            batches.append((inp, lbl))
            batch_inputs, batch_labels = [], []

        if len(batches) >= num_batches:
            break

    return batches


@torch.no_grad()
def compute_perplexity(model, batches):
    """
    Computes mean cross-entropy loss over all batches,
    then returns perplexity = exp(mean_loss).
    Ignores PAD tokens in the loss.
    """
    total_loss = 0.0
    total_tokens = 0

    loss_fn = torch.nn.CrossEntropyLoss(
        ignore_index=PAD_TOKEN_ID,
        reduction="sum"
    )

    for input_ids, labels in batches:
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            # text-only forward pass -- no audio features
            logits = model(input_ids)   # (B, T, vocab)

            # if your model returns a tuple (logits, aux_loss, ...) unwrap it
            if isinstance(logits, tuple):
                logits = logits[0]

        B, T, V = logits.shape
        loss = loss_fn(
            logits.reshape(B * T, V),
            labels.reshape(B * T)
        )

        # count non-pad tokens
        non_pad = (labels != PAD_TOKEN_ID).sum().item()
        total_loss += loss.item()
        total_tokens += non_pad

    mean_loss = total_loss / total_tokens
    perplexity = math.exp(mean_loss)
    return mean_loss, perplexity


def print_table(results):
    """Prints a formatted table and the LaTeX version for the report."""

    print("\n" + "=" * 75)
    print(f"{'Stage':<45} {'Loss':>8} {'Perplexity':>12}")
    print("=" * 75)
    for r in results:
        print(f"{r['stage']:<45} {r['loss']:>8.4f} {r['perplexity']:>12.2f}")
    print("=" * 75)

    # Word-friendly CSV for pasting into a table
    print("\n\nCSV (paste into Excel or Word table):\n")
    print("Stage,CE Loss,Perplexity")
    for r in results:
        print(f"{r['stage']},{r['loss']:.4f},{r['perplexity']:.2f}")


def main():
    results = []

    for stage in STAGES:
        name      = stage["name"]
        ckpt_path = stage["checkpoint"]
        data_paths = stage["data_paths"]

        print(f"\nEvaluating: {name}")
        print(f"  Checkpoint : {ckpt_path}")
        print(f"  Data       : {data_paths}")

        if not Path(ckpt_path).exists():
            print(f"  SKIPPED -- checkpoint not found")
            continue
        missing = [p for p in data_paths if not Path(p).exists()]
        if missing:
            print(f"  SKIPPED -- data file(s) not found: {missing}")
            continue

        model   = load_model(ckpt_path)
        batches = load_data_batches(data_paths, NUM_BATCHES)

        if not batches:
            print(f"  SKIPPED -- no valid batches loaded from data file")
            continue

        print(f"  Loaded {len(batches)} batches ({len(batches)*BATCH_SIZE*SEQ_LEN:,} tokens)")
        loss, ppl = compute_perplexity(model, batches)
        print(f"  Loss: {loss:.4f}  |  Perplexity: {ppl:.2f}")

        results.append({
            "stage":      name,
            "loss":       loss,
            "perplexity": ppl,
        })

        # free VRAM between stages
        del model
        torch.cuda.empty_cache()

    if results:
        print_table(results)
    else:
        print("\nNo results -- check your checkpoint and data paths.")


if __name__ == "__main__":
    main()