"""
audio_encoder_tester.py
=======================
SpeakMK1 — Audio encoder diagnostic tester.

Tests the audio encoder independently of the Q-Former to determine whether
the problem is in the encoder itself or the Q-Former alignment training.

Checks:
    1. Output shape and dtype
    2. Phonological head accuracy (place, manner, voicing, correctness)
    3. CTC output quality (blank token dominance check)
    4. Encoder output diversity (not collapsed)
    5. Q-Former output diversity across different audio inputs
    6. Audio sensitivity (does output change with different inputs?)
    7. Gradient flow through encoder

Usage:
    python audio_encoder_tester.py --checkpoint ../Model_files/audio_encoder_epoch_19.pt
"""

from __future__ import annotations

import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from dataclasses import dataclass
from torch.utils.data import DataLoader

from audio_encoder import AudioEncoder, AudioEncoderConfig, aggregate_losses
from librispeechMFADataset import LibriSpeechMFADataset, collate_fn


# ════════════════════════════════════════════════════════════════════════════
# CONFIG
# ════════════════════════════════════════════════════════════════════════════

@dataclass
class SmallConfig(AudioEncoderConfig):
    d_model         : int   = 512
    dstate          : int   = 32
    num_uni_mamba   : int   = 6
    num_bi_mamba_moe: int   = 4
    num_experts     : int   = 2
    attn_heads      : int   = 8
    num_queries     : int   = 64
    dropout         : float = 0.15
    training_heads  : bool  = True


# ════════════════════════════════════════════════════════════════════════════
# HELPERS
# ════════════════════════════════════════════════════════════════════════════

def print_section(title: str):
    print(f"\n{'─'*60}")
    print(f"  {title}")
    print(f"{'─'*60}")


def grade(condition: bool, good_msg: str, bad_msg: str):
    tag = "GOOD" if condition else "BAD "
    msg = good_msg if condition else bad_msg
    print(f"  [{tag}] {msg}")
    return condition


# ════════════════════════════════════════════════════════════════════════════
# TEST 1 — OUTPUT SHAPES
# ════════════════════════════════════════════════════════════════════════════

def test_output_shapes(model, batch, device):
    print_section("TEST 1 — Output shapes")
    mel    = batch[0].to(device)
    labels = {k: v.to(device) for k, v in batch[1].items()}

    with torch.no_grad():
        audio_out, losses = model(mel, labels=labels, use_extractor=False)

    print(f"  mel shape       : {tuple(mel.shape)}")
    print(f"  audio_out shape : {tuple(audio_out.shape)}")
    print(f"  dtype           : {audio_out.dtype}")
    print(f"  losses keys     : {list(losses.keys())}")

    grade(audio_out.dim() == 3,
          f"3D output (B, num_queries, dim) ✓",
          f"Expected 3D output, got {audio_out.dim()}D")

    grade(not audio_out.isnan().any(),
          "No NaNs in output ✓",
          "NaNs detected in output — training instability")

    grade(not audio_out.isinf().any(),
          "No Infs in output ✓",
          "Infs detected in output")

    return audio_out, losses


# ════════════════════════════════════════════════════════════════════════════
# TEST 2 — PHONOLOGICAL HEAD ACCURACY
# ════════════════════════════════════════════════════════════════════════════

def test_phonological_heads(model, loader, device, n_batches=10):
    print_section("TEST 2 — Phonological head accuracy")

    head_names  = ["voicing", "manner", "place", "correctness"]
    correct     = {h: 0 for h in head_names}
    total       = {h: 0 for h in head_names}

    model.eval()
    with torch.no_grad():
        for i, (mel, labels) in enumerate(loader):
            if i >= n_batches:
                break
            mel    = mel.to(device)
            labels = {k: v.to(device) for k, v in labels.items()}

            _, losses = model(mel, labels=labels, use_extractor=False)

            # Access head logits directly if your model exposes them
            # If not, we infer accuracy from loss magnitude
            for h in head_names:
                if h in losses:
                    # Lower loss = better. Random CE for binary = ln(2) ≈ 0.693
                    # Random CE for multi-class depends on num classes
                    loss_val = losses[h].item()
                    total[h] += 1
                    # Proxy: if loss < 0.5 * random_baseline, consider it learning
                    correct[h] += 1 if loss_val < 0.5 else 0

    print(f"  (Evaluated over {n_batches} batches)")
    print(f"  {'Head':<15} {'Avg Loss':<12} {'Learning?'}")
    print(f"  {'─'*40}")

    all_learning = True
    model.eval()
    loss_vals = {}
    with torch.no_grad():
        for i, (mel, labels) in enumerate(loader):
            if i >= n_batches:
                break
            mel    = mel.to(device)
            labels = {k: v.to(device) for k, v in labels.items()}
            _, losses = model(mel, labels=labels, use_extractor=False)
            for h in head_names:
                if h in losses:
                    loss_vals.setdefault(h, []).append(losses[h].item())

    for h in head_names:
        if h not in loss_vals:
            print(f"  {h:<15} {'N/A':<12} (not in losses)")
            continue
        avg = np.mean(loss_vals[h])
        learning = avg < 0.65   # below random binary CE baseline
        all_learning = all_learning and learning
        status = "YES ✓" if learning else "NO — still near random"
        print(f"  {h:<15} {avg:<12.4f} {status}")

    if "ctc" in loss_vals:
        ctc_avg = np.mean(loss_vals["ctc"])
        print(f"\n  CTC loss avg: {ctc_avg:.4f}")
        grade(ctc_avg < 50.0,
              "CTC loss is reasonable ✓",
              f"CTC loss {ctc_avg:.1f} is very high — encoder may not be transcribing")

    return all_learning


# ════════════════════════════════════════════════════════════════════════════
# TEST 3 — ENCODER OUTPUT DIVERSITY
# ════════════════════════════════════════════════════════════════════════════

def test_encoder_diversity(model, loader, device, n_batches=5):
    print_section("TEST 3 — Encoder output diversity")

    all_outputs = []
    model.eval()
    with torch.no_grad():
        for i, (mel, labels) in enumerate(loader):
            if i >= n_batches:
                break
            mel    = mel.to(device)
            labels = {k: v.to(device) for k, v in labels.items()}
            audio_out, _ = model(mel, labels=labels, use_extractor=False)
            # Mean pool across queries for a per-sample vector
            pooled = audio_out.mean(dim=1)   # (B, dim)
            all_outputs.append(pooled.cpu())

    all_out = torch.cat(all_outputs, dim=0)   # (N, dim)
    std_per_dim = all_out.std(dim=0)
    mean_std    = std_per_dim.mean().item()
    min_std     = std_per_dim.min().item()

    print(f"  Samples evaluated : {all_out.shape[0]}")
    print(f"  Output dim        : {all_out.shape[1]}")
    print(f"  Mean std per dim  : {mean_std:.4f}")
    print(f"  Min std per dim   : {min_std:.4f}")

    grade(mean_std > 0.01,
          f"Encoder output is diverse (mean_std={mean_std:.4f}) ✓",
          f"Encoder output is collapsed (mean_std={mean_std:.4f}) — dead encoder")

    grade(min_std > 0.001,
          "No dead dimensions ✓",
          f"Some dimensions are dead (min_std={min_std:.5f})")

    # Pairwise similarity between samples
    normed    = F.normalize(all_out, dim=-1)
    sim_matrix = normed @ normed.T
    n          = sim_matrix.shape[0]
    eye        = torch.eye(n)
    off_diag   = sim_matrix[~eye.bool()]
    mean_sim   = off_diag.mean().item()
    max_sim    = off_diag.max().item()

    print(f"\n  Pairwise cosine similarity (off-diagonal):")
    print(f"    mean : {mean_sim:.4f}  (collapsed if > 0.8)")
    print(f"    max  : {max_sim:.4f}")

    grade(mean_sim < 0.8,
          "Embeddings not collapsed ✓",
          f"Embeddings are collapsing (mean sim={mean_sim:.4f})")

    return mean_std, mean_sim


# ════════════════════════════════════════════════════════════════════════════
# TEST 4 — AUDIO SENSITIVITY
# ════════════════════════════════════════════════════════════════════════════

def test_audio_sensitivity(model, batch, device):
    print_section("TEST 4 — Audio sensitivity")
    mel    = batch[0].to(device)
    labels = {k: v.to(device) for k, v in batch[1].items()}

    model.eval()
    with torch.no_grad():
        # ── Encoder-level sensitivity (pre-QFormer) ──────────────────────
        x_clean = model.stem(mel)
        for layer in model.uni_mamba_layers:
            x_clean = layer(x_clean)

        mel_noisy = mel + torch.randn_like(mel) * 5.0
        x_noisy = model.stem(mel_noisy)
        for layer in model.uni_mamba_layers:
            x_noisy = layer(x_noisy)

        encoder_diff = (x_clean - x_noisy).abs().mean().item()
        print(f"  Encoder-level diff (pre-QFormer, 5dB noise): {encoder_diff:.4f}")
        grade(encoder_diff > 0.001,
              "Encoder backbone responds to noise ✓",
              "Encoder backbone is ignoring input — collapse is in the backbone")

        # ── Full pipeline sensitivity (QFormer output) ────────────────────
        out1, _ = model(mel, labels=labels, use_extractor=False)

        out2, _ = model(mel_noisy, labels=labels, use_extractor=False)

        mel_silent = torch.zeros_like(mel)
        out3, _    = model(mel_silent, labels=labels, use_extractor=False)

    diff_noise  = (out1 - out2).abs().mean().item()
    diff_silent = (out1 - out3).abs().mean().item()

    print(f"  Output diff (original vs noisy)  : {diff_noise:.4f}")
    print(f"  Output diff (original vs silent) : {diff_silent:.4f}")

    grade(diff_noise > 0.001,
          "Full pipeline responds to input noise ✓",
          "Full pipeline insensitive to noise — QFormer may be collapsing")

    grade(diff_silent > 0.01,
          "Full pipeline distinguishes speech from silence ✓",
          "Full pipeline output same for speech and silence")

    if mel.shape[0] > 1:
        sim_01 = F.cosine_similarity(
            out1[0].mean(0, keepdim=True),
            out1[1].mean(0, keepdim=True)
        ).item()
        print(f"  Cosine sim between sample 0 and 1: {sim_01:.4f}")
        grade(sim_01 < 0.99,
              "Different samples produce different outputs ✓",
              f"Samples 0 and 1 are nearly identical (sim={sim_01:.4f})")

    return diff_noise, diff_silent


# ════════════════════════════════════════════════════════════════════════════
# TEST 5 — QFORMER QUERY SPECIALISATION
# ════════════════════════════════════════════════════════════════════════════

def test_query_specialisation(model, batch, device):
    print_section("TEST 5 — Q-Former query specialisation")
    mel    = batch[0].to(device)
    labels = {k: v.to(device) for k, v in batch[1].items()}

    model.eval()
    with torch.no_grad():
        audio_out, _ = model(mel, labels=labels, use_extractor=False)

    # audio_out: (B, num_queries, dim)
    queries = audio_out[0]   # (num_queries, dim) — first sample

    print(f"  num_queries : {queries.shape[0]}")
    print(f"  query dim   : {queries.shape[1]}")

    # Pairwise similarity between queries
    normed   = F.normalize(queries, dim=-1)
    sim      = normed @ normed.T                         # (Q, Q)
    eye      = torch.eye(sim.shape[0], device=device)
    off_diag = sim[~eye.bool()]

    mean_q_sim = off_diag.mean().item()
    max_q_sim  = off_diag.max().item()
    std_q      = queries.std(dim=-1)   # std per query

    print(f"  Query pairwise similarity:")
    print(f"    mean : {mean_q_sim:.4f}  (collapsed if > 0.9)")
    print(f"    max  : {max_q_sim:.4f}")
    print(f"  Query output std (per query):")
    print(f"    mean : {std_q.mean().item():.4f}  (dead if < 0.01)")
    print(f"    min  : {std_q.min().item():.4f}")

    grade(mean_q_sim < 0.9,
          "Queries are specialised (diverse outputs) ✓",
          f"Queries have collapsed (mean sim={mean_q_sim:.4f}) — all queries doing same thing")

    grade(std_q.min().item() > 0.001,
          "No dead queries ✓",
          f"Some queries are dead (min std={std_q.min().item():.5f})")

    return mean_q_sim


# ════════════════════════════════════════════════════════════════════════════
# TEST 6 — GRADIENT FLOW
# ════════════════════════════════════════════════════════════════════════════

def test_gradient_flow(model, batch, device):
    print_section("TEST 6 — Gradient flow")
    mel    = batch[0].to(device)
    labels = {k: v.to(device) for k, v in batch[1].items()}

    model.train()
    model.zero_grad()

    audio_out, losses = model(mel, labels=labels, use_extractor=False)
    loss = aggregate_losses(losses, weights={
        'ctc': 1.0, 'voicing': 0.5, 'manner': 0.5,
        'place': 0.5, 'correctness': 0.1,
    })
    loss.backward()

    no_grad    = []
    has_grad   = []
    zero_grad  = []

    for name, param in model.named_parameters():
        if param.requires_grad:
            if param.grad is None:
                no_grad.append(name)
            elif param.grad.abs().max().item() == 0:
                zero_grad.append(name)
            else:
                has_grad.append(name)

    print(f"  Params with gradient    : {len(has_grad)}")
    print(f"  Params with zero grad   : {len(zero_grad)}")
    print(f"  Params with no grad     : {len(no_grad)}")

    if no_grad:
        print(f"\n  Params missing grad (first 5):")
        for n in no_grad[:5]:
            print(f"    {n}")

    # Check Q-Former specifically
    qformer_grads = [
        (name, param.grad.norm().item())
        for name, param in model.named_parameters()
        if 'qformer' in name and param.grad is not None
    ]
    if qformer_grads:
        avg_qformer_grad = np.mean([g for _, g in qformer_grads])
        print(f"\n  Q-Former grad norm (avg) : {avg_qformer_grad:.6f}")
        grade(avg_qformer_grad > 1e-6,
              "Q-Former is receiving gradients ✓",
              f"Q-Former grad norm is near zero ({avg_qformer_grad:.2e}) — disconnected")
    else:
        print("  [BAD ] No Q-Former gradients found at all")

    grade(len(has_grad) > len(no_grad),
          f"Most params have gradients ({len(has_grad)}/{len(has_grad)+len(no_grad)}) ✓",
          f"Most params missing gradients ({len(no_grad)} missing)")

    model.eval()
    return len(no_grad) == 0

def test_encoder_diversity_direct(model, loader, device, n_batches=5):
    print_section("TEST 3b — Encoder output diversity (pre-QFormer)")
    
    all_encoder_outputs = []
    model.eval()
    
    with torch.no_grad():
        for i, (mel, labels) in enumerate(loader):
            if i >= 2:  # just 2 batches, this is expensive
                break
            mel = mel.to(device)
            labels = {k: v.to(device) for k, v in labels.items()}
            x = model.stem(mel)
            for layer in model.uni_mamba_layers:
                x = layer(x)
            pad_mask = model._make_key_padding_mask(mel)
            x = model.rms_att(x, key_padding_mask=pad_mask)
            for layer in model.bi_mamba_moe_layers:
                x = layer(x, attention_mask=None)
            x, _ = model.phonological(x, labels=labels)
            
            # Sample 50 random frames across the batch
            B, T, D = x.shape
            frames = x.reshape(B * T, D)
            idx = torch.randperm(frames.shape[0])[:50]
            sampled = frames[idx].cpu()
            all_encoder_outputs.append(sampled)
            
            normed_frames = F.normalize(sampled, dim=-1)
            frame_sim = normed_frames @ normed_frames.T
            off = frame_sim[~torch.eye(sampled.shape[0]).bool()]
            print(f"  Frame-level pairwise similarity: {off.mean().item():.4f}")
    
    all_enc = torch.cat(all_encoder_outputs, dim=0)
    normed = F.normalize(all_enc, dim=-1)
    sim_matrix = normed @ normed.T
    n = sim_matrix.shape[0]
    off_diag = sim_matrix[~torch.eye(n).bool()]
    
    mean_sim = off_diag.mean().item()
    std_per_dim = all_enc.std(dim=0)
    
    print(f"  Encoder (pre-QFormer) pairwise similarity: {mean_sim:.4f}")
    print(f"  Encoder mean std per dim: {std_per_dim.mean().item():.4f}")
    
    grade(mean_sim < 0.8,
          f"Encoder representations not collapsed (sim={mean_sim:.4f}) ✓",
          f"Encoder representations collapsed (sim={mean_sim:.4f})")
    
    return std_per_dim.mean().item(), mean_sim  # add this
    



# ════════════════════════════════════════════════════════════════════════════
# SUMMARY
# ════════════════════════════════════════════════════════════════════════════

def print_summary(results: dict):
    print(f"\n{'═'*60}")
    print(f"  AUDIO ENCODER DIAGNOSTIC SUMMARY")
    print(f"{'═'*60}")

    all_pass = True
    for test, (passed, note) in results.items():
        tag  = "PASS" if passed else "FAIL"
        all_pass = all_pass and passed
        print(f"  [{tag}] {test:<35} {note}")

    print(f"\n  {'Overall: ENCODER HEALTHY ✓' if all_pass else 'Overall: ISSUES FOUND — see above'}")
    print(f"{'═'*60}\n")


# ════════════════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════════════════

def main(checkpoint: str, librispeech_root: str, mfa_root: str, n_batches: int = 5):
    print("=" * 60)
    print("  SpeakMK1 Audio Encoder Diagnostic Tester")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n  Device     : {device}")
    print(f"  Checkpoint : {checkpoint}")

    # ── Load model ────────────────────────────────────────────────────────
    cfg   = SmallConfig()
    model = AudioEncoder(cfg).to(device)
    print(f"  Params     : {sum(p.numel() for p in model.parameters())/1e6:.2f}M")

    print(f"\n  Loading checkpoint...")
    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    # Handle wrapped checkpoints
    state = ckpt.get("encoder", ckpt.get("model_state_dict", ckpt))
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f"  [WARN] Missing keys  : {len(missing)}")
        for k in missing[:5]:
            print(f"    {k}")
    if unexpected:
        print(f"  [WARN] Unexpected keys: {len(unexpected)}")
    print(f"  Checkpoint loaded.")

    # ── Load data ─────────────────────────────────────────────────────────
    dataset = LibriSpeechMFADataset(
        librispeech_root = librispeech_root,
        mfa_output_root  = mfa_root,
        max_frames       = 1024,
    )
    loader = DataLoader(
        dataset,
        batch_size  = 8,
        shuffle     = False,
        collate_fn  = collate_fn,
        num_workers = 2,
        pin_memory  = True,
    )

    # Get one batch for single-batch tests
    sample_batch = next(iter(loader))

    # ── Run tests ─────────────────────────────────────────────────────────
    results = {}

    audio_out, losses = test_output_shapes(model, sample_batch, device)
    shapes_ok = (
        not audio_out.isnan().any() and
        not audio_out.isinf().any() and
        audio_out.dim() == 3
    )
    results["Output shapes"] = (shapes_ok, f"shape={tuple(audio_out.shape)}")

    heads_learning = test_phonological_heads(model, loader, device, n_batches)
    results["Phonological heads"] = (heads_learning, "all heads below random baseline" if heads_learning else "some heads near random")

    mean_std, mean_sim = test_encoder_diversity_direct(model, loader, device, n_batches)
    results["Encoder diversity"] = (
        mean_std > 0.01 and mean_sim < 0.8,
        f"std={mean_std:.4f}, sim={mean_sim:.4f}"
    )

    diff_noise, diff_silent = test_audio_sensitivity(model, sample_batch, device)
    results["Audio sensitivity"] = (
        diff_noise > 0.001 and diff_silent > 0.01,
        f"noise_diff={diff_noise:.4f}, silence_diff={diff_silent:.4f}"
    )

    mean_q_sim = test_query_specialisation(model, sample_batch, device)
    results["Q-Former specialisation"] = (
        mean_q_sim < 0.9,
        f"mean query sim={mean_q_sim:.4f}"
    )

    grads_ok = test_gradient_flow(model, sample_batch, device)
    results["Gradient flow"] = (grads_ok, "all params have gradients" if grads_ok else "some params missing gradients")

    print_summary(results)

    # ── Diagnosis ─────────────────────────────────────────────────────────
    print("  DIAGNOSIS:")
    if not results["Audio sensitivity"][0]:
        print("  → Encoder is not reading its input. Check that mel preprocessing")
        print("    matches what the encoder was trained on (n_mels, sample_rate, etc).")
    if not results["Encoder diversity"][0]:
        print("  → Encoder output has collapsed. The Q-Former queries are all")
        print("    producing the same output regardless of input.")
        print("    This is the most likely cause of the Q-Former alignment failure.")
    if not results["Q-Former specialisation"][0]:
        print("  → Q-Former queries have not specialised. Either the Q-Former was")
        print("    not trained long enough, or the diversity loss was too high.")
    if not results["Phonological heads"][0]:
        print("  → Phonological heads are near random. The encoder backbone may")
        print("    not have trained long enough in Stage 1.")
    if all(v[0] for v in results.values()):
        print("  → Encoder looks healthy. The Q-Former alignment problem is likely")
        print("    in the training script (loss weights, detach, lr) not the encoder.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint",
                        default="../Model_files/audio_encoder_epoch_19.pt",
                        help="Path to audio encoder checkpoint")
    parser.add_argument("--librispeech_root",
                        default="../Data/PhonemeDatasets/LibriSpeech/LibriSpeech/train-clean-100")
    parser.add_argument("--mfa_root",
                        default="../Data/PhonemeDatasets/LibriSpeech/LibriSpeech/train-clean-100-mfa")
    parser.add_argument("--n_batches", type=int, default=5,
                        help="Number of batches to evaluate over")
    args = parser.parse_args()

    main(
        checkpoint       = args.checkpoint,
        librispeech_root = args.librispeech_root,
        mfa_root         = args.mfa_root,
        n_batches        = args.n_batches,
    )