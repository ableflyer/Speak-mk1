"""
train_visual_encoder.py
unfortunately it's not used
=======================
SpeakMK1 — Visual encoder training.

What gets trained and when
--------------------------
The visual encoder has no labelled visual data of its own. It is trained
jointly with the LLM using text CE loss as the supervision signal — the
same loss that drives the LLM also flows back through the pooler into the
embedder. This is identical to how BLIP-2 trains its Q-Former.

Stage 3 — Pooler only (run this first):
    - LLM backbone: FROZEN
    - VisualEncoder.embedder: FROZEN
    - VisualEncoder.pooler: TRAINABLE
    - Loss: CE on text tokens only
    - Purpose: let the pooler queries learn what to look for before
      the embedder gets any gradient signal
    - LR: 1e-4
    - Duration: short — 1-3 epochs over your existing text+audio data
      with synthetic visual frames (webcam frames or random noise)

Stage 4 — Full unfreeze (run after Stage 3):
    - LLM backbone: TRAINABLE (low LR)
    - VisualEncoder.embedder: TRAINABLE
    - VisualEncoder.pooler: TRAINABLE
    - Loss: CE on text tokens only
    - LR: 2e-5 (LLM), 5e-5 (visual encoder)
    - Duration: your existing Stage 4 instruction tuning data

Data requirements
-----------------
You do NOT need labelled video data. You need:
    - Any video frames from a webcam (even random mouth movements)
    - Paired text (the existing CHILDES/TinyStories/Alpaca data)
    - The audio tokens from your already-trained audio encoder (or zeros
      if you want to train visual in isolation first)

The visual encoder learns to produce tokens that are useful for the LLM
to predict the next text token. This is self-supervised — the text
provides the supervision signal.

Usage
-----
    # Stage 3 (pooler only, LLM frozen)
    python train_visual_encoder.py --stage 3 --llm_ckpt path/to/llm.pt

    # Stage 4 (full unfreeze)
    python train_visual_encoder.py --stage 4 --llm_ckpt path/to/stage3.pt
"""

from __future__ import annotations

import argparse
import math
import os
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from vision_part import MouthTracker, RunningMode
from vision_encoder import VisualEncoder, TARGET_DIM
from speak_mk1_llm import SpeakMK1LLM, SpeakMK1LLMConfig


# ════════════════════════════════════════════════════════════════════════════
# 0.  CONFIG
# ════════════════════════════════════════════════════════════════════════════

LLM_CONFIG = SpeakMK1LLMConfig(
    vocab_size    = 50283,
    d_model       = 512,
    d_state       = 64,
    num_blocks    = 6,
    nheads_ssm    = 8,
    nheads_attn   = 8,
    top_k_audio   = 32,
    num_experts   = 8,
    top_k_experts = 2,
)

VISUAL_CONFIG = dict(
    d_model         = 512,
    n_visual_tokens = 8,
    n_heads         = 8,
)

TRAIN_CONFIG = {
    3: dict(
        lr_visual  = 1e-4,
        lr_llm     = 0.0,       # LLM frozen in Stage 3
        epochs     = 3,
        warmup_steps = 100,
        grad_clip  = 1.0,
        log_every  = 10,
        save_every = 500,
    ),
    4: dict(
        lr_visual  = 5e-5,
        lr_llm     = 2e-5,
        epochs     = 2,
        warmup_steps = 50,
        grad_clip  = 1.0,
        log_every  = 10,
        save_every = 500,
    ),
}


# ════════════════════════════════════════════════════════════════════════════
# 1.  SYNTHETIC VISUAL DATA GENERATOR
# ════════════════════════════════════════════════════════════════════════════

class WebcamFrameBuffer:
    """
    Collects real webcam frames into windows of T frames each.

    If no webcam is available (or --no_webcam is set), generates synthetic
    random frame vectors instead. Synthetic frames are still useful for
    verifying gradient flow — the pooler will learn nothing meaningful from
    noise, but the training loop and shapes will be validated.

    In production, replace this with your actual video dataset loader.
    """

    def __init__(
        self,
        window       : int  = 32,
        camera_index : int  = 0,
        use_webcam   : bool = True,
    ):
        self.window     = window
        self.use_webcam = use_webcam
        self.tracker    = MouthTracker(running_mode=RunningMode.IMAGE)
        self.cap        = None

        if use_webcam:
            self.cap = cv2.VideoCapture(camera_index)
            if not self.cap.isOpened():
                print("[WebcamFrameBuffer] Webcam not available — falling back to synthetic frames.")
                self.use_webcam = False

    def get_window(self, device: torch.device) -> torch.Tensor:
        """
        Returns (1, window, 180) tensor of frame features.
        Uses real webcam frames if available, else synthetic.
        """
        if not self.use_webcam:
            return torch.randn(1, self.window, TARGET_DIM, device=device)

        frames = []
        for _ in range(self.window):
            ok, frame = self.cap.read()
            if not ok:
                frames.append(np.zeros(TARGET_DIM, dtype=np.float32))
            else:
                result = self.tracker.process_frame(frame)
                frames.append(result.to_vector())

        vecs = np.stack(frames, axis=0)                         # (T, 180)
        return torch.from_numpy(vecs).unsqueeze(0).to(device)   # (1, T, 180)

    def close(self):
        self.tracker.close()
        if self.cap is not None:
            self.cap.release()


# ════════════════════════════════════════════════════════════════════════════
# 2.  SYNTHETIC TEXT BATCH GENERATOR
# ════════════════════════════════════════════════════════════════════════════

def synthetic_text_batch(
    batch_size : int,
    seq_len    : int,
    vocab_size : int,
    device     : torch.device,
):
    """
    Generates a random text batch for smoke-testing the training loop.

    Replace this with your real DataLoader over tokenized CHILDES/TinyStories
    or Alpaca data in actual training.

    Returns (input_ids, labels) both of shape (B, seq_len).
    """
    ids    = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
    labels = ids.clone()
    return ids, labels


# ════════════════════════════════════════════════════════════════════════════
# 3.  FORWARD PASS
# ════════════════════════════════════════════════════════════════════════════

def forward_with_visual(
    llm           : SpeakMK1LLM,
    visual_encoder: VisualEncoder,
    input_ids     : torch.Tensor,
    labels        : torch.Tensor,
    mouth_frames  : torch.Tensor,
    aux_weight    : float = 1e-2,
):
    """
    Full forward pass: visual tokens prepended to text embeddings.

    Sequence layout: [ visual (8) | text (L) ]

    Labels are padded with -100 at visual positions so CE loss only
    counts over text positions.
    """
    B, L = input_ids.shape

    # Visual tokens: (B, 8, 512)
    # Expand mouth_frames to match batch size if needed
    if mouth_frames.size(0) == 1 and B > 1:
        mouth_frames = mouth_frames.expand(B, -1, -1)
    visual_tokens = visual_encoder(mouth_frames)   # (B, 8, 512)

    # Text embeddings: (B, L, 512)
    text_embeds = llm.embed_proj(llm.embedding(input_ids))

    # Concatenate: (B, 8+L, 512)
    inputs_embeds = torch.cat([visual_tokens, text_embeds], dim=1)

    # Shift labels: -100 for visual positions
    n_vis    = visual_tokens.size(1)
    ignore   = labels.new_full((B, n_vis), -100)
    labels_  = torch.cat([ignore, labels], dim=1)   # (B, 8+L)

    logits, loss, aux = llm(
        input_ids     = None,
        inputs_embeds = inputs_embeds,
        labels        = labels_,
    )
    return loss, aux


# ════════════════════════════════════════════════════════════════════════════
# 4.  TRAINING LOOP
# ════════════════════════════════════════════════════════════════════════════

def train(
    stage       : int,
    llm_ckpt    : str,
    save_dir    : str  = "checkpoints/visual",
    batch_size  : int  = 4,
    seq_len     : int  = 128,
    use_webcam  : bool = True,
    camera_idx  : int  = 0,
    steps       : int  = 1000,    # steps per epoch; replace with len(dataloader)
):
    assert stage in (3, 4), "stage must be 3 or 4"
    cfg = TRAIN_CONFIG[stage]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*60}")
    print(f"  SpeakMK1 Visual Encoder — Stage {stage} Training")
    print(f"{'='*60}")
    print(f"  Device : {device}")
    print(f"  Stage  : {stage}")
    print(f"  Steps  : {steps} per epoch x {cfg['epochs']} epochs")

    os.makedirs(save_dir, exist_ok=True)

    # ── Load LLM ─────────────────────────────────────────────────────────
    print(f"\n  Loading LLM from {llm_ckpt} ...")
    llm = SpeakMK1LLM(LLM_CONFIG).to(device)
    ckpt = torch.load(llm_ckpt, map_location=device)
    # Handle both raw state_dict and wrapped checkpoint
    state = ckpt.get("model_state_dict", ckpt)
    llm.load_state_dict(state, strict=False)
    print(f"  LLM loaded.")

    # ── Build visual encoder ──────────────────────────────────────────────
    visual_encoder = VisualEncoder(**VISUAL_CONFIG).to(device)

    if stage == 3:
        # Freeze LLM entirely
        for p in llm.parameters():
            p.requires_grad = False
        # Freeze visual embedder, train pooler only
        visual_encoder.freeze_embedder()
        print(f"  LLM frozen. Pooler trainable params: {visual_encoder.trainable_params():,}")

        optimizer = AdamW(
            filter(lambda p: p.requires_grad, visual_encoder.parameters()),
            lr=cfg["lr_visual"],
            weight_decay=0.01,
        )

    else:  # stage 4
        # Load Stage 3 visual encoder checkpoint if it exists
        stage3_path = Path(save_dir) / "visual_encoder_stage3_final.pt"
        if stage3_path.exists():
            print(f"  Loading Stage 3 visual encoder from {stage3_path} ...")
            visual_encoder.load_state_dict(torch.load(stage3_path, map_location=device))
        else:
            print("  [WARNING] No Stage 3 checkpoint found — starting visual encoder from scratch.")

        # Unfreeze everything
        visual_encoder.unfreeze_all()
        for p in llm.parameters():
            p.requires_grad = True

        optimizer = AdamW(
            [
                {"params": visual_encoder.parameters(), "lr": cfg["lr_visual"]},
                {"params": llm.parameters(),            "lr": cfg["lr_llm"]},
            ],
            weight_decay=0.01,
        )
        print(f"  Visual encoder trainable: {visual_encoder.trainable_params():,}")
        print(f"  LLM trainable: {sum(p.numel() for p in llm.parameters() if p.requires_grad):,}")

    total_steps = cfg["epochs"] * steps
    scheduler   = CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=1e-6)

    # ── Data sources ──────────────────────────────────────────────────────
    frame_buf = WebcamFrameBuffer(
        window       = 32,
        camera_index = camera_idx,
        use_webcam   = use_webcam,
    )

    # ── Training loop ─────────────────────────────────────────────────────
    llm.train()
    visual_encoder.train()

    global_step = 0
    for epoch in range(cfg["epochs"]):
        epoch_loss = 0.0
        t0 = time.time()

        for step in range(steps):
            # Get visual frames (real webcam or synthetic)
            mouth_frames = frame_buf.get_window(device)   # (1, 32, 180)

            # Get text batch (replace with real DataLoader)
            input_ids, labels = synthetic_text_batch(
                batch_size = batch_size,
                seq_len    = seq_len,
                vocab_size = LLM_CONFIG.vocab_size,
                device     = device,
            )

            # Warmup LR
            if global_step < cfg["warmup_steps"]:
                warmup_factor = global_step / max(1, cfg["warmup_steps"])
                for pg in optimizer.param_groups:
                    pg["lr"] = pg["lr"] * warmup_factor

            optimizer.zero_grad()

            loss, aux = forward_with_visual(
                llm            = llm,
                visual_encoder = visual_encoder,
                input_ids      = input_ids,
                labels         = labels,
                mouth_frames   = mouth_frames,
                aux_weight     = LLM_CONFIG.aux_loss_weight,
            )

            total_loss = loss + LLM_CONFIG.aux_loss_weight * aux
            total_loss.backward()

            nn.utils.clip_grad_norm_(
                list(visual_encoder.parameters()) + list(llm.parameters()),
                cfg["grad_clip"],
            )

            optimizer.step()
            if global_step >= cfg["warmup_steps"]:
                scheduler.step()

            epoch_loss  += total_loss.item()
            global_step += 1

            if global_step % cfg["log_every"] == 0:
                elapsed = time.time() - t0
                avg_loss = epoch_loss / (step + 1)
                lr_vis = optimizer.param_groups[0]["lr"]
                print(
                    f"  epoch {epoch+1}/{cfg['epochs']}  "
                    f"step {global_step:5d}  "
                    f"loss {avg_loss:.4f}  "
                    f"lr_vis {lr_vis:.2e}  "
                    f"({elapsed:.1f}s)"
                )

            if global_step % cfg["save_every"] == 0:
                ckpt_path = Path(save_dir) / f"visual_encoder_stage{stage}_step{global_step}.pt"
                torch.save(visual_encoder.state_dict(), ckpt_path)
                print(f"  Saved: {ckpt_path}")

        print(f"\n  Epoch {epoch+1} complete. Avg loss: {epoch_loss/steps:.4f}\n")

    # ── Save final checkpoint ─────────────────────────────────────────────
    final_path = Path(save_dir) / f"visual_encoder_stage{stage}_final.pt"
    torch.save(visual_encoder.state_dict(), final_path)
    print(f"\n  Final checkpoint saved: {final_path}")

    frame_buf.close()
    return visual_encoder


# ════════════════════════════════════════════════════════════════════════════
# 5.  SMOKE TEST  (no LLM checkpoint needed)
# ════════════════════════════════════════════════════════════════════════════

def smoke_test():
    """
    Validates the training loop shapes and gradient flow without a real
    LLM checkpoint. Uses random LLM weights.
    """
    print("=" * 60)
    print("  train_visual_encoder.py — smoke test (random LLM weights)")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n  Device: {device}")

    llm            = SpeakMK1LLM(LLM_CONFIG).to(device)
    visual_encoder = VisualEncoder(**VISUAL_CONFIG).to(device)

    # Stage 3 setup
    for p in llm.parameters():
        p.requires_grad = False
    visual_encoder.freeze_embedder()

    optimizer = AdamW(
        filter(lambda p: p.requires_grad, visual_encoder.parameters()),
        lr=1e-4,
    )

    B, L   = 2, 32
    frames = torch.randn(1, 32, TARGET_DIM, device=device)
    ids    = torch.randint(0, LLM_CONFIG.vocab_size, (B, L), device=device)
    labels = ids.clone()

    print("\n  [1] Forward pass ...")
    loss, aux = forward_with_visual(llm, visual_encoder, ids, labels, frames)
    print(f"      CE loss : {loss.item():.4f}")
    print(f"      aux     : {aux.item():.6f}")

    print("\n  [2] Backward pass ...")
    total = loss + LLM_CONFIG.aux_loss_weight * aux
    total.backward()

    pooler_grads = [
        p.grad for p in visual_encoder.pooler.parameters()
        if p.requires_grad and p.grad is not None
    ]
    embedder_grads = [
        p.grad for p in visual_encoder.embedder.parameters()
        if p.grad is not None
    ]
    llm_grads = [
        p.grad for p in llm.parameters()
        if p.grad is not None
    ]

    print(f"      Pooler params with grad    : {len(pooler_grads)} ✓")
    print(f"      Embedder params with grad  : {len(embedder_grads)}  (should be 0 — frozen)")
    print(f"      LLM params with grad       : {len(llm_grads)}  (should be 0 — frozen)")

    assert len(embedder_grads) == 0, "Embedder should be frozen!"
    assert len(llm_grads) == 0,      "LLM should be frozen!"
    assert len(pooler_grads) > 0,    "Pooler should have gradients!"
    print("\n  All assertions passed ✓")

    optimizer.step()
    optimizer.zero_grad()
    print("  Optimizer step OK ✓")

    print("\nDone ✓")


# ════════════════════════════════════════════════════════════════════════════
# 6.  ENTRY POINT
# ════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage",      type=int,  default=0,
                        help="Training stage: 3 or 4. 0 = smoke test.")
    parser.add_argument("--llm_ckpt",   type=str,  default="",
                        help="Path to LLM checkpoint (.pt)")
    parser.add_argument("--save_dir",   type=str,  default="checkpoints/visual")
    parser.add_argument("--batch_size", type=int,  default=4)
    parser.add_argument("--seq_len",    type=int,  default=128)
    parser.add_argument("--steps",      type=int,  default=1000,
                        help="Steps per epoch (replace with len(dataloader))")
    parser.add_argument("--no_webcam",  action="store_true",
                        help="Use synthetic frames instead of webcam")
    parser.add_argument("--camera",     type=int,  default=0)
    args = parser.parse_args()

    if args.stage == 0:
        smoke_test()
    elif args.stage in (3, 4):
        if not args.llm_ckpt:
            print("ERROR: --llm_ckpt is required for stage 3 or 4.")
            print("Run with --stage 0 for smoke test without a checkpoint.")
        else:
            train(
                stage      = args.stage,
                llm_ckpt   = args.llm_ckpt,
                save_dir   = args.save_dir,
                batch_size = args.batch_size,
                seq_len    = args.seq_len,
                use_webcam = not args.no_webcam,
                camera_idx = args.camera,
                steps      = args.steps,
            )
    else:
        print("ERROR: --stage must be 3 or 4 (or 0 for smoke test).")