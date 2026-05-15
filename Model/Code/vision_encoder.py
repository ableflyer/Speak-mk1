"""
visual_encoder.py
this is not used
=================
SpeakMK1 — Visual encoder for mouth tracking features.

Architecture (simplified — no TemporalEncoder):

    MouthTracker (per frame)
        └─ MouthLandmarkResult.to_tensor()  →  (T, 180)
               │
    VisualFrameEmbedder                     →  (B, T, d_model)
        │  Linear(360 → d_model)  [frame + delta concatenated]
        │  + sinusoidal position encoding
        │
    VisualPooler                            →  (B, n_visual_tokens, d_model)
        │  n learned queries cross-attend over T frame embeddings
        │  + modality token offset
        │
    VisualEncoder.forward()                 →  (B, n_visual_tokens, d_model)

The TemporalEncoder (LocalWindowAttention stack) is intentionally omitted.
The VisualPooler's cross-attention already provides inter-frame context when
the learned queries attend over the full sequence. MediaPipe has already done
the geometric understanding — we just need to project and pool.

Usage
-----
    from vision_part import MouthTracker, RunningMode
    from visual_encoder import VisualEncoder

    tracker = MouthTracker(model_path="face_landmarker.task",
                           running_mode=RunningMode.IMAGE)
    encoder = VisualEncoder(d_model=512, n_visual_tokens=8).cuda()

    # offline batch
    frames = [tracker.process_frame(f) for f in video_frames]
    x      = MouthTracker.batch_to_tensor(frames)    # (T, 180)
    tokens = encoder(x.unsqueeze(0).cuda())          # (1, 8, 512)

    # streaming rolling buffer
    buf = encoder.make_buffer(window=32)
    for frame_bgr in camera_stream:
        result = tracker.process_frame(frame_bgr)
        tokens = encoder.step(buf, result)           # (1, 8, 512) or None
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np
import torch
import torch.nn as nn

from vision_part import MouthLandmarkResult, MouthTracker, TARGET_DIM


# ════════════════════════════════════════════════════════════════════════════
# 0.  HELPERS
# ════════════════════════════════════════════════════════════════════════════

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps   = eps
        self.scale = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = x.pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return x * rms * self.scale


def sinusoidal_pos_enc(seq_len: int, d_model: int, device: torch.device) -> torch.Tensor:
    """Returns (1, seq_len, d_model) sinusoidal position encodings."""
    pos = torch.arange(seq_len, device=device).unsqueeze(1).float()
    dim = torch.arange(0, d_model, 2, device=device).float()
    div = torch.exp(-dim * math.log(10000.0) / d_model)
    enc = torch.zeros(seq_len, d_model, device=device)
    enc[:, 0::2] = torch.sin(pos * div)
    enc[:, 1::2] = torch.cos(pos * div)
    return enc.unsqueeze(0)


# ════════════════════════════════════════════════════════════════════════════
# 1.  FRAME EMBEDDER  —  (B, T, 180) → (B, T, d_model)
# ════════════════════════════════════════════════════════════════════════════

class VisualFrameEmbedder(nn.Module):
    """
    Per-frame projection + position encoding.

    Concatenates each frame with its delta (frame[t] - frame[t-1]) before
    projecting, giving the pooler explicit access to articulatory velocity.
    Delta of the first frame is zero-padded.

    input_dim * 2 → d_model  (360 → 512 by default)
    """

    def __init__(self, input_dim: int = TARGET_DIM, d_model: int = 512):
        super().__init__()
        self.d_model = d_model
        self.proj = nn.Sequential(
            nn.Linear(input_dim * 2, d_model * 2),
            nn.GELU(),
            nn.Linear(d_model * 2, d_model),
        )
        self.norm = RMSNorm(d_model)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x : (B, T, input_dim)
        returns : (B, T, d_model)
        """
        delta = torch.zeros_like(x)
        delta[:, 1:] = x[:, 1:] - x[:, :-1]

        x_aug = torch.cat([x, delta], dim=-1)   # (B, T, input_dim*2)
        emb   = self.proj(x_aug)                # (B, T, d_model)
        emb   = self.norm(emb)

        pos = sinusoidal_pos_enc(x.size(1), self.d_model, x.device)
        return emb + pos                        # (B, T, d_model)


# ════════════════════════════════════════════════════════════════════════════
# 2.  VISUAL POOLER  —  (B, T, d_model) → (B, n_visual_tokens, d_model)
# ════════════════════════════════════════════════════════════════════════════

class VisualPooler(nn.Module):
    """
    Learned cross-attention pooling from T frames → n_visual_tokens.

    n_visual_tokens learned query vectors attend over the T frame embeddings,
    compressing a variable-length frame window into a fixed token count for
    the LLM regardless of how many frames were collected.

    The cross-attention provides inter-frame context — each query attends
    to any frame in the window, learning to pick out phonemically relevant
    moments rather than averaging everything uniformly.

    A modality token offset shifts the output into a distinct region of
    embedding space so the LLM can distinguish visual from audio tokens.
    """

    def __init__(self, d_model: int = 512, n_visual_tokens: int = 8,
                 n_heads: int = 8):
        super().__init__()
        self.n_visual_tokens = n_visual_tokens

        self.queries = nn.Parameter(
            torch.randn(1, n_visual_tokens, d_model) * 0.02
        )
        self.cross_attn = nn.MultiheadAttention(
            embed_dim   = d_model,
            num_heads   = n_heads,
            batch_first = True,
        )
        self.norm_q   = RMSNorm(d_model)
        self.norm_kv  = RMSNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Linear(d_model * 2, d_model),
        )
        self.norm_out = RMSNorm(d_model)

        # Shifts visual tokens into distinct embedding region from audio tokens
        self.modality_token = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.trunc_normal_(self.modality_token, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x : (B, T, d_model)
        returns : (B, n_visual_tokens, d_model)
        """
        B  = x.size(0)
        q  = self.queries.expand(B, -1, -1)
        kv = self.norm_kv(x)
        q  = self.norm_q(q)

        pooled, _ = self.cross_attn(q, kv, kv, need_weights=False)
        pooled    = self.norm_out(q + pooled + self.ffn(pooled))
        return pooled + self.modality_token


# ════════════════════════════════════════════════════════════════════════════
# 3.  VISUAL ENCODER  —  full pipeline
# ════════════════════════════════════════════════════════════════════════════

class VisualEncoder(nn.Module):
    """
    Full visual encoding pipeline for SpeakMK1.

    Input  : (B, T, 180)  — MouthTracker frame vectors
    Output : (B, n_visual_tokens, d_model)  — visual tokens for the LLM

    Parameters
    ----------
    d_model         : must match LLM d_model (default 512)
    n_visual_tokens : output token count (default 8)
    n_heads         : attention heads in VisualPooler (default 8)
    input_dim       : frame feature dim from MouthTracker (default 180)

    Training strategy
    -----------------
    Stage 3 — call freeze_embedder() first:
        only pooler queries + cross-attn update
        LLM backbone is also frozen, only pooler trains
    Stage 4 — call unfreeze_all():
        everything trains jointly with the LLM
    """

    def __init__(
        self,
        d_model         : int = 512,
        n_visual_tokens : int = 8,
        n_heads         : int = 8,
        input_dim       : int = TARGET_DIM,
    ):
        super().__init__()
        self.embedder = VisualFrameEmbedder(input_dim, d_model)
        self.pooler   = VisualPooler(d_model, n_visual_tokens, n_heads)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x : (B, T, input_dim)
        returns : (B, n_visual_tokens, d_model)
        """
        x = self.embedder(x)   # (B, T, d_model)
        x = self.pooler(x)     # (B, n_visual_tokens, d_model)
        return x

    def freeze_embedder(self):
        """Stage 3: freeze embedder, train pooler only."""
        for p in self.embedder.parameters():
            p.requires_grad = False
        for p in self.pooler.parameters():
            p.requires_grad = True
        print("[VisualEncoder] Embedder frozen — pooler only.")

    def unfreeze_all(self):
        """Stage 4: unfreeze everything."""
        for p in self.parameters():
            p.requires_grad = True
        print("[VisualEncoder] All parameters unfrozen.")

    def trainable_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def total_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def make_buffer(self, window: int = 32) -> dict:
        """Rolling frame buffer for streaming. window=32 → ~1s at 30fps."""
        return {"frames": [], "window": window}

    @torch.no_grad()
    def step(
        self,
        buf    : dict,
        result : MouthLandmarkResult,
        device : torch.device = torch.device("cpu"),
    ) -> Optional[torch.Tensor]:
        """
        Push one MouthLandmarkResult into the rolling buffer.
        Returns (1, n_visual_tokens, d_model) when full, else None.
        Slides by 50% hop after each emission.
        """
        buf["frames"].append(result.to_vector())
        window = buf["window"]
        if len(buf["frames"]) < window:
            return None
        vecs   = np.stack(buf["frames"][-window:], axis=0)
        x      = torch.from_numpy(vecs).unsqueeze(0).to(device)
        tokens = self.forward(x)
        buf["frames"] = buf["frames"][window // 2:]
        return tokens


# ════════════════════════════════════════════════════════════════════════════
# 4.  MULTIMODAL CONCAT HELPER
# ════════════════════════════════════════════════════════════════════════════

def concat_visual_audio(
    visual_tokens : torch.Tensor,
    audio_tokens  : torch.Tensor,
) -> torch.Tensor:
    """
    visual_tokens : (B, n_visual_tokens, d_model)
    audio_tokens  : (B, n_audio_tokens, d_model)
    returns       : (B, n_visual + n_audio, d_model)
    """
    return torch.cat([visual_tokens, audio_tokens], dim=1)


# ════════════════════════════════════════════════════════════════════════════
# 5.  SMOKE TEST
# ════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 60)
    print("  SpeakMK1 VisualEncoder — smoke test")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n  Device: {device}")

    encoder = VisualEncoder(d_model=512, n_visual_tokens=8).to(device)
    print(f"  Total params    : {encoder.total_params():,}")
    print(f"  Trainable params: {encoder.trainable_params():,}")

    print("\n  [1] Batch forward ...")
    B, T   = 2, 32
    x      = torch.randn(B, T, TARGET_DIM).to(device)
    tokens = encoder(x)
    assert tokens.shape == (B, 8, 512)
    print(f"      {tuple(x.shape)} → {tuple(tokens.shape)} ✓")

    print("\n  [2] Freeze embedder (Stage 3) ...")
    encoder.freeze_embedder()
    assert all(not p.requires_grad for p in encoder.embedder.parameters())
    assert all(p.requires_grad for p in encoder.pooler.parameters())
    print(f"      Trainable params: {encoder.trainable_params():,} ✓")

    print("\n  [3] Unfreeze all (Stage 4) ...")
    encoder.unfreeze_all()
    print(f"      Trainable params: {encoder.trainable_params():,} ✓")

    print("\n  [4] Streaming buffer ...")
    from vision_part import MouthTracker, RunningMode
    tracker   = MouthTracker(running_mode=RunningMode.IMAGE)
    buf       = encoder.make_buffer(window=32)
    fake      = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    emissions = 0
    for i in range(64):
        result = tracker.process_frame(fake)
        out    = encoder.step(buf, result, device=device)
        if out is not None:
            emissions += 1
            print(f"      frame {i:02d}: emitted {tuple(out.shape)}")
    tracker.close()
    print(f"      Total emissions: {emissions}")

    print("\n  [5] Gradient flow ...")
    encoder.train()
    x = torch.randn(1, 16, TARGET_DIM).to(device)
    encoder(x).mean().backward()
    grads_ok = all(
        p.grad is not None and not p.grad.isnan().any()
        for p in encoder.parameters() if p.requires_grad
    )
    print(f"      Gradients healthy: {grads_ok} ✓" if grads_ok else "      Gradient issue!")

    print("\n  [6] Multimodal concat ...")
    visual = torch.randn(2, 8,  512).to(device)
    audio  = torch.randn(2, 64, 512).to(device)
    joined = concat_visual_audio(visual, audio)
    assert joined.shape == (2, 72, 512)
    print(f"      {tuple(visual.shape)} + {tuple(audio.shape)} → {tuple(joined.shape)} ✓")

    print("\nDone ✓")