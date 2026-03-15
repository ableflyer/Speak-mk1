import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from UniMamba import RMSNorm


class LocalWindowAttention(nn.Module):
    """
    Local Window Self-Attention.

    Each token at position t attends only to positions [t-window, t+window],
    giving O(L * window) complexity instead of O(L^2).

    The mask is causal-optional:
      - causal=False (default for encoder): symmetric window, full [t-w, t+w]
      - causal=True  (for decoder use):     left-only window, [t-w, t]

    Args:
        d_model:     model hidden dimension
        nheads:      number of attention heads
        window:      one-sided window size — token t sees [t-window, t+window]
        dropout:     attention dropout probability
        causal:      if True, only attend to past positions (left window only)
        eps:         RMSNorm epsilon
    """

    def __init__(
        self,
        d_model: int,
        nheads: int,
        window: int = 32,
        dropout: float = 0.0,
        causal: bool = False,
        eps: float = 1e-6,
    ):
        super().__init__()
        assert d_model % nheads == 0, "d_model must be divisible by nheads"

        self.d_model  = d_model
        self.nheads   = nheads
        self.headdim  = d_model // nheads
        self.window   = window
        self.causal   = causal
        self.scale    = self.headdim ** -0.5

        self.norm = RMSNorm(d_model, eps=eps)

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

        self.attn_drop = nn.Dropout(dropout)

    # ------------------------------------------------------------------
    def _build_window_mask(self, L: int, device: torch.device) -> torch.Tensor:
        """
        Build a boolean mask of shape (L, L) where True = MASKED (blocked).

        For each query position i, only keys j where |i - j| <= window are
        allowed. Everything outside the window is masked to -inf.

        If causal=True, additionally block j > i (future positions).
        """
        # positions i (rows) and j (cols)
        i = torch.arange(L, device=device).unsqueeze(1)  # (L, 1)
        j = torch.arange(L, device=device).unsqueeze(0)  # (1, L)

        # distance mask: block anything outside the window
        dist_mask = (i - j).abs() > self.window          # (L, L)  True = blocked

        if self.causal:
            # additionally block future positions
            causal_mask = j > i                           # (L, L)  True = blocked
            return dist_mask | causal_mask

        return dist_mask                                  # (L, L)

    # ------------------------------------------------------------------
    def forward(
        self,
        x: torch.Tensor,                              # (B, L, d_model)
        key_padding_mask: Optional[torch.Tensor] = None,  # (B, L) True = padded
    ) -> torch.Tensor:                                # (B, L, d_model)
        B, L, D = x.shape
        residual = x

        xn = self.norm(x)

        # Project to Q, K, V
        Q = self.q_proj(xn)   # (B, L, d_model)
        K = self.k_proj(xn)
        V = self.v_proj(xn)

        # Split into heads: (B, nheads, L, headdim)
        def split_heads(t):
            return t.view(B, L, self.nheads, self.headdim).transpose(1, 2)

        Q, K, V = split_heads(Q), split_heads(K), split_heads(V)

        # Scaled dot-product scores: (B, nheads, L, L)
        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale

        # Apply local window mask
        window_mask = self._build_window_mask(L, x.device)  # (L, L)  True=blocked
        scores = scores.masked_fill(window_mask.unsqueeze(0).unsqueeze(0), float("-inf"))

        # Apply padding mask if provided
        if key_padding_mask is not None:
            # key_padding_mask: (B, L)  True = padded position
            # Expand to (B, 1, 1, L) so it broadcasts over heads and query positions
            scores = scores.masked_fill(
                key_padding_mask.unsqueeze(1).unsqueeze(2), float("-inf")
            )

        # Softmax — positions where ALL keys are masked will produce NaN
        # (can happen at boundaries if window > L). Clamp to 0 via nan_to_num.
        attn = torch.softmax(scores, dim=-1)
        attn = torch.nan_to_num(attn, nan=0.0)
        attn = self.attn_drop(attn)

        # Weighted sum of values: (B, nheads, L, headdim)
        out = torch.matmul(attn, V)

        # Merge heads: (B, L, d_model)
        out = out.transpose(1, 2).contiguous().view(B, L, D)
        out = self.out_proj(out)

        return residual + out