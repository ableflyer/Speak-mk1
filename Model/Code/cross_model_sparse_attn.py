# cross_model_sparse_attn.py
#
# Cross-Model Sparse Attention (decoder side of diagram).
#
# What it does:
#   Text decoder tokens (queries) attend to audio encoder output (keys/values).
#   Instead of attending to ALL audio frames (expensive, noisy), each text token
#   selects only the top-k most relevant audio frames via a lightweight relevance
#   score, then runs attention only over those k frames.
#
# Why sparse:
#   A typical utterance at 10ms frame shift produces 300-500 audio frames.
#   Full cross-attention from every text token to every audio frame is O(T_text * T_audio).
#   With top-k selection we reduce this to O(T_text * k) where k << T_audio.
#
# Flow:
#   text_hidden (B, T_text, d)   ← queries  (from decoder layer)
#   audio_out   (B, T_audio, d)  ← keys & values  (from audio encoder)
#
#   1. Score: relevance[b, t, a] = dot(Q[b,t], K[b,a]) / sqrt(d_head)
#   2. Select: for each text token t, pick top-k audio indices by score
#   3. Attend: softmax over only those k scores, weighted sum of k values
#   4. Gated residual: tanh gate (Flamingo-style) controls injection strength
#
# The tanh gate initialised to 0 means the layer starts as an identity
# (audio has no influence at init) and gradually opens during training.
# This is critical for not destabilising a pretrained LLM backbone.

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from UniMamba import RMSNorm


class CrossModelSparseAttention(nn.Module):
    """
    Cross-Model Sparse Attention.

    Decoder text tokens attend to audio encoder output, but only to their
    top-k most relevant audio frames (sparse selection).

    Args:
        d_model:      model hidden dimension (must match both encoder and decoder)
        nheads:       number of attention heads
        top_k:        number of audio frames each text token attends to
        dropout:      attention dropout
        use_gate:     if True, wrap output with a learned tanh gate (Flamingo-style)
                      so the layer starts as identity and opens gradually during training.
                      Strongly recommended when inserting into a pretrained LLM.
        eps:          RMSNorm epsilon
    """

    def __init__(
        self,
        d_model: int,
        nheads: int,
        top_k: int = 32,
        dropout: float = 0.0,
        use_gate: bool = True,
        eps: float = 1e-6,
    ):
        super().__init__()
        assert d_model % nheads == 0, "d_model must be divisible by nheads"

        self.d_model  = d_model
        self.nheads   = nheads
        self.headdim  = d_model // nheads
        self.top_k    = top_k
        self.scale    = self.headdim ** -0.5
        self.use_gate = use_gate

        # Pre-norms — one for each input stream (standard practice in cross-attn)
        self.norm_text  = RMSNorm(d_model, eps=eps)   # normalise query side
        self.norm_audio = RMSNorm(d_model, eps=eps)   # normalise key/value side

        # Projections
        # Q comes from text decoder hidden states
        # K, V come from audio encoder output
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

        # Flamingo-style learned tanh gate
        # Initialised to 0 → tanh(0) = 0 → no audio influence at init
        # Gradually opens as training progresses
        if use_gate:
            self.gate = nn.Parameter(torch.tensor(0.1))

        self.attn_drop = nn.Dropout(dropout)

    # ------------------------------------------------------------------
    def _split_heads(self, t: torch.Tensor, B: int, L: int) -> torch.Tensor:
        """(B, L, d_model) → (B, nheads, L, headdim)"""
        return t.view(B, L, self.nheads, self.headdim).transpose(1, 2)

    # ------------------------------------------------------------------
    def forward(
        self,
        text_hidden: torch.Tensor,                      # (B, T_text,  d_model)
        audio_out:   torch.Tensor,                      # (B, T_audio, d_model)
        audio_padding_mask: Optional[torch.Tensor] = None,  # (B, T_audio) True=padded
    ) -> torch.Tensor:                                  # (B, T_text, d_model)
        """
        Args:
            text_hidden:        decoder hidden states — these become the queries
            audio_out:          audio encoder output — these become keys and values
            audio_padding_mask: True at padded audio positions (variable-length audio)

        Returns:
            text_hidden + gated cross-attention output  (B, T_text, d_model)
        """
        B, T_text,  D = text_hidden.shape
        _,  T_audio, _ = audio_out.shape

        # Clamp top_k to available audio frames — handles short utterances
        k = min(self.top_k, T_audio)

        # Pre-norm both streams independently
        q_in = self.norm_text(text_hidden)    # (B, T_text,  D)
        kv_in = self.norm_audio(audio_out)    # (B, T_audio, D)

        # Project
        Q = self.q_proj(q_in)    # (B, T_text,  D)
        K = self.k_proj(kv_in)   # (B, T_audio, D)
        V = self.v_proj(kv_in)   # (B, T_audio, D)

        # Split into heads
        Q = self._split_heads(Q, B, T_text)    # (B, H, T_text,  headdim)
        K = self._split_heads(K, B, T_audio)   # (B, H, T_audio, headdim)
        V = self._split_heads(V, B, T_audio)   # (B, H, T_audio, headdim)

        # Full cross-attention scores: (B, H, T_text, T_audio)
        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale

        # Mask padded audio frames before top-k selection
        if audio_padding_mask is not None:
            # (B, T_audio) → (B, 1, 1, T_audio) — broadcast over heads and text positions
            scores = scores.masked_fill(
                audio_padding_mask.unsqueeze(1).unsqueeze(2), float("-inf")
            )

        # -----------------------------------------------------------------
        # Sparse selection: top-k audio frames per text token per head
        # -----------------------------------------------------------------
        # topk_scores:   (B, H, T_text, k)  — scores of top-k audio frames
        # topk_indices:  (B, H, T_text, k)  — which audio frame indices they are
        topk_scores, topk_indices = scores.topk(k, dim=-1)

        # Softmax over only the k selected scores
        # nan_to_num handles the all-padding edge case:
        # if every score is -inf (all audio frames padded), softmax gives NaN.
        # We replace NaN with 0.0 so the weighted sum produces a zero vector.
        attn_weights = torch.softmax(topk_scores, dim=-1)  # (B, H, T_text, k)
        attn_weights = self.attn_drop(attn_weights)
        attn_weights = torch.nan_to_num(attn_weights, nan=0.0)

        # Gather the corresponding values
        # V:            (B, H, T_audio, headdim)
        # topk_indices: (B, H, T_text, k)
        # We need:      (B, H, T_text, k, headdim)
        # topk_indices_exp = topk_indices.unsqueeze(-1).expand(
        #     B, self.nheads, T_text, k, self.headdim
        # )
        # V_exp = V.unsqueeze(2).expand(
        #     B, self.nheads, T_text, T_audio, self.headdim
        # ).contiguous()
        # V_selected = V_exp.gather(
        #     dim=3, index=topk_indices_exp
        # )  # (B, H, T_text, k, headdim)

        # # Weighted sum over k selected frames
        # # attn_weights: (B, H, T_text, k) → (B, H, T_text, k, 1)
        # out = (attn_weights.unsqueeze(-1) * V_selected).sum(dim=3)  # (B, H, T_text, headdim)
        
        # Reshape indices: (B, H, T_text*k)
        idx_flat = topk_indices.reshape(B, self.nheads, T_text * k)

        # Expand headdim dim: (B, H, T_text*k, headdim)
        idx_flat_exp = idx_flat.unsqueeze(-1).expand(-1, -1, -1, self.headdim)

        # Gather from V along dim=2: (B, H, T_text*k, headdim)
        V_gathered = V.gather(dim=2, index=idx_flat_exp)

        # Reshape back: (B, H, T_text, k, headdim)
        V_selected = V_gathered.reshape(B, self.nheads, T_text, k, self.headdim)

        # Weighted sum
        out = (attn_weights.unsqueeze(-1) * V_selected).sum(dim=3)

        # Merge heads: (B, T_text, D)
        out = out.transpose(1, 2).contiguous().view(B, T_text, D)
        out = self.out_proj(out)

        # Tanh gate — starts at 0 (no audio influence), opens during training
        if self.use_gate:
            out = torch.tanh(self.gate) * out

        return text_hidden + out