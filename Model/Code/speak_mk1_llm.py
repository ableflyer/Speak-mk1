"""
speakmk1_llm_block.py
=====================
SpeakMK1 LLM — Nemotron 3 Nano-style hybrid architecture using UniMamba (Mamba-3),
LatentMoE, and CrossModelSparseAttention.

Block pattern (matches the diagram):

    Outer block repeated x3:
    ┌─ inner repeated x3 ──────────────────┐
    │   UniMamba → LatentMoE               │
    └──────────────────────────────────────┘
        → UniMamba
        → CrossModelSparseAttention
        → LatentMoE

Full model:
    Embedding
    → [OuterBlock] × 3
    → RMSNorm
    → LM Head

Notes:
    - UniMamba, LatentMoE, CrossModelSparseAttention all handle their own
      pre-norm and residual internally. This module adds NO extra residuals
      around them.
    - CrossModelSparseAttention uses a tanh gate initialised to 0, meaning
      audio has zero influence at init and gradually opens during training.
      During text-only pretraining, pass audio_out=None and the cross-
      attention layer is skipped entirely — no wasted compute.
    - aux_loss from all LatentMoE routers is accumulated and returned from
      every forward call so the training loop can add it to the task loss.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn

from UniMamba import UniMamba, RMSNorm
from LatentMoE import LatentMoE
from cross_model_sparse_attn import CrossModelSparseAttention


# ════════════════════════════════════════════════════════════════════════════
# 1.  INNER UNIT  —  UniMamba + LatentMoE
#     The cheap repeating pair shown inside the small inner box in the diagram.
#     UniMamba does sequence mixing; LatentMoE does token-wise sparse FFN.
# ════════════════════════════════════════════════════════════════════════════

class UniMambaMoEUnit(nn.Module):
    """
    Single inner unit:

        x  →  UniMamba  →  LatentMoE  →  out

    Both sub-layers carry their own pre-norm and residual connection, so
    this wrapper is purely sequential — no extra norms or residuals here.

    Args:
        d_model:     hidden dimension
        d_state:     SSM state size passed to UniMamba
        d_conv:      depthwise conv width passed to UniMamba
        expand:      expansion ratio passed to UniMamba
        nheads:      number of heads inside UniMamba
        latent_dim:  LatentMoE bottleneck dim (None → d_model // 4 inside LatentMoE)
        num_experts: number of routed experts in LatentMoE
        top_k:       experts activated per token
        dropout:     dropout applied inside LatentMoE
    """

    def __init__(
        self,
        d_model:     int,
        d_state:     int = 64,
        d_conv:      int = 4,
        expand:      int = 2,
        nheads:      int = 8,
        latent_dim:  Optional[int] = None,
        num_experts: int = 8,
        top_k:       int = 2,
        dropout:     float = 0.0,
    ):
        super().__init__()

        self.ssm = UniMamba(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            nheads=nheads,
        )

        self.moe = LatentMoE(
            d_model=d_model,
            latent_dim=latent_dim,
            num_experts=num_experts,
            top_k=top_k,
            dropout=dropout,
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (B, L, D)
        Returns:
            out:      (B, L, D)
            aux_loss: scalar — router load-balancing loss from LatentMoE
        """
        x = self.ssm(x)
        x = self.moe(x)
        return x, self.moe.last_aux_loss


# ════════════════════════════════════════════════════════════════════════════
# 2.  OUTER BLOCK
#     [UniMamba + LatentMoE] × 3  →  UniMamba  →  CrossAttn  →  LatentMoE
# ════════════════════════════════════════════════════════════════════════════

class SpeakMK1OuterBlock(nn.Module):
    """
    One complete outer block as shown in the diagram:

        ┌── inner × num_inner_repeats ──────────────┐
        │   UniMamba → LatentMoE  (repeated)         │
        └────────────────────────────────────────────┘
            → UniMamba                    (pre-attention sequence mixing)
            → CrossModelSparseAttention   (audio grounding — skipped if audio_out=None)
            → LatentMoE                   (post-attention FFN)

    Audio grounding is optional at runtime:
        audio_out=None  → CrossModelSparseAttention is skipped entirely.
                          Use during pure text pretraining (Stages 1 and 2).
        audio_out given → sparse cross-attention runs normally.
                          Use during multimodal fine-tuning (Stage 3+).

    The tanh gate inside CrossModelSparseAttention ensures audio contributes
    zero at initialisation even when it is running, so it is safe to enable
    audio_out as soon as the audio encoder is ready — no LLM destabilisation.

    Args:
        d_model:            hidden dimension
        d_state:            SSM state size
        d_conv:             depthwise conv width for UniMamba
        expand:             expansion ratio for UniMamba
        nheads_ssm:         heads inside UniMamba
        nheads_attn:        heads for CrossModelSparseAttention
        top_k_audio:        audio frames each text token attends to (sparse k)
        latent_dim:         LatentMoE bottleneck (None → d_model // 4)
        num_experts:        routed experts per LatentMoE
        top_k_experts:      experts activated per token
        num_inner_repeats:  how many UniMamba+MoE pairs before the attention layer
        dropout:            dropout throughout
    """

    def __init__(
        self,
        d_model:           int,
        d_state:           int = 64,
        d_conv:            int = 4,
        expand:            int = 2,
        nheads_ssm:        int = 8,
        nheads_attn:       int = 8,
        top_k_audio:       int = 32,
        latent_dim:        Optional[int] = None,
        num_experts:       int = 8,
        top_k_experts:     int = 2,
        num_inner_repeats: int = 3,
        dropout:           float = 0.0,
    ):
        super().__init__()

        # ── [UniMamba → LatentMoE] × num_inner_repeats ────────────────────
        self.inner_units = nn.ModuleList([
            UniMambaMoEUnit(
                d_model=d_model,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
                nheads=nheads_ssm,
                latent_dim=latent_dim,
                num_experts=num_experts,
                top_k=top_k_experts,
                dropout=dropout,
            )
            for _ in range(num_inner_repeats)
        ])

        # ── UniMamba before cross-attention ────────────────────────────────
        self.pre_attn_ssm = UniMamba(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            nheads=nheads_ssm,
        )

        # ── CrossModelSparseAttention (audio grounding) ────────────────────
        self.cross_attn = CrossModelSparseAttention(
            d_model=d_model,
            nheads=nheads_attn,
            top_k=top_k_audio,
            dropout=dropout,
            use_gate=True,
        )

        # ── LatentMoE after cross-attention (post-attention FFN) ───────────
        self.post_attn_moe = LatentMoE(
            d_model=d_model,
            latent_dim=latent_dim,
            num_experts=num_experts,
            top_k=top_k_experts,
            dropout=dropout,
        )

    def forward(
        self,
        x:                  torch.Tensor,
        audio_out:          Optional[torch.Tensor] = None,
        audio_padding_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x:                  (B, L, D) text token hidden states
            audio_out:          (B, T_audio, D) or None
            audio_padding_mask: (B, T_audio) True=padded, or None

        Returns:
            out:      (B, L, D)
            aux_loss: scalar — sum of all LatentMoE router losses in this block
        """
        total_aux = x.new_zeros(1).squeeze()

        # ── [UniMamba → LatentMoE] × num_inner_repeats ────────────────────
        for unit in self.inner_units:
            x, aux = unit(x)
            total_aux = total_aux + aux
        total_aux = total_aux / len(self.inner_units)

        # ── UniMamba (pre-attention) ───────────────────────────────────────
        x = self.pre_attn_ssm(x)

        # ── CrossModelSparseAttention ─────────────────────────────────────
        # Skipped during text-only pretraining when audio_out is None.
        if audio_out is not None:
            x = self.cross_attn(
                text_hidden=x,
                audio_out=audio_out,
                audio_padding_mask=audio_padding_mask,
            )

        # ── LatentMoE (post-attention FFN) ────────────────────────────────
        x = self.post_attn_moe(x)
        total_aux = (total_aux + self.post_attn_moe.last_aux_loss) / 2

        return x, total_aux


# ════════════════════════════════════════════════════════════════════════════
# 3.  CONFIG
# ════════════════════════════════════════════════════════════════════════════

@dataclass
class SpeakMK1LLMConfig:
    # ── Vocabulary ────────────────────────────────────────────────────────
    vocab_size:        int   = 50277    # GPT-NeoX tokenizer default; change to match yours

    # ── Model dimensions ──────────────────────────────────────────────────
    d_model:           int   = 1024
    d_state:           int   = 64
    d_conv:            int   = 4
    expand:            int   = 2

    # ── Block structure (matches diagram exactly) ─────────────────────────
    num_outer_blocks:  int   = 3        # outer box ×3
    num_inner_repeats: int   = 3        # inner [UniMamba+MoE] ×3

    # ── Attention ─────────────────────────────────────────────────────────
    nheads_ssm:        int   = 8
    nheads_attn:       int   = 8
    top_k_audio:       int   = 32

    # ── MoE ───────────────────────────────────────────────────────────────
    latent_dim:        Optional[int] = None    # None → d_model // 4 inside LatentMoE
    num_experts:       int   = 8
    top_k_experts:     int   = 2

    # ── Regularisation ────────────────────────────────────────────────────
    dropout:           float = 0.0

    # ── Training ──────────────────────────────────────────────────────────
    aux_loss_weight:   float = 1e-2    # multiply total_aux by this before adding to loss


# ════════════════════════════════════════════════════════════════════════════
# 4.  FULL LLM
# ════════════════════════════════════════════════════════════════════════════

class SpeakMK1LLM(nn.Module):
    """
    Full SpeakMK1 LLM.

    Text-only forward (pretraining Stages 1 and 2 — no audio):
        logits, loss, aux_loss = model(input_ids, labels=labels)

    Multimodal forward (Stage 3+ — paired with frozen audio encoder):
        logits, loss, aux_loss = model(
            input_ids,
            labels=labels,
            audio_out=encoder_tokens,           # (B, num_queries, d_model)
            audio_padding_mask=padding_mask,    # (B, num_queries) True=padded or None
        )

    Training loss:
        total_loss = loss + config.aux_loss_weight * aux_loss

    The aux_loss is the sum of all LatentMoE load-balancing losses across
    all blocks. It keeps experts from collapsing — always include it.
    """

    def __init__(self, config: SpeakMK1LLMConfig):
        super().__init__()
        self.config = config
        D = config.d_model

        # ── Token embedding ───────────────────────────────────────────────
        embed_dim = 128
        self.embedding = nn.Embedding(config.vocab_size, embed_dim)
        self.embed_proj = nn.Linear(embed_dim, D, bias=False)
        # ── [OuterBlock] × num_outer_blocks ──────────────────────────────
        self.blocks = nn.ModuleList([
            SpeakMK1OuterBlock(
                d_model=D,
                d_state=config.d_state,
                d_conv=config.d_conv,
                expand=config.expand,
                nheads_ssm=config.nheads_ssm,
                nheads_attn=config.nheads_attn,
                top_k_audio=config.top_k_audio,
                latent_dim=config.latent_dim,
                num_experts=config.num_experts,
                top_k_experts=config.top_k_experts,
                num_inner_repeats=config.num_inner_repeats,
                dropout=config.dropout,
            )
            for _ in range(config.num_outer_blocks)
        ])

        # ── Final norm + LM head ──────────────────────────────────────────
        self.norm = RMSNorm(D)
        self.lm_head = nn.Linear(D, config.vocab_size, bias=False)

        # Weight tying: LM head shares weights with embedding.
        # Saves ~50M params at d_model=1024, vocab=50k. Standard practice.
        # self.lm_head.weight = self.embedding.weight

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.embedding.weight, std=0.02)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(
        self,
        input_ids:          torch.Tensor,
        labels:             Optional[torch.Tensor] = None,
        audio_out:          Optional[torch.Tensor] = None,
        audio_padding_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor]:
        """
        Args:
            input_ids:          (B, L)
            labels:             (B, L) with -100 at ignored positions, or None
            audio_out:          (B, T_audio, D) or None
            audio_padding_mask: (B, T_audio) True=padded, or None

        Returns:
            logits:    (B, L, vocab_size)
            loss:      scalar CE loss if labels provided, else None
            aux_loss:  scalar — sum of all router load-balancing losses
        """
        x = self.embed_proj(self.embedding(input_ids))  # (B, L, D)
        total_aux = x.new_zeros(1).squeeze()

        for block in self.blocks:
            x, aux = block(
                x,
                audio_out=audio_out,
                audio_padding_mask=audio_padding_mask,
            )
            total_aux = total_aux + aux
        total_aux = total_aux / len(self.blocks)

        x = self.norm(x)
        logits = self.lm_head(x)                # (B, L, vocab_size)

        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = nn.CrossEntropyLoss(ignore_index=-100)(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
            )

        return logits, loss, total_aux

    @torch.no_grad()
    def generate(
        self,
        input_ids:          torch.Tensor,
        max_new_tokens:     int = 128,
        temperature:        float = 1.0,
        top_p:              float = 0.9,
        audio_out:          Optional[torch.Tensor] = None,
        audio_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Autoregressive generation with top-p (nucleus) sampling.

        Args:
            input_ids:      (B, L) prompt tokens
            max_new_tokens: tokens to generate
            temperature:    softmax temperature (lower = more deterministic)
            top_p:          nucleus sampling threshold
            audio_out:      optional audio context (B, T_audio, D)
            audio_padding_mask: (B, T_audio) True=padded

        Returns:
            (B, L + max_new_tokens) — prompt + generated tokens
        """
        self.eval()
        generated = input_ids

        for _ in range(max_new_tokens):
            logits, _, _ = self.forward(
                generated,
                audio_out=audio_out,
                audio_padding_mask=audio_padding_mask,
            )
            next_logits = logits[:, -1, :] / temperature

            # Top-p nucleus filtering
            sorted_logits, sorted_indices = torch.sort(next_logits, descending=True)
            cumprobs = sorted_logits.softmax(dim=-1).cumsum(dim=-1)
            remove = cumprobs - sorted_logits.softmax(dim=-1) > top_p
            sorted_logits = sorted_logits.masked_fill(remove, float("-inf"))
            next_logits = torch.zeros_like(next_logits).scatter(1, sorted_indices, sorted_logits)

            probs = next_logits.softmax(dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            generated = torch.cat([generated, next_token], dim=1)

        return generated


# ════════════════════════════════════════════════════════════════════════════
# 5.  UTILITIES
# ════════════════════════════════════════════════════════════════════════════

def count_parameters(model: nn.Module) -> dict:
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        "total_M":     round(total / 1e6, 1),
        "trainable_M": round(trainable / 1e6, 1),
    }


# ════════════════════════════════════════════════════════════════════════════
# 6.  SMOKE TEST
# ════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("Building SpeakMK1LLM …")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device}")

    cfg = SpeakMK1LLMConfig(
        vocab_size=50277,
        d_model=1024,
        d_state=64,
        num_outer_blocks=3,
        num_inner_repeats=3,
        nheads_ssm=8,
        nheads_attn=8,
        top_k_audio=32,
        num_experts=8,
        top_k_experts=2,
    )

    model = SpeakMK1LLM(cfg).to(device)
    params = count_parameters(model)
    print(f"  Total parameters  : {params['total_M']}M")
    print(f"  Trainable params  : {params['trainable_M']}M")
    print(f"  d_model           : {cfg.d_model}")
    print(f"  Outer blocks ×{cfg.num_outer_blocks}   inner units ×{cfg.num_inner_repeats}")
    print(f"  Experts           : {cfg.num_experts} routed, top-{cfg.top_k_experts}")

    B, L = 2, 128
    input_ids = torch.randint(0, cfg.vocab_size, (B, L), device=device)
    labels    = torch.randint(0, cfg.vocab_size, (B, L), device=device)

    # ── 1. Text-only (pretraining) ────────────────────────────────────────
    print("\n[1] Text-only forward (audio_out=None) …")
    logits, loss, aux = model(input_ids, labels=labels)
    print(f"  logits : {logits.shape}")
    print(f"  CE loss: {loss.item():.4f}")
    print(f"  aux    : {aux.item():.6f}")
    assert logits.shape == (B, L, cfg.vocab_size), "shape mismatch"

    # ── 2. Multimodal (with audio encoder Q-Former output) ────────────────
    print("\n[2] Multimodal forward (audio_out provided) …")
    audio_tokens = torch.randn(B, 64, cfg.d_model, device=device)
    logits_mm, loss_mm, aux_mm = model(
        input_ids, labels=labels, audio_out=audio_tokens
    )
    print(f"  logits : {logits_mm.shape}")
    print(f"  CE loss: {loss_mm.item():.4f}")
    print(f"  aux    : {aux_mm.item():.6f}")

    # ── 3. Training loss combination ──────────────────────────────────────
    print("\n[3] Combined training loss …")
    total_loss = loss_mm + cfg.aux_loss_weight * aux_mm
    print(f"  total_loss = CE + {cfg.aux_loss_weight} * aux = {total_loss.item():.4f}")

    # ── 4. Generation ─────────────────────────────────────────────────────
    print("\n[4] Generation …")
    prompt = torch.randint(0, cfg.vocab_size, (1, 8), device=device)
    out = model.generate(prompt, max_new_tokens=16, audio_out=audio_tokens[:1])
    print(f"  output shape: {out.shape}")
    assert out.shape == (1, 24), "generation shape mismatch"

    # ── 5. Single training step sanity check ──────────────────────────────
    print("\n[5] Single training step sanity check …")
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    model.train()
    logits, loss, aux = model(input_ids, labels=labels)
    total = loss + cfg.aux_loss_weight * aux
    total.backward()
    opt.step()
    opt.zero_grad()
    logits2, loss2, aux2 = model(input_ids, labels=labels)
    assert not torch.isnan(loss2), "NaN loss after one step"
    print(f"  loss before: {loss.item():.4f}  after: {loss2.item():.4f}")
    print("  Backward pass OK ✓")

    print("\nDone ✓")