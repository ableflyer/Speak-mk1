"""
speakmk1_llm_block.py
=====================
SpeakMK1 LLM decoder.

Block pattern (x num_blocks, matches diagram):

    UniMamba
    -> CrossModelSparseAttention   (skipped if audio_out=None)
    -> UniMambaMoE

Full model:
    Embedding (128-dim bottleneck -> d_model)
    -> [DecoderBlock] x num_blocks
    -> RMSNorm
    -> LM Head

Notes:
    - All sub-layers handle their own pre-norm and residual internally.
    - CrossModelSparseAttention tanh gate is initialised to 0, so audio
      contributes zero at init. Pass audio_out=None during text-only
      pretraining and the cross-attn is skipped entirely (no wasted compute).
    - aux_loss from all LatentMoE routers is accumulated and returned so
      the training loop can add it to the task loss.
    - Weight tying between embedding and LM head is not used because the
      factorised embedding (128-dim bottleneck) means shapes do not match.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn

from UniMamba import UniMamba, RMSNorm
from LatentMoE import LatentMoE
from cross_model_sparse_attn import CrossModelSparseAttention


# ============================================================================
# 1.  DECODER BLOCK  --  UniMamba -> CrossAttn -> UniMambaMoE
# ============================================================================

class SpeakMK1DecoderBlock(nn.Module):
    """
    Single decoder block matching the diagram:

        x -> UniMamba -> CrossModelSparseAttention -> UniMambaMoE -> out

    CrossModelSparseAttention is skipped when audio_out=None, which is the
    case during text-only pretraining (Stages 1 and 2).

    Args:
        d_model:       hidden dimension
        d_state:       SSM state size for UniMamba
        d_conv:        depthwise conv width for UniMamba
        expand:        expansion ratio for UniMamba
        nheads_ssm:    heads inside UniMamba
        nheads_attn:   heads for CrossModelSparseAttention
        top_k_audio:   audio frames each text token attends to (sparse k)
        latent_dim:    LatentMoE bottleneck dim (None -> d_model // 4)
        num_experts:   number of routed experts in LatentMoE
        top_k_experts: experts activated per token
        dropout:       dropout throughout
    """

    def __init__(
        self,
        d_model:       int,
        d_state:       int = 64,
        d_conv:        int = 4,
        expand:        int = 2,
        nheads_ssm:    int = 8,
        nheads_attn:   int = 8,
        top_k_audio:   int = 32,
        latent_dim:    Optional[int] = None,
        num_experts:   int = 8,
        top_k_experts: int = 2,
        dropout:       float = 0.0,
    ):
        super().__init__()

        # UniMamba -- sequence mixing
        self.ssm = UniMamba(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            nheads=nheads_ssm,
        )

        # CrossModelSparseAttention -- audio grounding
        self.cross_attn = CrossModelSparseAttention(
            d_model=d_model,
            nheads=nheads_attn,
            top_k=top_k_audio,
            dropout=dropout,
            use_gate=True,
        )

        # UniMambaMoE -- UniMamba followed by sparse MoE FFN
        self.moe_ssm = UniMamba(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            nheads=nheads_ssm,
        )
        self.moe = LatentMoE(
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
            aux_loss: scalar router load-balancing loss from LatentMoE
        """
        # UniMamba
        x = self.ssm(x)

        # CrossModelSparseAttention (skipped during text-only pretraining)
        if audio_out is not None:
            x = self.cross_attn(
                text_hidden=x,
                audio_out=audio_out,
                audio_padding_mask=audio_padding_mask,
            )

        # UniMambaMoE
        x = self.moe_ssm(x)
        x = self.moe(x)

        return x, self.moe.last_aux_loss


# ============================================================================
# 2.  CONFIG
# ============================================================================

@dataclass
class SpeakMK1LLMConfig:
    # Vocabulary
    vocab_size:    int   = 50277       # GPT-NeoX tokenizer

    # Model dimensions
    d_model:       int   = 512
    d_state:       int   = 64
    d_conv:        int   = 4
    expand:        int   = 2

    # Block count
    num_blocks:    int   = 6           # number of DecoderBlocks

    # Attention
    nheads_ssm:    int   = 8
    nheads_attn:   int   = 8
    top_k_audio:   int   = 32

    # MoE
    latent_dim:    Optional[int] = None   # None -> d_model // 4 inside LatentMoE
    num_experts:   int   = 8
    top_k_experts: int   = 2

    # Regularisation
    dropout:       float = 0.0

    # Training
    aux_loss_weight: float = 1e-2

class SpeakMK1Embedding(nn.Module):
    """Wrapper to make embedding callable like nn.Embedding."""
    def __init__(self, embedding, embed_proj):
        super().__init__()
        self.embedding = embedding
        self.embed_proj = embed_proj
    
    def forward(self, input_ids):
        return self.embed_proj(self.embedding(input_ids))

# ============================================================================
# 3.  FULL LLM
# ============================================================================

class SpeakMK1LLM(nn.Module):
    """
    Full SpeakMK1 LLM.

    Text-only forward (Stages 1 and 2):
        logits, loss, aux_loss = model(input_ids, labels=labels)

    Multimodal forward (Stage 3+):
        logits, loss, aux_loss = model(
            input_ids,
            labels=labels,
            audio_out=encoder_tokens,
            audio_padding_mask=padding_mask,
        )

    Training loss:
        total_loss = loss + config.aux_loss_weight * aux_loss
    """

    def __init__(self, config: SpeakMK1LLMConfig):
        super().__init__()
        self.config = config
        D = config.d_model

        # Factorised embedding: vocab -> 128 -> d_model
        self.embedding  = nn.Embedding(config.vocab_size, 128)
        self.embed_proj = nn.Linear(128, D, bias=False)

        # [DecoderBlock] x num_blocks
        self.blocks = nn.ModuleList([
            SpeakMK1DecoderBlock(
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
                dropout=config.dropout,
            )
            for _ in range(config.num_blocks)
        ])

        # Final norm + LM head
        self.norm    = RMSNorm(D)
        self.lm_head = nn.Linear(D, config.vocab_size, bias=False)

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.embedding.weight, std=0.02)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def get_input_embeddings(self):
        """Return the full embedding pipeline for external use."""
        # Return a wrapper that applies both embedding + projection
        return SpeakMK1Embedding(self.embedding, self.embed_proj)

    def forward(
        self,
        input_ids:          torch.Tensor = None,
        inputs_embeds:      torch.Tensor = None,
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
            aux_loss:  scalar sum of all router load-balancing losses
        """
        if inputs_embeds is None:
            x = self.embed_proj(self.embedding(input_ids))
        else:
            x = inputs_embeds
        total_aux = x.new_zeros(1).squeeze()

        for block in self.blocks:
            x, aux = block(
                x,
                audio_out=audio_out,
                audio_padding_mask=audio_padding_mask,
            )
            total_aux = total_aux + aux

        total_aux = total_aux / len(self.blocks)

        x      = self.norm(x)
        logits = self.lm_head(x)                          # (B, L, vocab_size)

        loss = None
        if labels is not None:
            # shift_logits = logits[..., :-1, :].contiguous()
            # shift_labels = labels[..., 1:].contiguous()
            loss = nn.CrossEntropyLoss(ignore_index=-100)(
                logits.view(-1, logits.size(-1)),
                labels.view(-1),
            )

        return logits, loss, total_aux

    @torch.no_grad()
    def generate(
        self,
        input_ids:          torch.Tensor,
        max_new_tokens:     int   = 128,
        temperature:        float = 1.0,
        top_p:              float = 0.9,
        audio_out:          Optional[torch.Tensor] = None,
        audio_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Autoregressive generation with top-p nucleus sampling.

        Args:
            input_ids:          (B, L) prompt tokens
            max_new_tokens:     tokens to generate
            temperature:        softmax temperature
            top_p:              nucleus sampling threshold
            audio_out:          (B, T_audio, D) or None
            audio_padding_mask: (B, T_audio) True=padded, or None

        Returns:
            (B, L + max_new_tokens)
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

            sorted_logits, sorted_indices = torch.sort(next_logits, descending=True)
            cumprobs      = sorted_logits.softmax(dim=-1).cumsum(dim=-1)
            remove        = cumprobs - sorted_logits.softmax(dim=-1) > top_p
            sorted_logits = sorted_logits.masked_fill(remove, float("-inf"))
            next_logits   = torch.zeros_like(next_logits).scatter(1, sorted_indices, sorted_logits)

            probs      = next_logits.softmax(dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            generated  = torch.cat([generated, next_token], dim=1)

        return generated


# ============================================================================
# 4.  UTILITIES
# ============================================================================

def count_parameters(model: nn.Module) -> dict:
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        "total_M":     round(total / 1e6, 1),
        "trainable_M": round(trainable / 1e6, 1),
    }


# ============================================================================
# 5.  SMOKE TEST
# ============================================================================

if __name__ == "__main__":
    print("Building SpeakMK1LLM ...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device}")

    cfg = SpeakMK1LLMConfig(
        vocab_size=50277,
        d_model=512,
        d_state=64,
        num_blocks=6,
        nheads_ssm=8,
        nheads_attn=8,
        top_k_audio=32,
        num_experts=8,
        top_k_experts=2,
    )

    model  = SpeakMK1LLM(cfg).to(device)
    params = count_parameters(model)
    print(f"  Total parameters  : {params['total_M']}M")
    print(f"  Trainable params  : {params['trainable_M']}M")
    print(f"  d_model           : {cfg.d_model}")
    print(f"  Blocks x{cfg.num_blocks}  (UniMamba -> CrossAttn -> UniMambaMoE)")
    print(f"  Experts           : {cfg.num_experts} routed, top-{cfg.top_k_experts}")

    B, L = 2, 128
    input_ids = torch.randint(0, cfg.vocab_size, (B, L), device=device)
    labels    = torch.randint(0, cfg.vocab_size, (B, L), device=device)

    # 1. Text-only
    print("\n[1] Text-only forward (audio_out=None) ...")
    logits, loss, aux = model(input_ids, labels=labels)
    print(f"  logits : {logits.shape}")
    print(f"  CE loss: {loss.item():.4f}")
    print(f"  aux    : {aux.item():.6f}")
    assert logits.shape == (B, L, cfg.vocab_size)

    # 2. Multimodal
    print("\n[2] Multimodal forward (audio_out provided) ...")
    audio_tokens = torch.randn(B, 64, cfg.d_model, device=device)
    logits_mm, loss_mm, aux_mm = model(input_ids, labels=labels, audio_out=audio_tokens)
    print(f"  logits : {logits_mm.shape}")
    print(f"  CE loss: {loss_mm.item():.4f}")
    print(f"  aux    : {aux_mm.item():.6f}")

    # 3. Combined loss
    print("\n[3] Combined training loss ...")
    total_loss = loss_mm + cfg.aux_loss_weight * aux_mm
    print(f"  total_loss = {total_loss.item():.4f}")

    # 4. Generation
    print("\n[4] Generation ...")
    prompt = torch.randint(0, cfg.vocab_size, (1, 8), device=device)
    out    = model.generate(prompt, max_new_tokens=16, audio_out=audio_tokens[:1])
    print(f"  output shape: {out.shape}")
    assert out.shape == (1, 24)

    # 5. Single training step
    print("\n[5] Single training step sanity check ...")
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    model.train()
    logits, loss, aux = model(input_ids, labels=labels)
    total = loss + cfg.aux_loss_weight * aux
    total.backward()
    opt.step()
    opt.zero_grad()
    logits2, loss2, _ = model(input_ids, labels=labels)
    assert not torch.isnan(loss2)
    print(f"  loss before: {loss.item():.4f}  after: {loss2.item():.4f}")
    print("  Backward pass OK")

    print("\nDone")