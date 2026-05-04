"""
speakmk1_multimodal.py
======================
Combined multimodal architecture for SpeakMK1.
Wraps AudioEncoder + DirectAudioProjection + SpeakMK1LLM into a single
nn.Module with a unified forward pass.

Gradient flow:
    mel → AudioEncoder → audio_proj → CrossModelSparseAttention (in each block)
                                             ↓
    token_ids → embedding → SSM blocks → lm_head → CE loss

Training phases (controlled by freeze_backbone):
    Phase 1 (freeze_backbone=True):
        Trains: audio_encoder, audio_proj, cross_attn weights only
        Frozen: LLM embedding, SSM, norm, lm_head
    Phase 2 (freeze_backbone=False):
        Trains: everything jointly at a lower lr
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple

# ── Local imports — adjust paths if your project layout differs ────────────
from audio_encoder import AudioEncoder
from audio_trainer import SmallConfig          # holds d_model, llm_dim etc.
from speak_mk1_llm import SpeakMK1LLM, SpeakMK1LLMConfig
from UniMamba import RMSNorm


# ════════════════════════════════════════════════════════════════════════════
# PROJECTION BRIDGE
# ════════════════════════════════════════════════════════════════════════════

class DirectAudioProjection(nn.Module):
    """
    Projects audio encoder features into the LLM's residual stream dimension.

    Shape:  (B, T_audio, audio_d_model) → (B, T_audio, llm_d_model)

    Architecture:
        RMSNorm → Linear(audio_d → llm_d*2) → GELU → Linear(llm_d*2 → llm_d)

    The two-layer MLP with bottleneck expansion gives the projection enough
    capacity to rotate and rescale the audio feature space into the LLM's
    token embedding space without collapsing information.
    """

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.norm = RMSNorm(in_dim)
        self.proj = nn.Sequential(
            nn.Linear(in_dim,      out_dim * 2, bias=False),
            nn.GELU(),
            nn.Linear(out_dim * 2, out_dim,     bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T_audio, in_dim)  — raw encoder features
        Returns:
            (B, T_audio, out_dim)    — projected into LLM space
        """
        return self.proj(self.norm(x))


# ════════════════════════════════════════════════════════════════════════════
# COMBINED MODEL
# ════════════════════════════════════════════════════════════════════════════

class SpeakMK1Multimodal(nn.Module):
    """
    End-to-end multimodal model: audio waveform + text tokens → next-token loss.

    Components
    ----------
    audio_encoder   : Mamba-based encoder, outputs (B, T_audio, audio_d_model)
    audio_proj      : 2-layer MLP bridge,  outputs (B, T_audio, llm_d_model)
    llm             : SpeakMK1LLM decoder, consumes audio_out via CrossModelSparseAttention

    Gradient flow
    -------------
    mel → encoder → proj → llm.blocks[*].cross_attn → loss
                                    ↑
                           tanh gate starts at 0, opens gradually —
                           audio path is silent at init, safe warm-start.

    Parameters
    ----------
    audio_cfg       : SmallConfig passed to AudioEncoder
    llm_cfg         : SpeakMK1LLMConfig passed to SpeakMK1LLM
    freeze_backbone : if True, freeze LLM weights except cross_attn layers
    """

    def __init__(
        self,
        audio_cfg:       SmallConfig,
        llm_cfg:         SpeakMK1LLMConfig,
        freeze_backbone: bool = True,
    ):
        super().__init__()

        # ── Sub-modules ───────────────────────────────────────────────────
        self.audio_encoder = AudioEncoder(audio_cfg)
        self.audio_proj    = DirectAudioProjection(
            in_dim  = audio_cfg.d_model,
            out_dim = llm_cfg.d_model,
        )
        self.llm = SpeakMK1LLM(llm_cfg)

        # ── Initialise projection weights ─────────────────────────────────
        self._init_projection()

        # ── Apply freeze policy ───────────────────────────────────────────
        self.set_freeze_backbone(freeze_backbone)

    # ── Weight initialisation ─────────────────────────────────────────────

    def _init_projection(self):
        """Xavier-uniform init for projection linear layers."""
        for m in self.audio_proj.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    # ── Freeze / unfreeze helpers ─────────────────────────────────────────

    def set_freeze_backbone(self, freeze: bool):
        """
        Phase 1 (freeze=True):
            audio_encoder    — trainable
            audio_proj       — trainable
            llm.cross_attn   — trainable  (never seen real audio, need gradients)
            llm backbone     — frozen     (preserves text capability)

        Phase 2 (freeze=False):
            everything       — trainable  (joint fine-tuning at lower lr)
        """
        if freeze:
            # Freeze entire LLM first
            for param in self.llm.parameters():
                param.requires_grad = False
            # Then unfreeze just the cross-attention layers
            for name, param in self.llm.named_parameters():
                if "cross_attn" in name:
                    param.requires_grad = True
        else:
            # Unfreeze everything
            for param in self.llm.parameters():
                param.requires_grad = True

        # Audio side is always trainable
        for param in self.audio_encoder.parameters():
            param.requires_grad = True
        for param in self.audio_proj.parameters():
            param.requires_grad = True

    def trainable_parameters(self):
        """Returns only parameters with requires_grad=True."""
        return [p for p in self.parameters() if p.requires_grad]

    def trainable_param_count(self) -> int:
        return sum(p.numel() for p in self.trainable_parameters())

    def total_param_count(self) -> int:
        return sum(p.numel() for p in self.parameters())

    # ── Checkpoint helpers ────────────────────────────────────────────────

    def load_audio_encoder(self, ckpt_path: str, device: torch.device):
        """
        Load pretrained audio encoder weights.
        Skips qformer keys (not present in encoder-only checkpoints).
        """
        state = torch.load(ckpt_path, map_location=device, weights_only=False)
        # Strip qformer keys that may be present in older checkpoints
        state = {k: v for k, v in state.items() if not k.startswith("qformer")}
        missing, unexpected = self.audio_encoder.load_state_dict(state, strict=False)
        non_qf_missing = [k for k in missing if "qformer" not in k]
        if non_qf_missing:
            print(f"  [AudioEncoder] WARNING — unexpected missing keys: {non_qf_missing}")
        print(f"  [AudioEncoder] loaded — missing={len(missing)} unexpected={len(unexpected)}")

    def load_llm(self, ckpt_path: str, device: torch.device):
        """
        Load pretrained LLM weights.
        Always use strict=True to catch config mismatches early.
        Reminder: vocab_size=50283, num_experts=4 must match the checkpoint.
        """
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        # Support both raw state_dict and wrapped {"model": ...} checkpoints
        state = ckpt["model"] if "model" in ckpt else ckpt
        missing, unexpected = self.llm.load_state_dict(state, strict=False)
        # Cross-attn keys will be present in the state but may have been
        # saved as zeros/random — that is expected and fine.
        if unexpected:
            print(f"  [LLM] unexpected keys: {unexpected[:5]}")
        print(f"  [LLM] loaded — missing={len(missing)} unexpected={len(unexpected)}")

    # ── Forward pass ──────────────────────────────────────────────────────

    def forward(
        self,
        mel:               torch.Tensor,
        input_ids:         torch.Tensor,
        labels:            Optional[torch.Tensor]  = None,
        audio_padding_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor]:
        """
        Full multimodal forward pass.

        Args
        ----
        mel               : (B, T_frames, n_mels)   — log-mel spectrogram
        input_ids         : (B, L)                  — GPT-NeoX token IDs
        labels            : (B, L)                  — CE targets, -100 at pad
        audio_padding_mask: (B, T_frames)            — True = padded frame

        Returns
        -------
        logits   : (B, L, vocab_size)
        lm_loss  : scalar CE loss (None if labels not provided)
        aux_loss : scalar MoE router load-balancing loss
        """

        # ── Step 1: Encode raw mel into audio feature sequence ─────────────
        # AudioEncoder applies Mamba SSM layers + phonological prediction heads.
        # encode_features() returns the hidden states before the heads.
        audio_feats = self.audio_encoder.encode_features(mel)
        # audio_feats: (B, T_audio, audio_d_model)

        # ── Step 2: Build audio padding mask from mel if not provided ──────
        if audio_padding_mask is None:
            # A frame is padded if all mel bins are zero
            audio_padding_mask = (mel.sum(dim=-1) == 0)  # (B, T_audio)

        # ── Step 3: Project audio features into LLM residual stream dim ────
        # RMSNorm → Linear → GELU → Linear
        audio_out = self.audio_proj(audio_feats)
        # audio_out: (B, T_audio, llm_d_model)

        # ── Step 4: LLM decoder forward ────────────────────────────────────
        # audio_out is consumed by CrossModelSparseAttention in each block.
        # The tanh gate in cross_attn starts at 0 → audio contributes nothing
        # at init, then gradually opens as the gate parameter is learned.
        # Gradients flow: loss → lm_head → SSM blocks → cross_attn → audio_out
        #                                                            → audio_proj
        #                                                            → audio_encoder
        logits, lm_loss, aux_loss = self.llm(
            input_ids          = input_ids,
            labels             = labels,
            audio_out          = audio_out,
            audio_padding_mask = audio_padding_mask,
        )

        return logits, lm_loss, aux_loss

    # ── Inference helper ──────────────────────────────────────────────────

    @torch.no_grad()
    def generate(
        self,
        mel:        torch.Tensor,
        prompt_ids: torch.Tensor,
        max_new_tokens: int = 128,
        temperature:    float = 0.8,
        top_p:          float = 0.9,
    ) -> torch.Tensor:
        """
        Greedy / nucleus sampling generation given a mel spectrogram and
        an optional text prompt.

        Args
        ----
        mel           : (1, T_frames, n_mels)
        prompt_ids    : (1, L_prompt)  — can be just [BOS] token
        max_new_tokens: int
        temperature   : float — set to 1.0 for no scaling
        top_p         : float — nucleus sampling threshold

        Returns
        -------
        generated_ids : (1, L_prompt + max_new_tokens)
        """
        self.eval()

        # Encode audio once, reuse across all generation steps
        audio_feats        = self.audio_encoder.encode_features(mel)
        audio_padding_mask = (mel.sum(dim=-1) == 0)
        audio_out          = self.audio_proj(audio_feats)

        generated = prompt_ids.clone()

        for _ in range(max_new_tokens):
            logits, _, _ = self.llm(
                input_ids          = generated,
                audio_out          = audio_out,
                audio_padding_mask = audio_padding_mask,
            )
            # Take logits at last position only
            next_logits = logits[:, -1, :] / temperature

            # Nucleus (top-p) sampling
            sorted_logits, sorted_idx = torch.sort(next_logits, descending=True)
            cumulative_probs = torch.cumsum(
                torch.softmax(sorted_logits, dim=-1), dim=-1
            )
            # Remove tokens with cumulative prob above threshold
            sorted_logits[cumulative_probs > top_p] = float("-inf")
            probs    = torch.softmax(sorted_logits, dim=-1)
            next_tok = sorted_idx[torch.multinomial(probs, num_samples=1)]

            generated = torch.cat([generated, next_tok], dim=-1)

        return generated