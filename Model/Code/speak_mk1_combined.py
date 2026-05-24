import torch
import torch.nn as nn
import torch.nn.functional as F
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
        endturn_id:      int = 50282,
    ):
        super().__init__()

        # ── Sub-modules ───────────────────────────────────────────────────
        self.audio_encoder = AudioEncoder(audio_cfg)
        self.audio_proj    = DirectAudioProjection(
            in_dim  = audio_cfg.d_model,
            out_dim = llm_cfg.d_model,
        )
        self.llm = SpeakMK1LLM(llm_cfg)
        self.endturn_id = endturn_id

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
        if freeze:
            # Freeze entire LLM first
            for param in self.llm.parameters():
                param.requires_grad = False
            # Then unfreeze just the cross-attention layers
            for name, param in self.llm.named_parameters():
                if "cross_attn" in name:
                    param.requires_grad = True
            # for block in self.llm.blocks:
            #     if hasattr(block, 'cross_attn') and block.cross_attn.use_gate:
            #         block.cross_attn.gate.requires_grad = False
        else:
            # Unfreeze everything
            for param in self.llm.parameters():
                param.requires_grad = True
            for block in self.llm.blocks:
                if hasattr(block, 'cross_attn') and block.cross_attn.use_gate:
                    block.cross_attn.gate.requires_grad = True

        # Audio side is always trainable
        for param in self.audio_encoder.parameters():
            param.requires_grad = True
        for param in self.audio_proj.parameters():
            param.requires_grad = True
        # for param in self.audio_to_text.parameters():
        #     param.requires_grad = True

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
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        # Handle both formats
        if "llm" in ckpt:
            state = ckpt["llm"]
        elif "model" in ckpt:
            state = ckpt["model"]
        else:
            state = ckpt
        missing, unexpected = self.llm.load_state_dict(state, strict=False)
        if unexpected:
            print(f"  [LLM] unexpected keys: {unexpected[:5]}")
        print(f"  [LLM] loaded — missing={len(missing)} unexpected={len(unexpected)}")
    # ── Forward pass ──────────────────────────────────────────────────────

    def forward(
        self,
        mel: torch.Tensor,
        input_ids: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        audio_padding_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor]:

        audio_feats = self.audio_encoder.encode_features(mel)

        if audio_padding_mask is None:
            audio_padding_mask = (mel.sum(dim=-1) == 0)

        audio_out = self.audio_proj(audio_feats)

        logits, lm_loss, aux_loss = self.llm(
            input_ids=input_ids,
            labels=labels,
            audio_out=audio_out,
            audio_padding_mask=audio_padding_mask,
        )

        return logits, lm_loss, aux_loss

    # ── Inference helper ──────────────────────────────────────────────────

    @torch.no_grad()
    def generate(
        self,
        mel:            torch.Tensor,
        prompt_ids:     torch.Tensor,
        max_new_tokens: int   = 128,
        temperature:    float = 0.9,
        top_p:          float = 0.92,
        repetition_penalty: float = 1.5,
    ) -> torch.Tensor:
        self.eval()
        print(f"mel mean={mel.mean():.4f} std={mel.std():.4f} nonzero={mel.nonzero().shape[0]}")
        audio_feats        = self.audio_encoder.encode_features(mel)
        print(f"audio_feats stats: {audio_feats}")
        print(f"audio_feats stats: mean={audio_feats.mean():.4f} std={audio_feats.std():.4f} max={audio_feats.abs().max():.4f}")
        audio_padding_mask = (mel.sum(dim=-1) == 0)
        audio_out          = self.audio_proj(audio_feats)
        print(f"audio_out stats: {audio_out}")
        print(f"audio_out stats: mean={audio_out.mean():.4f} std={audio_out.std():.4f}")
        audio_out = audio_out / (audio_out.std() + 1e-6)
        zero_mel = torch.zeros_like(mel)
        zero_feats = self.audio_encoder.encode_features(zero_mel)
        print(f"zero_feats mean={zero_feats.mean():.4f} std={zero_feats.std():.4f}")
        print(f"same as real? {torch.allclose(audio_feats, zero_feats, atol=1e-3)}")
        
        print(f"prompt_ids: {prompt_ids[0].tolist()}")

        generated  = prompt_ids.clone()
        endturn_id = self.endturn_id

        for _ in range(max_new_tokens):
            logits, _, _ = self.llm(
                input_ids          = generated,
                audio_out          = audio_out,
                audio_padding_mask = audio_padding_mask,
            )

            next_logits = logits[:, -1, :].clone()  # (1, vocab)

            # # Repetition penalty — applied before temperature
            # for token_id in set(generated[0].tolist()[-50:]):
            #     next_logits[0, token_id] /= repetition_penalty

            # Temperature
            next_logits = next_logits / temperature
            
            for token_id in set(generated[0].tolist()[-50:]):
                next_logits[0, token_id] /= repetition_penalty

            # Top-p nucleus sampling
            sorted_logits, sorted_indices = torch.sort(next_logits, descending=True)
            cumprobs = sorted_logits.softmax(dim=-1).cumsum(dim=-1)
            remove = cumprobs - sorted_logits.softmax(dim=-1) > top_p
            sorted_logits = sorted_logits.masked_fill(remove, float("-inf"))
            next_logits = torch.zeros_like(next_logits).scatter(1, sorted_indices, sorted_logits)

            probs    = torch.softmax(sorted_logits, dim=-1)
            next_idx = torch.multinomial(probs, num_samples=1)          # index in sorted space
            next_tok = sorted_indices.gather(1, next_idx)               # actual vocab ID

            generated = torch.cat([generated, next_tok], dim=-1)
            
            eot_id =0
            
            if next_tok.item() == endturn_id or next_tok.item() == eot_id:
                break

        return generated