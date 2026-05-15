import math
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import librosa
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from UniMamba import UniMamba, RMSNorm
from BiMamba import BiMamba
from LatentMoE import LatentMoE
from local_window_attn import LocalWindowAttention


# ════════════════════════════════════════════════════════════════════════════
# 1.  AUDIO FEATURE EXTRACTOR
# ════════════════════════════════════════════════════════════════════════════

class AudioExtractor(nn.Module):
    """Raw waveform → Log-Mel spectrogram tensor."""

    def __init__(self, sr: int = 16_000, n_mels: int = 80, hop_length: int = 160,
                 n_fft: int = 400):
        super().__init__()
        self.sr = sr
        self.n_mels = n_mels
        self.hop_length = hop_length
        self.n_fft = n_fft

    def extract(self, y: np.ndarray) -> np.ndarray:
        melspec = librosa.feature.melspectrogram(
            y=y, sr=self.sr, n_mels=self.n_mels,
            hop_length=self.hop_length, n_fft=self.n_fft,
        )
        return librosa.power_to_db(melspec)   # (n_mels, T)

    def forward(self, waveforms: torch.Tensor) -> torch.Tensor:
        device = waveforms.device
        batch = []
        for wav in waveforms.cpu().numpy():
            lm = self.extract(wav.astype(np.float32))
            batch.append(torch.from_numpy(lm).T)         # (T, n_mels)
        max_t = max(x.size(0) for x in batch)
        padded = torch.zeros(len(batch), max_t, self.n_mels)
        for i, x in enumerate(batch):
            padded[i, :x.size(0)] = x
        return padded.to(device)


# ════════════════════════════════════════════════════════════════════════════
# 2.  CgMLP
# ════════════════════════════════════════════════════════════════════════════

class CgMLP(nn.Module):
    """Convolutional-gated MLP: depth-wise conv + SiGLU gate."""

    def __init__(self, d_model: int, expansion: int = 4, kernel_size: int = 3,
                 dropout: float = 0.0):
        super().__init__()
        d_inner = d_model * expansion
        self.norm = RMSNorm(d_model)
        self.gate_proj = nn.Linear(d_model, d_inner * 2, bias=False)
        self.dw_conv = nn.Conv1d(d_inner, d_inner, kernel_size,
                                 padding=kernel_size // 2,
                                 groups=d_inner, bias=False)
        self.out_proj = nn.Linear(d_inner, d_model, bias=False)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm(x)
        gate, val = self.gate_proj(x).chunk(2, dim=-1)
        gate = F.silu(gate)
        val = self.dw_conv(val.transpose(1, 2)).transpose(1, 2)
        return self.out_proj(self.drop(gate * val))


# ════════════════════════════════════════════════════════════════════════════
# 3.  MERGE BLOCK  (Bi-Mamba MoE  +  CgMLP  → Add+Norm)
# ════════════════════════════════════════════════════════════════════════════

class BiMambaMoECgMLPBlock(nn.Module):
    """
    The combined block from panel (a):

        BiMambaMoE ─┐
                    ├─ Add ─ LayerNorm  →  output
        CgMLP      ─┘
    """

    def __init__(self, d_model: int, num_experts: int = 4,
                 cgmlp_expansion: int = 4, dropout: float = 0.0,
                 d_state: int = 64, d_conv: int = 4, expand: int = 2):
        super().__init__()

        # BiMamba receives d_model and inherits d_state/d_conv/expand
        bi_mamba = BiMamba(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            dropout=dropout,
        )

        # LatentMoE wraps BiMamba — base_block is stored but routing is independent
        self.bi_mamba_moe = LatentMoE(
            d_model=d_model,
            num_experts=num_experts,
            dropout=dropout,
            base_block=bi_mamba,   # accepted for compat, routing is MoE-only
        )

        self.cgmlp = CgMLP(d_model, expansion=cgmlp_expansion, dropout=dropout)
        self.merge_norm = RMSNorm(d_model)

    def forward(self, x: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        moe_out = self.bi_mamba_moe(x)
        cgmlp_out = self.cgmlp(x)
        return self.merge_norm(x + moe_out + cgmlp_out)


# ════════════════════════════════════════════════════════════════════════════
# 4.  SHARED PHONOLOGICAL LAYER + HEADS
# ════════════════════════════════════════════════════════════════════════════

class PhonologicalHead(nn.Module):
    def __init__(self, d_model: int, num_classes: int):
        super().__init__()
        self.proj = nn.Linear(d_model, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


class SharedPhonologicalLayer(nn.Module):
    HEADS = {
        "voicing":     2,
        "manner":      8,
        "ctc":         40,
        "place":       7,
        "correctness": 2,
    }

    def __init__(self, d_model: int, training_heads: bool = True):
        super().__init__()
        self.norm = RMSNorm(d_model)
        self.proj = nn.Linear(d_model, d_model)
        self.training_heads = training_heads

        if training_heads:
            self.heads = nn.ModuleDict({
                name: PhonologicalHead(d_model, n_cls)
                for name, n_cls in self.HEADS.items()
            })

    def forward(self, x, labels=None):
        normed = self.norm(x)
        projected = self.proj(F.silu(normed))
        x = x + projected 

        losses = None
        if self.training_heads and labels is not None:
            losses = {}
            for name, head in self.heads.items():
                if name in labels:
                    logits = head(normed)                          # (B, T, C)
                    tgt = labels[name]                        # (B, T)
                    n_classes = logits.size(-1)
                    
                    # Fix: ensure padding is exactly -100 and nothing else
                    # is out of range. Clamp valid labels to [0, n_classes-1]
                    valid_mask = tgt != -100
                    tgt = tgt.clone()
                    tgt[valid_mask] = tgt[valid_mask].clamp(0, n_classes - 1)
                    
                    losses[name] = F.cross_entropy(
                        logits.reshape(-1, n_classes),
                        tgt.reshape(-1),
                        ignore_index=-100,
                    )
        return x, losses


# ════════════════════════════════════════════════════════════════════════════
# 5.  Q-FORMER PROJECTION
# ════════════════════════════════════════════════════════════════════════════

class QFormerProjection(nn.Module):
    """BLIP-2 style learnable query projection."""

    def __init__(self, d_model: int, llm_dim: int, num_queries: int = 64,
                 num_heads: int = 8, num_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.queries = nn.Parameter(torch.empty(1, num_queries, d_model))
        nn.init.trunc_normal_(self.queries, std=0.02 * math.sqrt(d_model))
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=num_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout, batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.out_proj = nn.Linear(d_model, llm_dim)

    def forward(self, memory: torch.Tensor,
                memory_key_padding_mask: Optional[torch.Tensor] = None
                ) -> torch.Tensor:
        B = memory.size(0)
        q = self.queries.expand(B, -1, -1)
        out = self.transformer(q, memory,
                               memory_key_padding_mask=memory_key_padding_mask)
        return self.out_proj(out)


# ════════════════════════════════════════════════════════════════════════════
# 5b.  DIRECT PROJECTION (Cuz the Qformer refuses to learn anything in early tests)
# ════════════════════════════════════════════════════════════════════════════

class DirectProjection(nn.Module):
    def __init__(self, d_model: int, llm_dim: int, dropout: float = 0.1):
        super().__init__()
        self.norm = RMSNorm(d_model)
        self.proj = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, llm_dim),
        )

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        # x: (B, T, d_model) → (B, T, llm_dim)
        return self.proj(self.norm(x))

# ════════════════════════════════════════════════════════════════════════════
# 6.  CONFIG
# ════════════════════════════════════════════════════════════════════════════

@dataclass
class AudioEncoderConfig:
    # ── Feature extraction ────────────────────────────────────────────────
    sample_rate:        int   = 16_000
    n_mels:             int   = 80
    hop_length:         int   = 160
    n_fft:              int   = 400

    # ── Model dimensions ──────────────────────────────────────────────────
    d_model:            int   = 1024
    dstate:             int   = 64      # SSM state size
    d_conv:             int   = 4       # Mamba depthwise conv width (passed through)
    expand:             int   = 2       # Mamba expansion (passed through)

    # ── Stack depths ──────────────────────────────────────────────────────
    num_uni_mamba:      int   = 6
    num_bi_mamba_moe:   int   = 4

    # ── Mixture of Experts ────────────────────────────────────────────────
    num_experts:        int   = 4

    # ── Local attention (RMS-ATT) ─────────────────────────────────────────
    attn_heads:         int   = 16      # 1024 / 16 = 64-dim per head
    window_size:        int   = 64

    # ── CgMLP ─────────────────────────────────────────────────────────────
    cgmlp_expansion:    int   = 4

    # ── Q-Former ──────────────────────────────────────────────────────────
    num_queries:        int   = 64
    num_qformer_layers: int   = 2
    llm_dim:            int   = 4096

    # ── Training phonological heads ───────────────────────────────────────
    training_heads:     bool  = True

    # ── Regularisation ────────────────────────────────────────────────────
    dropout:            float = 0.1


# ════════════════════════════════════════════════════════════════════════════
# 7.  FULL AUDIO ENCODER
# ════════════════════════════════════════════════════════════════════════════

class AudioEncoder(nn.Module):
    """
    Complete Audio Encoder for SpeakMK1.

    Forward pass (inference):
        encoder(mel)  →  (B, num_queries, llm_dim)

    Forward pass (training):
        encoder(mel, labels=labels)  →  ((B, num_queries, llm_dim), loss_dict)
    """

    def __init__(self, config: AudioEncoderConfig):
        super().__init__()
        cfg = config
        D = cfg.d_model

        # ── Feature extractor ──────────────────────────────────────────────
        self.extractor = AudioExtractor(
            sr=cfg.sample_rate, n_mels=cfg.n_mels,
            hop_length=cfg.hop_length, n_fft=cfg.n_fft,
        )

        # ── Stem: project n_mels → d_model ────────────────────────────────
        self.stem = nn.Sequential(
            nn.Linear(cfg.n_mels, D),
            nn.GELU(),
        )

        # ── Uni-Mamba stack  (×M) ─────────────────────────────────────────
        # Pass all matching kwargs; UniMamba accepts d_model, d_state, d_conv, expand
        self.uni_mamba_layers = nn.ModuleList([
            UniMamba(
                d_model=D,
                d_state=cfg.dstate,
                d_conv=cfg.d_conv,
                expand=cfg.expand,
            )
            for _ in range(cfg.num_uni_mamba)
        ])

        # ── RMS-ATT  (local window attention) ─────────────────────────────
        # Use num_heads / window_size aliases (accepted by LocalWindowAttention)
        self.rms_att = LocalWindowAttention(
            d_model=D,
            num_heads=cfg.attn_heads,
            window_size=cfg.window_size,
            dropout=cfg.dropout,
        )

        # ── Bi-Mamba MoE + CgMLP  (×N) ────────────────────────────────────
        self.bi_mamba_moe_layers = nn.ModuleList([
            BiMambaMoECgMLPBlock(
                d_model=D,
                num_experts=cfg.num_experts,
                cgmlp_expansion=cfg.cgmlp_expansion,
                dropout=cfg.dropout,
                d_state=cfg.dstate,
                d_conv=cfg.d_conv,
                expand=cfg.expand,
            )
            for _ in range(cfg.num_bi_mamba_moe)
        ])

        # ── Shared Phonological Layer ──────────────────────────────────────
        self.phonological = SharedPhonologicalLayer(
            d_model=D, training_heads=cfg.training_heads
        )

        # ── Q-Former Projection → LLM tokens ──────────────────────────────
        self.qformer = QFormerProjection(
            d_model=D, llm_dim=cfg.llm_dim,
            num_queries=cfg.num_queries,
            num_heads=cfg.attn_heads,
            num_layers=cfg.num_qformer_layers,
            dropout=cfg.dropout,
        )

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.trunc_normal_(module.weight, std=0.02)

    @staticmethod
    def _make_key_padding_mask(mel: torch.Tensor) -> torch.Tensor:
        """Returns (B, T) bool mask; True where frame is padding."""
        return (mel.sum(dim=-1) == 0)
    
    def encode_features(self, mel, attention_mask=None):
        """Returns raw (B, T, d_model) before QFormer."""
        x = self.stem(mel)
        for layer in self.uni_mamba_layers:
            x = layer(x)
        pad_mask = (attention_mask == 0) if attention_mask is not None \
                    else self._make_key_padding_mask(mel)
        x = self.rms_att(x, key_padding_mask=pad_mask)
        for layer in self.bi_mamba_moe_layers:
            x = layer(x)
        x, _ = self.phonological(x)
        return x   # (B, T, d_model) — feeds into DirectAudioProjection

    def forward(
        self,
        audio_input: torch.Tensor,
        labels: Optional[Dict[str, torch.Tensor]] = None,
        use_extractor: bool = False,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[Dict[str, torch.Tensor]]]:
        # ── 0. Feature extraction ─────────────────────────────────────────
        if use_extractor:
            mel = self.extractor(audio_input)
        else:
            mel = audio_input

        # ── 1. Stem projection ────────────────────────────────────────────
        x = self.stem(mel)                             # (B, T, D)

        # ── 2. Uni-Mamba stack ────────────────────────────────────────────
        for layer in self.uni_mamba_layers:
            x = layer(x)

        # ── 3. RMS-ATT (local window attention) ───────────────────────────
        pad_mask = (attention_mask == 0) if attention_mask is not None \
                   else self._make_key_padding_mask(mel)
        x = self.rms_att(x, key_padding_mask=pad_mask)

        # ── 4. Bi-Mamba MoE + CgMLP stack ────────────────────────────────
        for layer in self.bi_mamba_moe_layers:
            x = layer(x, attention_mask=attention_mask)

        # ── 5. Shared Phonological Layer ──────────────────────────────────
        x, losses = self.phonological(x, labels=labels)

        # ── 6. Q-Former Projection → fixed LLM token budget ───────────────
        llm_tokens = self.qformer(x, memory_key_padding_mask=pad_mask)

        return llm_tokens, losses

    @torch.no_grad()
    def encode(self, audio_input: torch.Tensor,
               use_extractor: bool = False) -> torch.Tensor:
        self.eval()
        tokens, _ = self.forward(audio_input, use_extractor=use_extractor)
        return tokens


# ════════════════════════════════════════════════════════════════════════════
# 8.  LOSS AGGREGATOR
# ════════════════════════════════════════════════════════════════════════════

def aggregate_losses(losses: Dict[str, torch.Tensor],
                     weights: Optional[Dict[str, float]] = None) -> torch.Tensor:
    default_w = {"voicing": 1.0, "manner": 1.0, "ctc": 1.0,
                 "place": 1.0, "correctness": 1.0}
    w = {**default_w, **(weights or {})}
    return sum(w.get(k, 1.0) * v for k, v in losses.items())


# ════════════════════════════════════════════════════════════════════════════
# 9.  QUICK SMOKE TEST
# ════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("Building AudioEncoder  [d_model=1024, dstate=64] …")
    cfg = AudioEncoderConfig()

    model = AudioEncoder(cfg)
    total = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"  Parameters : {total:.1f}M")
    print(f"  d_model    : {cfg.d_model}")
    print(f"  dstate     : {cfg.dstate}")
    print(f"  num_queries: {cfg.num_queries}  ->  ({cfg.num_queries}, {cfg.llm_dim}) LLM tokens")

    B, T = 2, 200
    mel = torch.randn(B, T, cfg.n_mels)

    labels = {
        "voicing":     torch.randint(0, 2,  (B, T)),
        "manner":      torch.randint(0, 8,  (B, T)),
        "ctc":         torch.randint(0, 40, (B, T)),
        "place":       torch.randint(0, 7,  (B, T)),
        "correctness": torch.randint(0, 2,  (B, T)),
    }

    tokens, losses = model(mel, labels=labels)
    print(f"  Output shape : {tokens.shape}")
    assert tokens.shape == (B, cfg.num_queries, cfg.llm_dim), "Shape mismatch!"

    if losses:
        total_loss = aggregate_losses(losses)
        print(f"  Total aux loss : {total_loss.item():.4f}")
        for name, val in losses.items():
            print(f"    {name:>12s} : {val.item():.4f}")

    print("Done ✓")