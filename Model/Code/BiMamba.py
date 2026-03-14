from typing import Optional, Tuple
import math
 
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
 
from mamba_ssm.ops.triton.ssd_combined import mamba_chunk_scan_combined
from UniMamba import RMSNorm, RoPE, MIMO

class BiMamba(nn.Module):
    """
    Bidirectional Mamba block (diagram panel d).
 
    Mirrors UniMamba's internal structure exactly:
 
    Input → RMSNorm → in_proj → in_mimo → ssm_norm → RoPE → split(x, B, C)
                                                               → SSM (fwd) ──────────────┐
                                                    (flipped) → SSM (bwd) → flip back ──┤
                                                                                  Merge (sum)
                                                                               → out_mimo
                                                                               → Multiply (gate)
                                                                               → out_proj
                   └→ gate_proj → gate_mimo → SiLU ────────────────────────────────────↑
    """
 
    def __init__(
        self,
        d_model: int,
        nheads: int,
        headdim: int,
        dstate: int = 64,
        chunk_size: int = 256,
        ngroups: int = 1,
        dt_min: float = 0.001,
        dt_max: float = 0.1,
        eps: float = 1e-6,
        dropout: float = 0.0,
    ):
        super().__init__()
        assert d_model == nheads * headdim, "d_model must equal nheads * headdim"
 
        self.d_model  = d_model
        self.nheads   = nheads
        self.headdim  = headdim
        self.dstate   = dstate
        self.chunk_size = chunk_size
        self.ngroups  = ngroups
 
        # packed dim: x + B + C  (mirrors UniMamba's ssm_in_dim)
        ssm_in_dim = nheads * headdim + 2 * ngroups * dstate
 
        # --- outer pre-norm ---
        self.norm = RMSNorm(d_model, eps=eps)
 
        # --- SSM branch (top path — mirrors UniMamba exactly) ---
        self.in_proj  = nn.Linear(d_model, ssm_in_dim, bias=False)
        self.in_mimo  = MIMO(ssm_in_dim)
        self.ssm_norm = RMSNorm(ssm_in_dim, eps=eps)
        self.rope     = RoPE(ssm_in_dim)
 
        # dt projection on the full ssm_in_dim feature (same as UniMamba)
        self.dt_proj  = nn.Linear(ssm_in_dim, nheads, bias=False)
 
        # dt_bias as standalone nn.Parameter — NOT baked into dt_proj.bias
        # (matches UniMamba's self.dt_bias exactly)
        self.dt_bias  = nn.Parameter(torch.zeros(nheads))
        self._init_dt_bias(dt_min, dt_max)
 
        # SSM parameters
        self.A_log = nn.Parameter(
            torch.log(torch.rand(nheads) * (dt_max - dt_min) + dt_min)
        )
        self.D = nn.Parameter(torch.ones(nheads, headdim))
 
        # post-SSM: out_mimo → out_proj (mirrors UniMamba)
        self.out_mimo = MIMO(d_model)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
 
        # --- gate branch (bottom path — mirrors UniMamba exactly) ---
        self.gate_proj = nn.Linear(d_model, d_model, bias=False)
        self.gate_mimo = MIMO(d_model)
 
        self.drop = nn.Dropout(dropout)
 
    # ------------------------------------------------------------------
    def _init_dt_bias(self, dt_min: float, dt_max: float):
        """Initialise dt_bias so softplus(dt_bias) lands in [dt_min, dt_max]."""
        dt = torch.exp(
            torch.rand(self.nheads) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=1e-4)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_bias.copy_(inv_dt)
 
    # ------------------------------------------------------------------
    def _run_ssm(
        self,
        xBC: torch.Tensor,                       # (B, L, ssm_in_dim) — already prepared
        seq_idx: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:                            # (B, L, d_model)
        """
        Receives xBC already processed through in_proj → in_mimo → ssm_norm → rope.
        Splits into x / B / C, runs the SSM kernel, returns (B, L, d_model).
        xBC is prepared once outside and shared by both fwd and bwd passes.
        """
        x_ssm, B_ssm, C_ssm = torch.split(
            xBC,
            [self.nheads * self.headdim, self.ngroups * self.dstate, self.ngroups * self.dstate],
            dim=-1,
        )
        x_ssm = rearrange(x_ssm, "b l (h p) -> b l h p", h=self.nheads)
        B_ssm = rearrange(B_ssm, "b l (g n) -> b l g n", g=self.ngroups)
        C_ssm = rearrange(C_ssm, "b l (g n) -> b l g n", g=self.ngroups)
 
        dt = self.dt_proj(xBC)                           # (B, L, nheads)
        A  = -torch.exp(self.A_log.to(dtype=xBC.dtype))  # (nheads,) — must be negative
 
        y = mamba_chunk_scan_combined(
            x_ssm, dt, A, B_ssm, C_ssm,
            chunk_size=self.chunk_size,
            D=self.D.to(dtype=xBC.dtype),
            z=None,
            dt_bias=self.dt_bias,                        # standalone param, matches UniMamba
            dt_softplus=True,
            seq_idx=seq_idx,
        )
        return rearrange(y, "b l h p -> b l (h p)")      # (B, L, d_model)
 
    # ------------------------------------------------------------------
    def forward(
        self,
        x: torch.Tensor,                         # (B, L, d_model)
        seq_idx: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:                            # (B, L, d_model)
        residual = x
 
        # outer pre-norm
        xn = self.norm(x)                        # (B, L, d_model)
 
        # gate branch — mirrors UniMamba bottom path exactly
        gate = self.gate_proj(xn)                # (B, L, d_model)
        gate = self.gate_mimo(gate)              # (B, L, d_model)
        gate = F.silu(gate)                      # (B, L, d_model)
 
        # SSM branch — prepare xBC once, both directions share it
        xBC = self.in_proj(xn)                   # (B, L, ssm_in_dim)
        xBC = self.in_mimo(xBC)                  # (B, L, ssm_in_dim)
        xBC = self.ssm_norm(xBC)                 # (B, L, ssm_in_dim)
        xBC = self.rope(xBC)                     # (B, L, ssm_in_dim)
 
        # Front (causal) pass
        y_fwd = self._run_ssm(xBC, seq_idx)      # (B, L, d_model)
 
        # Back (anti-causal) pass
        # Flip the already-prepared xBC — not raw x — so we don't redo the
        # projection/norm/RoPE work, and positional encoding stays consistent
        xBC_flip     = xBC.flip(1)
        seq_idx_flip = seq_idx.flip(1) if seq_idx is not None else None
        y_bwd = self._run_ssm(xBC_flip, seq_idx_flip).flip(1)  # (B, L, d_model)
 
        # Merge (sum) → out_mimo → multiply gate → out_proj
        y = self.out_mimo(y_fwd + y_bwd)         # (B, L, d_model)
        y = y * gate                             # (B, L, d_model)
        y = self.out_proj(y)                     # (B, L, d_model)
 
        return residual + self.drop(y)
