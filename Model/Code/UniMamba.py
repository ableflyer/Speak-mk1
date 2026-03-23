import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from mamba_ssm.ops.triton.ssd_combined import mamba_chunk_scan_combined


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        orig_dtype = x.dtype
        x_fp32 = x.float()  # upcast to fp32 for numerical stability
        norm = x_fp32.pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return (x_fp32 * norm).to(orig_dtype) * self.weight


class RoPE(nn.Module):
    def __init__(self, dim: int, max_seq_len: int = 4096):
        super().__init__()
        self.dim = dim
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, x):
        """
        x: (batch, seq_len, dim)
        Applies RoPE in-place style — returns rotated x.
        """
        B, T, D = x.shape
        t = torch.arange(T, device=x.device).float()
        freqs = torch.outer(t, self.inv_freq)           # (T, D//2)
        emb = torch.cat([freqs, freqs], dim=-1)         # (T, D)
        cos = emb.cos().unsqueeze(0).to(x.dtype)        # (1, T, D)
        sin = emb.sin().unsqueeze(0).to(x.dtype)        # (1, T, D)

        # Rotate half
        x1, x2 = x[..., :D//2], x[..., D//2:]
        x_rot = torch.cat([-x2, x1], dim=-1)
        return x * cos + x_rot * sin


class MIMO(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.proj = nn.Linear(dim, dim, bias=False)

    def forward(self, x):
        return self.proj(x)


class UniMamba(nn.Module):
    """
    Uni-Mamba block as shown in diagram (b):

    Input → RMS-Norm → Linear Proj → MIMO → RMS-Norm → RoPE → [A, X, B, C] → SSM → Y → Multiply → MIMO → Linear Proj → Output
                                  └→ Linear Proj → MIMO → SILU ──────────────────────────────────────────────────────────────↑

    Args:
        d_model:    hidden dimension (replaces old `dim` parameter)
        d_state:    SSM state size (alias: dstate)
        d_conv:     accepted for API compatibility with AudioEncoderConfig; not used
                    internally since UniMamba uses a purely linear (no-conv) stem.
        expand:     accepted for API compatibility; inner dim is fixed to nheads*headdim.
    """
    def __init__(
        self,
        d_model: int,
        d_state: int = 64,
        d_conv: int = 4,        # accepted for compat, not used internally
        expand: int = 2,        # accepted for compat, not used internally
        nheads: int = 8,
        headdim: int = None,
        ngroups: int = 1,
        chunk_size: int = 256,
        dt_min: float = 0.001,
        dt_max: float = 0.1,
        dt_init_floor: float = 1e-4,
        eps: float = 1e-6,
    ):
        super().__init__()
        assert d_model % nheads == 0
        self.d_model = d_model
        self.dstate = d_state
        self.nheads = nheads
        self.headdim = headdim if headdim is not None else d_model // nheads
        self.ngroups = ngroups
        self.chunk_size = chunk_size

        # --- Pre-norm ---
        self.norm = RMSNorm(d_model, eps=eps)

        # --- SSM branch ---
        # Projects input into the SSM space: x, B, C projections
        # x: (nheads * headdim), B: (ngroups * dstate), C: (ngroups * dstate)
        ssm_in_dim = self.nheads * self.headdim + 2 * self.ngroups * self.dstate
        self.in_proj = nn.Linear(d_model, ssm_in_dim, bias=False)   # Linear Proj (SSM branch)
        self.in_mimo = MIMO(ssm_in_dim)                              # MIMO after first Linear Proj
        self.ssm_norm = RMSNorm(ssm_in_dim, eps=eps)                 # Second RMS-Norm before RoPE
        self.rope = RoPE(ssm_in_dim)                                 # RoPE

        # SSM parameters: A (log), dt, dt_bias
        self.A_log = nn.Parameter(
            torch.log(torch.rand(nheads) * (dt_max - dt_min) + dt_min)
        )
        self.dt_bias = nn.Parameter(torch.zeros(nheads))

        # D (skip connection / residual scalar or vector per head)
        self.D = nn.Parameter(torch.ones(nheads, self.headdim))

        # dt projection (seqlen → per-head dt)
        self.dt_proj = nn.Linear(ssm_in_dim, nheads, bias=False)

        # Post-SSM MIMO and output projection
        self.out_mimo = MIMO(self.nheads * self.headdim)
        self.out_proj = nn.Linear(self.nheads * self.headdim, d_model, bias=False)

        # --- Gate branch (bottom path in diagram) ---
        # Linear Proj → MIMO → SILU
        self.gate_proj = nn.Linear(d_model, self.nheads * self.headdim, bias=False)
        self.gate_mimo = MIMO(self.nheads * self.headdim)

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        """
        u: (batch, seqlen, d_model)
        returns: (batch, seqlen, d_model)
        """
        B_sz, T, D = u.shape

        # 1. Pre-norm
        u_norm = self.norm(u)  # (B, T, D)

        # ── Gate branch (bottom path) ──────────────────────────────────────
        # Linear Proj → MIMO → SILU
        gate = self.gate_proj(u_norm)          # (B, T, nheads*headdim)
        gate = self.gate_mimo(gate)            # (B, T, nheads*headdim)
        gate = F.silu(gate)                    # (B, T, nheads*headdim)

        # ── SSM branch (top path) ──────────────────────────────────────────
        # Linear Proj → MIMO → RMS-Norm → RoPE
        xBC = self.in_proj(u_norm)             # (B, T, ssm_in_dim)
        xBC = self.in_mimo(xBC)                # (B, T, ssm_in_dim)
        xBC = self.ssm_norm(xBC)               # (B, T, ssm_in_dim)
        xBC = self.rope(xBC)                   # (B, T, ssm_in_dim)  — applies RoPE

        # Split into x, B, C
        x_ssm, B_ssm, C_ssm = torch.split(
            xBC,
            [self.nheads * self.headdim, self.ngroups * self.dstate, self.ngroups * self.dstate],
            dim=-1
        )  # x: (B,T,H*P), B/C: (B,T,G*N)

        # Reshape for mamba_chunk_scan
        x_ssm = rearrange(x_ssm, "b l (h p) -> b l h p", h=self.nheads)
        B_ssm = rearrange(B_ssm, "b l (g n) -> b l g n", g=self.ngroups)
        C_ssm = rearrange(C_ssm, "b l (g n) -> b l g n", g=self.ngroups)

        # dt: (B, T, nheads)
        dt = self.dt_proj(xBC)                 # (B, T, nheads)

        A = -torch.exp(self.A_log.to(dtype=u.dtype))     # (nheads,)  — must be negative

        # 2. SSM via mamba chunk scan (Y output)
        # out: (B, T, nheads, headdim)
        out = mamba_chunk_scan_combined(
            x_ssm, dt, A, B_ssm, C_ssm,
            chunk_size=self.chunk_size,
            D=self.D.to(dtype=u.dtype),
            z=None,
            dt_bias=self.dt_bias,
            dt_softplus=True,
        )

        # 3. Multiply SSM output (Y) with gate
        out = rearrange(out, "b l h p -> b l (h p)")   # (B, T, H*P)
        out = out * gate                                 # element-wise gate (Multiply node)

        # 4. Post-SSM: MIMO → Linear Proj
        out = self.out_mimo(out)                         # (B, T, H*P)
        out = self.out_proj(out)                         # (B, T, d_model)

        return out + u                                   # residual add