from typing import Optional, Tuple
import math
 
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
 
from mamba_ssm.ops.triton.ssd_combined import mamba_chunk_scan_combined
from UniMamba import RMSNorm

class LatentMoE(nn.Module):
    """
    LatentMoE flow:
        x (d)
        ├─ router logits = x @ W_router.T        (full d — gating quality preserved)
        │    → top-k gates
        ├─ z = W_down(x)                          (d → ℓ, shared, before dispatch)
        │    → top-k experts in latent space      (ℓ → d_ff → ℓ)
        │    → weighted combine
        │    → W_up(z_out)                        (ℓ → d, shared, after combine)
        └─ shared_expert(x)                       (always active, full d)
        out = y_routed + y_shared
    """
 
    def __init__(
        self,
        d_model: int,
        latent_dim: int,
        num_experts: int = 8,
        top_k: int = 2,
        expert_dim_ff: Optional[int] = None,
        shared_expert: bool = True,
        shared_ff_mult: int = 4,
        aux_loss_coeff: float = 1e-2,
        dropout: float = 0.0,
    ):
        super().__init__()
        assert latent_dim <= d_model
        assert top_k <= num_experts
 
        self.d_model      = d_model
        self.latent_dim   = latent_dim
        self.num_experts  = num_experts
        self.top_k        = top_k
        self.aux_loss_coeff = aux_loss_coeff
        d_ff = expert_dim_ff if expert_dim_ff is not None else latent_dim
 
        # router sees full d
        self.router = nn.Linear(d_model, num_experts, bias=False)
 
        # shared projections wrapping the entire routed path
        self.W_down = nn.Linear(d_model, latent_dim, bias=False)   # d → ℓ
        self.W_up   = nn.Linear(latent_dim, d_model, bias=False)   # ℓ → d
 
        # routed experts — entirely in latent space (ℓ → d_ff → ℓ)
        self.expert_W1 = nn.Parameter(torch.empty(num_experts, d_ff, latent_dim))
        self.expert_W2 = nn.Parameter(torch.empty(num_experts, latent_dim, d_ff))
        nn.init.kaiming_uniform_(self.expert_W1, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.expert_W2, a=math.sqrt(5))
 
        # shared expert — always active, operates in full d
        self.has_shared = shared_expert
        if shared_expert:
            d_sh = d_model * shared_ff_mult
            self.shared_expert = nn.Sequential(
                nn.Linear(d_model, d_sh, bias=False),
                nn.SiLU(),
                nn.Linear(d_sh, d_model, bias=False),
            )
 
        self.norm = RMSNorm(d_model)
        self.drop = nn.Dropout(dropout)
 
    def _route(self, x_flat: torch.Tensor):
        logits = self.router(x_flat)                         # (N, E)
        gates_topk, indices = torch.topk(logits, self.top_k, dim=-1)
        gates_topk = torch.softmax(gates_topk, dim=-1)
 
        router_prob = torch.softmax(logits, dim=-1)
        one_hot = torch.zeros_like(router_prob)
        one_hot.scatter_(1, indices[:, :1], 1.0)
        f_e = one_hot.mean(0)
        P_e = router_prob.mean(0)
        aux = self.aux_loss_coeff * self.num_experts * (f_e * P_e).sum()
 
        return gates_topk, indices, aux
 
    def _dispatch_combine(self, z, gates, indices):
        out = torch.zeros_like(z)
        for k in range(self.top_k):
            idx = indices[:, k]
            w   = gates[:, k]
            for e in range(self.num_experts):
                mask = idx == e
                if not mask.any():
                    continue
                z_e     = z[mask]
                h       = F.silu(z_e @ self.expert_W1[e].t())
                z_e_out = h @ self.expert_W2[e].t()
                out[mask] += w[mask, None] * z_e_out
        return out
 
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B, L, D = x.shape
        residual = x
        x = self.norm(x)
        x_flat = x.view(B * L, D)
 
        gates, indices, aux_loss = self._route(x_flat)
 
        z        = self.W_down(x_flat)
        z_out    = self._dispatch_combine(z, gates, indices)
        y_routed = self.W_up(z_out).view(B, L, D)
 
        if self.has_shared:
            out = self.drop(y_routed) + self.shared_expert(x)
        else:
            out = self.drop(y_routed)
 
        return residual + out, aux_loss