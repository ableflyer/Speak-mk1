from typing import Optional, Tuple
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

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

    Args:
        d_model:        hidden dimension
        latent_dim:     bottleneck dimension for routed experts.
                        Defaults to d_model // 4 if not given.
        num_experts:    number of routed experts
        top_k:          experts activated per token
        base_block:     optional nn.Module to wrap (ignored in routing logic;
                        kept for API compatibility with BiMambaMoECgMLPBlock)
    """

    def __init__(
        self,
        d_model: int,
        latent_dim: Optional[int] = None,
        num_experts: int = 8,
        top_k: int = 2,
        expert_dim_ff: Optional[int] = None,
        shared_expert: bool = True,
        shared_ff_mult: int = 4,
        aux_loss_coeff: float = 1e-2,
        dropout: float = 0.0,
        # API compat: accept (and ignore) a base_block kwarg used by BiMambaMoECgMLPBlock
        base_block: Optional[nn.Module] = None,
    ):
        super().__init__()
        self.base_block = base_block
        # Default latent_dim to d_model // 4 if not supplied
        if latent_dim is None:
            latent_dim = max(d_model // 4, 64)

        assert latent_dim <= d_model, \
            f"latent_dim ({latent_dim}) must be <= d_model ({d_model})"
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
        self.current_step = 0

    def _route(self, x_flat: torch.Tensor):
        logits = self.router(x_flat)  # (N, E)

        if self.training:
            # FIXED: much smaller noise, only early training
            noise_std = max(0.01, 0.1 * (1.0 - self.current_step / 5000))
            logits = logits + torch.randn_like(logits) * noise_std

        gates_topk, indices = torch.topk(logits, self.top_k, dim=-1)
        gates_topk = torch.softmax(gates_topk, dim=-1)

        # FIXED: track ALL top_k not just top-1
        router_prob = torch.softmax(logits, dim=-1)
        N = x_flat.shape[0]
        one_hot = torch.zeros(N, self.num_experts, device=x_flat.device)
        one_hot.scatter_(1, indices, 1.0)  # all top_k experts
        f_e = one_hot.float().mean(0) / self.top_k
        P_e = router_prob.mean(0)
        aux = self.num_experts * (f_e * P_e).sum()

        return gates_topk, indices, aux

    
    def _dispatch_combine(self, z, gates, indices):
        # FIXED: sparse dispatch, only compute assigned experts
        N, L = z.shape
        out = torch.zeros(N, L, device=z.device, dtype=z.dtype)

        for k in range(self.top_k):
            expert_idx = indices[:, k]   # (N,)
            gate_w = gates[:, k]         # (N,)

            for e in range(self.num_experts):
                mask = (expert_idx == e)
                if not mask.any():
                    continue
                z_e = z[mask]                          # (n_e, L)
                h = F.silu(z_e @ self.expert_W1[e].T) # (n_e, d_ff)
                z_out_e = h @ self.expert_W2[e].T      # (n_e, L)
                out[mask] += gate_w[mask].unsqueeze(1) * z_out_e

        return out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.base_block is not None:
            x = self.base_block(x)

        B, L, D = x.shape
        residual = x
        x_normed = self.norm(x)
        x_flat = x_normed.view(B * L, D)

        gates, indices, aux_loss = self._route(x_flat)

        z = self.W_down(x_flat)
        z_out = self._dispatch_combine(z, gates, indices)
        y_routed = self.W_up(z_out).view(B, L, D)

        # FIXED: shared expert also uses normed input consistently
        if self.has_shared:
            y_shared = self.shared_expert(x_normed.view(B, L, D))
            out = self.drop(y_routed) + y_shared
        else:
            out = self.drop(y_routed)

        self.last_aux_loss = aux_loss
        return residual + out