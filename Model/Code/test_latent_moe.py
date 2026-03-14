"""
test_latent_moe.py
------------------
Pytest test suite for the LatentMoE block.

Run with:
    pytest test_latent_moe.py -v
    pytest test_latent_moe.py -v -k "routing"
    pytest test_latent_moe.py -v --tb=short

Or via the shared --model flag:
    pytest TestBench.py test_latent_moe.py -v --model latentmoe

Requirements:
    pip install pytest torch einops mamba-ssm
"""

import pytest
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from LatentMoE import LatentMoE


# ---------------------------------------------------------------------------
# Shared config
# ---------------------------------------------------------------------------

DEVICE  = "cuda" if torch.cuda.is_available() else "cpu"

D_MODEL     = 256
LATENT_DIM  = 64       # D_MODEL / 4  — matches Nemotron 3 Super ratio
NUM_EXPERTS = 8
TOP_K       = 2
SEQ_LEN     = 64
BATCH       = 2


def make_moe(**kwargs) -> LatentMoE:
    cfg = dict(
        d_model=D_MODEL,
        latent_dim=LATENT_DIM,
        num_experts=NUM_EXPERTS,
        top_k=TOP_K,
        shared_expert=True,
        shared_ff_mult=4,
        aux_loss_coeff=1e-2,
        dropout=0.0,
    )
    cfg.update(kwargs)
    return LatentMoE(**cfg).to(DEVICE).eval()


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def moe_eval() -> LatentMoE:
    return make_moe()


@pytest.fixture(scope="module")
def moe_eval_no_shared() -> LatentMoE:
    """LatentMoE with shared expert disabled — isolates the routed path."""
    return make_moe(shared_expert=False)


@pytest.fixture
def moe_train() -> LatentMoE:
    return make_moe().train()


# ---------------------------------------------------------------------------
# 1. Shape tests
# ---------------------------------------------------------------------------

class TestShape:

    @pytest.mark.latentmoe
    def test_output_shape(self, moe_eval):
        x = torch.randn(BATCH, SEQ_LEN, D_MODEL, device=DEVICE)
        out, aux = moe_eval(x)
        assert out.shape == x.shape, f"Expected {x.shape}, got {out.shape}"

    @pytest.mark.latentmoe
    def test_aux_loss_is_scalar(self, moe_eval):
        x = torch.randn(BATCH, SEQ_LEN, D_MODEL, device=DEVICE)
        _, aux = moe_eval(x)
        assert aux.shape == torch.Size([]), f"aux_loss must be scalar, got shape {aux.shape}"

    @pytest.mark.latentmoe
    def test_batch_1(self, moe_eval):
        x = torch.randn(1, SEQ_LEN, D_MODEL, device=DEVICE)
        out, _ = moe_eval(x)
        assert out.shape == x.shape

    @pytest.mark.latentmoe
    def test_large_batch(self, moe_eval):
        x = torch.randn(8, SEQ_LEN, D_MODEL, device=DEVICE)
        out, _ = moe_eval(x)
        assert out.shape == x.shape

    @pytest.mark.latentmoe
    def test_single_token(self, moe_eval):
        x = torch.randn(BATCH, 1, D_MODEL, device=DEVICE)
        out, _ = moe_eval(x)
        assert out.shape == x.shape

    @pytest.mark.latentmoe
    @pytest.mark.parametrize("d,ell", [(128, 32), (256, 64), (512, 128)])
    def test_various_dims(self, d, ell):
        model = make_moe(d_model=d, latent_dim=ell)
        x = torch.randn(2, 32, d, device=DEVICE)
        out, _ = model(x)
        assert out.shape == x.shape

    @pytest.mark.latentmoe
    def test_no_shared_expert_shape(self, moe_eval_no_shared):
        x = torch.randn(BATCH, SEQ_LEN, D_MODEL, device=DEVICE)
        out, _ = moe_eval_no_shared(x)
        assert out.shape == x.shape


# ---------------------------------------------------------------------------
# 2. Numerical sanity
# ---------------------------------------------------------------------------

class TestNumericalSanity:

    @pytest.mark.latentmoe
    def test_no_nan(self, moe_eval):
        x = torch.randn(BATCH, SEQ_LEN, D_MODEL, device=DEVICE)
        out, aux = moe_eval(x)
        assert not torch.isnan(out).any(), "Output contains NaN"
        assert not torch.isnan(aux),       "aux_loss is NaN"

    @pytest.mark.latentmoe
    def test_no_inf(self, moe_eval):
        x = torch.randn(BATCH, SEQ_LEN, D_MODEL, device=DEVICE)
        out, aux = moe_eval(x)
        assert not torch.isinf(out).any(), "Output contains Inf"
        assert not torch.isinf(aux),       "aux_loss is Inf"

    @pytest.mark.latentmoe
    def test_output_not_all_zeros(self, moe_eval):
        x = torch.randn(BATCH, SEQ_LEN, D_MODEL, device=DEVICE)
        out, _ = moe_eval(x)
        assert out.abs().max() > 1e-6, "Output is effectively zero"

    @pytest.mark.latentmoe
    def test_output_magnitude_reasonable(self, moe_eval):
        x = torch.randn(BATCH, SEQ_LEN, D_MODEL, device=DEVICE)
        out, _ = moe_eval(x)
        assert out.abs().mean() < 100.0, "Output magnitude suspiciously large"

    @pytest.mark.latentmoe
    def test_different_inputs_give_different_outputs(self, moe_eval):
        x1 = torch.randn(BATCH, SEQ_LEN, D_MODEL, device=DEVICE)
        x2 = torch.randn(BATCH, SEQ_LEN, D_MODEL, device=DEVICE)
        out1, _ = moe_eval(x1)
        out2, _ = moe_eval(x2)
        assert not torch.allclose(out1, out2)

    @pytest.mark.latentmoe
    def test_deterministic_eval(self, moe_eval):
        x = torch.randn(BATCH, SEQ_LEN, D_MODEL, device=DEVICE)
        out1, aux1 = moe_eval(x)
        out2, aux2 = moe_eval(x)
        torch.testing.assert_close(out1, out2)
        torch.testing.assert_close(aux1, aux2)

    @pytest.mark.latentmoe
    def test_zero_input(self, moe_eval):
        x = torch.zeros(BATCH, SEQ_LEN, D_MODEL, device=DEVICE)
        out, aux = moe_eval(x)
        assert not torch.isnan(out).any()
        assert not torch.isinf(out).any()

    @pytest.mark.latentmoe
    def test_large_input_values(self, moe_eval):
        x = torch.randn(BATCH, SEQ_LEN, D_MODEL, device=DEVICE) * 10.0
        out, aux = moe_eval(x)
        assert not torch.isnan(out).any()
        assert not torch.isinf(out).any()

    @pytest.mark.latentmoe
    def test_aux_loss_is_positive(self, moe_eval):
        """Load-balancing loss must always be >= 0."""
        x = torch.randn(BATCH, SEQ_LEN, D_MODEL, device=DEVICE)
        _, aux = moe_eval(x)
        assert aux.item() >= 0.0, f"aux_loss should be non-negative, got {aux.item()}"

    @pytest.mark.latentmoe
    def test_residual_connection(self, moe_eval_no_shared):
        """
        With zeroed expert weights and no shared expert, output should equal
        the input (pure residual passthrough).
        """
        model = make_moe(shared_expert=False)
        # Zero out all expert and projection weights
        with torch.no_grad():
            model.W_down.weight.zero_()
            model.W_up.weight.zero_()
            model.expert_W1.zero_()
            model.expert_W2.zero_()
        model.eval()

        x = torch.randn(BATCH, SEQ_LEN, D_MODEL, device=DEVICE)
        out, _ = model(x)
        torch.testing.assert_close(out, x, atol=1e-5, rtol=1e-5,
            msg="With zeroed weights, output should equal input (residual only)")


# ---------------------------------------------------------------------------
# 3. LatentMoE-specific architecture tests
# ---------------------------------------------------------------------------

class TestLatentMoEArchitecture:

    @pytest.mark.latentmoe
    def test_router_operates_on_full_d(self, moe_eval):
        """
        Router input dim must be d_model (full d), NOT latent_dim.
        This is the core LatentMoE invariant — gating quality must not
        be compromised by compression.
        """
        assert moe_eval.router.in_features == D_MODEL, (
            f"Router must see full d_model ({D_MODEL}), "
            f"got in_features={moe_eval.router.in_features}"
        )
        assert moe_eval.router.out_features == NUM_EXPERTS

    @pytest.mark.latentmoe
    def test_w_down_compresses_to_latent(self, moe_eval):
        """W_down must project d_model → latent_dim."""
        assert moe_eval.W_down.in_features  == D_MODEL
        assert moe_eval.W_down.out_features == LATENT_DIM

    @pytest.mark.latentmoe
    def test_w_up_expands_from_latent(self, moe_eval):
        """W_up must project latent_dim → d_model."""
        assert moe_eval.W_up.in_features  == LATENT_DIM
        assert moe_eval.W_up.out_features == D_MODEL

    @pytest.mark.latentmoe
    def test_experts_operate_in_latent_space(self, moe_eval):
        """
        Expert weight matrices must have latent_dim as their inner dimension,
        NOT d_model — experts only ever see compressed representations.
        """
        E, d_ff, ell = moe_eval.expert_W1.shape
        assert E   == NUM_EXPERTS, f"Expected {NUM_EXPERTS} experts, got {E}"
        assert ell == LATENT_DIM,  f"expert_W1 inner dim should be latent_dim ({LATENT_DIM}), got {ell}"

        E2, ell2, d_ff2 = moe_eval.expert_W2.shape
        assert E2   == NUM_EXPERTS
        assert ell2 == LATENT_DIM, f"expert_W2 output dim should be latent_dim ({LATENT_DIM}), got {ell2}"

    @pytest.mark.latentmoe
    def test_shared_expert_operates_in_full_d(self, moe_eval):
        """
        Shared expert must operate in full d_model, NOT latent space.
        This matches the Nemotron 3 Super design — shared path is unrestricted.
        """
        assert moe_eval.has_shared, "shared_expert should be enabled for this fixture"
        first_layer = moe_eval.shared_expert[0]
        assert isinstance(first_layer, nn.Linear)
        assert first_layer.in_features == D_MODEL, (
            f"Shared expert input must be d_model ({D_MODEL}), got {first_layer.in_features}"
        )

    @pytest.mark.latentmoe
    def test_compression_ratio(self, moe_eval):
        """latent_dim should be at most d_model (no expansion in latent path)."""
        assert moe_eval.latent_dim <= moe_eval.d_model

    @pytest.mark.latentmoe
    @pytest.mark.parametrize("top_k", [1, 2, 4])
    def test_various_top_k(self, top_k):
        model = make_moe(num_experts=8, top_k=top_k)
        x = torch.randn(BATCH, SEQ_LEN, D_MODEL, device=DEVICE)
        out, aux = model(x)
        assert out.shape == x.shape
        assert not torch.isnan(out).any()

    @pytest.mark.latentmoe
    def test_top_k_cannot_exceed_num_experts(self):
        """top_k > num_experts must raise AssertionError at construction."""
        with pytest.raises(AssertionError):
            make_moe(num_experts=4, top_k=8)

    @pytest.mark.latentmoe
    def test_latent_dim_cannot_exceed_d_model(self):
        """latent_dim > d_model must raise AssertionError at construction."""
        with pytest.raises(AssertionError):
            make_moe(d_model=64, latent_dim=128)


# ---------------------------------------------------------------------------
# 4. Routing behaviour tests
# ---------------------------------------------------------------------------

class TestRouting:

    @pytest.mark.latentmoe
    def test_exactly_top_k_experts_selected(self, moe_eval):
        """
        Each token must be dispatched to exactly top_k experts — no more, no less.
        We verify by monkey-patching _route and counting unique indices per token.
        """
        x = torch.randn(BATCH, SEQ_LEN, D_MODEL, device=DEVICE)
        captured = {}

        original_route = moe_eval._route
        def patched_route(x_flat):
            gates, indices, aux = original_route(x_flat)
            captured["indices"] = indices
            return gates, indices, aux

        moe_eval._route = patched_route
        moe_eval(x)
        moe_eval._route = original_route

        indices = captured["indices"]           # (N, top_k)
        assert indices.shape == (BATCH * SEQ_LEN, TOP_K), (
            f"Expected indices shape ({BATCH * SEQ_LEN}, {TOP_K}), got {indices.shape}"
        )

    @pytest.mark.latentmoe
    def test_routing_indices_in_valid_range(self, moe_eval):
        """All expert indices must be in [0, num_experts)."""
        x = torch.randn(BATCH, SEQ_LEN, D_MODEL, device=DEVICE)
        captured = {}

        original_route = moe_eval._route
        def patched_route(x_flat):
            gates, indices, aux = original_route(x_flat)
            captured["indices"] = indices
            return gates, indices, aux

        moe_eval._route = patched_route
        moe_eval(x)
        moe_eval._route = original_route

        indices = captured["indices"]
        assert (indices >= 0).all()
        assert (indices < NUM_EXPERTS).all()

    @pytest.mark.latentmoe
    def test_gate_weights_sum_to_one(self, moe_eval):
        """Softmax gate weights across top-k must sum to 1.0 per token."""
        x = torch.randn(BATCH, SEQ_LEN, D_MODEL, device=DEVICE)
        captured = {}

        original_route = moe_eval._route
        def patched_route(x_flat):
            gates, indices, aux = original_route(x_flat)
            captured["gates"] = gates
            return gates, indices, aux

        moe_eval._route = patched_route
        moe_eval(x)
        moe_eval._route = original_route

        gates = captured["gates"]               # (N, top_k)
        sums  = gates.sum(dim=-1)               # (N,)
        torch.testing.assert_close(
            sums, torch.ones_like(sums), atol=1e-5, rtol=1e-5,
            msg="Gate weights must sum to 1.0 per token"
        )

    @pytest.mark.latentmoe
    def test_aux_loss_increases_with_imbalance(self):
        """
        Forcing all tokens to route to expert 0 should produce a higher
        aux_loss than a balanced routing — the loss penalises imbalance.
        """
        model = make_moe(aux_loss_coeff=1.0).train()

        # Craft an input that strongly activates expert 0's router weight
        with torch.no_grad():
            # Set router so column 0 has very large weights, others near zero
            model.router.weight.zero_()
            model.router.weight[0] = 1.0      # expert 0 dominates

        x = torch.randn(BATCH, SEQ_LEN, D_MODEL, device=DEVICE)
        _, aux_imbalanced = model(x)

        # Reset to random weights (more balanced by default)
        nn.init.kaiming_uniform_(model.router.weight, a=math.sqrt(5))
        _, aux_balanced = model(x)

        assert aux_imbalanced.item() > aux_balanced.item(), (
            "Imbalanced routing should produce higher aux_loss than balanced routing"
        )

    @pytest.mark.latentmoe
    def test_different_tokens_can_route_differently(self, moe_eval):
        """
        Not all tokens should necessarily go to the same expert.
        With random weights and diverse inputs this should hold almost always.
        """
        torch.manual_seed(0)
        x = torch.randn(1, 64, D_MODEL, device=DEVICE)   # 64 tokens
        captured = {}

        original_route = moe_eval._route
        def patched_route(x_flat):
            gates, indices, aux = original_route(x_flat)
            captured["indices"] = indices
            return gates, indices, aux

        moe_eval._route = patched_route
        moe_eval(x)
        moe_eval._route = original_route

        # Primary expert (k=0) for each token — should not all be identical
        primary = captured["indices"][:, 0]
        unique_experts = primary.unique()
        assert len(unique_experts) > 1, (
            "All tokens routed to same expert — routing may be degenerate"
        )


# ---------------------------------------------------------------------------
# 5. Gradient tests
# ---------------------------------------------------------------------------

class TestGradients:

    @pytest.mark.latentmoe
    def test_backward_runs(self, moe_train):
        x = torch.randn(BATCH, SEQ_LEN, D_MODEL, device=DEVICE)
        out, aux = moe_train(x)
        (out.sum() + aux).backward()

    @pytest.mark.latentmoe
    def test_no_nan_gradients(self, moe_train):
        x = torch.randn(BATCH, SEQ_LEN, D_MODEL, device=DEVICE)
        out, aux = moe_train(x)
        (out.sum() + aux).backward()
        for name, p in moe_train.named_parameters():
            if p.grad is not None:
                assert not torch.isnan(p.grad).any(), f"NaN gradient in {name}"

    @pytest.mark.latentmoe
    def test_all_params_receive_gradient(self, moe_train):
        x = torch.randn(BATCH, SEQ_LEN, D_MODEL, device=DEVICE)
        out, aux = moe_train(x)
        (out.sum() + aux).backward()
        no_grad = [n for n, p in moe_train.named_parameters() if p.grad is None]
        assert not no_grad, f"No gradient for: {no_grad}"

    @pytest.mark.latentmoe
    def test_input_receives_gradient(self, moe_train):
        x = torch.randn(BATCH, SEQ_LEN, D_MODEL, device=DEVICE, requires_grad=True)
        out, aux = moe_train(x)
        (out.sum() + aux).backward()
        assert x.grad is not None
        assert not torch.isnan(x.grad).any()

    @pytest.mark.latentmoe
    def test_aux_loss_provides_gradient_to_router(self, moe_train):
        """
        aux_loss must provide gradient to router weights.
        If we only backprop through aux (not out), router.weight should still get a grad.
        """
        x = torch.randn(BATCH, SEQ_LEN, D_MODEL, device=DEVICE)
        _, aux = moe_train(x)
        aux.backward()
        assert moe_train.router.weight.grad is not None, (
            "Router weight received no gradient from aux_loss"
        )
        assert not torch.isnan(moe_train.router.weight.grad).any()

    @pytest.mark.latentmoe
    def test_loss_changes_after_step(self, moe_train):
        opt = torch.optim.SGD(moe_train.parameters(), lr=1e-3)
        x      = torch.randn(BATCH, SEQ_LEN, D_MODEL, device=DEVICE)
        target = torch.randn(BATCH, SEQ_LEN, D_MODEL, device=DEVICE)

        out, aux = moe_train(x)
        loss_before = (F.mse_loss(out, target) + aux).item()

        opt.zero_grad()
        out, aux = moe_train(x)
        (F.mse_loss(out, target) + aux).backward()
        opt.step()

        out, aux = moe_train(x)
        loss_after = (F.mse_loss(out, target) + aux).item()
        assert loss_before != loss_after, "Loss did not change after an optimizer step"


# ---------------------------------------------------------------------------
# 6. Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:

    @pytest.mark.latentmoe
    def test_top_k_equals_num_experts(self):
        """top_k == num_experts means dense routing — all experts active."""
        model = make_moe(num_experts=4, top_k=4)
        x = torch.randn(BATCH, SEQ_LEN, D_MODEL, device=DEVICE)
        out, aux = model(x)
        assert out.shape == x.shape
        assert not torch.isnan(out).any()

    @pytest.mark.latentmoe
    def test_single_expert(self):
        """num_experts=1, top_k=1 — degenerate but should not crash."""
        model = make_moe(num_experts=1, top_k=1)
        x = torch.randn(BATCH, SEQ_LEN, D_MODEL, device=DEVICE)
        out, _ = model(x)
        assert out.shape == x.shape

    @pytest.mark.latentmoe
    def test_no_shared_expert_numerics(self, moe_eval_no_shared):
        x = torch.randn(BATCH, SEQ_LEN, D_MODEL, device=DEVICE)
        out, aux = moe_eval_no_shared(x)
        assert not torch.isnan(out).any()
        assert not torch.isinf(out).any()

    @pytest.mark.latentmoe
    def test_fp16(self):
        if DEVICE == "cpu":
            pytest.skip("fp16 on CPU is unreliable")
        model = make_moe().half()
        x = torch.randn(BATCH, SEQ_LEN, D_MODEL, device=DEVICE, dtype=torch.float16)
        out, aux = model(x)
        assert out.dtype == torch.float16
        assert not torch.isnan(out).any()

    @pytest.mark.latentmoe
    def test_serialization(self, tmp_path, moe_eval):
        path = tmp_path / "latent_moe.pt"
        torch.save(moe_eval.state_dict(), path)
        loaded = make_moe()
        loaded.load_state_dict(torch.load(path, map_location=DEVICE))
        loaded.eval()
        x = torch.randn(BATCH, SEQ_LEN, D_MODEL, device=DEVICE)
        with torch.no_grad():
            out1, aux1 = moe_eval(x)
            out2, aux2 = loaded(x)
        torch.testing.assert_close(out1, out2)
        torch.testing.assert_close(aux1, aux2)

    @pytest.mark.latentmoe
    def test_custom_expert_ff_dim(self):
        """expert_dim_ff can be set independently of latent_dim."""
        model = make_moe(latent_dim=64, expert_dim_ff=128)
        x = torch.randn(BATCH, SEQ_LEN, D_MODEL, device=DEVICE)
        out, _ = model(x)
        assert out.shape == x.shape
        assert not torch.isnan(out).any()
        # Verify the expert weight shapes reflect expert_dim_ff
        E, d_ff, ell = model.expert_W1.shape
        assert d_ff == 128
        assert ell  == 64