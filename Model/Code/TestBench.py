"""
TestBench.py
-------------
Pytest test suite for UniMamba and BiMamba blocks.

Choose which model to test at runtime via --model flag:

    pytest test_mamba.py -v --model unimamba
    pytest test_mamba.py -v --model bimamba
    pytest test_mamba.py -v --model both        # runs all tests for both (default)

Filter by test class:
    pytest test_mamba.py -v --model bimamba -k "shape"
    pytest test_mamba.py -v --model both --tb=short

Requirements:
    pip install pytest torch einops mamba-ssm
"""

import math
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from UniMamba import UniMamba, RMSNorm, RoPE, MIMO
from BiMamba import BiMamba


# ---------------------------------------------------------------------------
# Shared config
# ---------------------------------------------------------------------------

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Canonical dims used across all tests — same for both models so results
# are directly comparable
DIM       = 256
NHEADS    = 4
HEADDIM   = 64    # DIM == NHEADS * HEADDIM
DSTATE    = 32
CHUNK     = 64
NGROUPS   = 1

assert DIM == NHEADS * HEADDIM


def make_unimamba(**kwargs) -> UniMamba:
    cfg = dict(dim=DIM, nheads=NHEADS, dstate=DSTATE, chunk_size=CHUNK, ngroups=NGROUPS)
    cfg.update(kwargs)
    return UniMamba(**cfg).to(DEVICE)


def make_bimamba(**kwargs) -> BiMamba:
    cfg = dict(d_model=DIM, nheads=NHEADS, headdim=HEADDIM,
               dstate=DSTATE, chunk_size=CHUNK, ngroups=NGROUPS)
    cfg.update(kwargs)
    return BiMamba(**cfg).to(DEVICE)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def uni_eval() -> UniMamba:
    return make_unimamba()


@pytest.fixture(scope="module")
def bi_eval() -> BiMamba:
    return make_bimamba()


@pytest.fixture
def uni_train() -> UniMamba:
    return make_unimamba().train()


@pytest.fixture
def bi_train() -> BiMamba:
    return make_bimamba().train()


# ---------------------------------------------------------------------------
# 1. Shape tests
# ---------------------------------------------------------------------------

class TestOutputShape:

    @pytest.mark.unimamba
    def test_basic_shape_uni(self, uni_eval):
        x = torch.randn(2, 128, DIM, device=DEVICE)
        assert uni_eval(x).shape == x.shape

    @pytest.mark.bimamba
    def test_basic_shape_bi(self, bi_eval):
        x = torch.randn(2, 128, DIM, device=DEVICE)
        assert bi_eval(x).shape == x.shape

    @pytest.mark.unimamba
    def test_batch_1_uni(self, uni_eval):
        x = torch.randn(1, 64, DIM, device=DEVICE)
        assert uni_eval(x).shape == x.shape

    @pytest.mark.bimamba
    def test_batch_1_bi(self, bi_eval):
        x = torch.randn(1, 64, DIM, device=DEVICE)
        assert bi_eval(x).shape == x.shape

    @pytest.mark.unimamba
    def test_large_batch_uni(self, uni_eval):
        x = torch.randn(8, 64, DIM, device=DEVICE)
        assert uni_eval(x).shape == x.shape

    @pytest.mark.bimamba
    def test_large_batch_bi(self, bi_eval):
        x = torch.randn(8, 64, DIM, device=DEVICE)
        assert bi_eval(x).shape == x.shape

    @pytest.mark.unimamba
    def test_seqlen_equals_chunk_uni(self, uni_eval):
        x = torch.randn(2, CHUNK, DIM, device=DEVICE)
        assert uni_eval(x).shape == x.shape

    @pytest.mark.bimamba
    def test_seqlen_equals_chunk_bi(self, bi_eval):
        x = torch.randn(2, CHUNK, DIM, device=DEVICE)
        assert bi_eval(x).shape == x.shape

    @pytest.mark.unimamba
    def test_seqlen_multiple_of_chunk_uni(self, uni_eval):
        x = torch.randn(2, CHUNK * 4, DIM, device=DEVICE)
        assert uni_eval(x).shape == x.shape

    @pytest.mark.bimamba
    def test_seqlen_multiple_of_chunk_bi(self, bi_eval):
        x = torch.randn(2, CHUNK * 4, DIM, device=DEVICE)
        assert bi_eval(x).shape == x.shape

    @pytest.mark.unimamba
    @pytest.mark.parametrize("dim,nheads,headdim", [(64, 4, 16), (128, 4, 32), (512, 8, 64)])
    def test_various_dims_uni(self, dim, nheads, headdim):
        model = UniMamba(dim=dim, nheads=nheads, dstate=32, chunk_size=32).to(DEVICE).eval()
        x = torch.randn(2, 64, dim, device=DEVICE)
        assert model(x).shape == x.shape

    @pytest.mark.bimamba
    @pytest.mark.parametrize("dim,nheads,headdim", [(64, 4, 16), (128, 4, 32), (512, 8, 64)])
    def test_various_dims_bi(self, dim, nheads, headdim):
        model = BiMamba(d_model=dim, nheads=nheads, headdim=headdim, dstate=32, chunk_size=32).to(DEVICE).eval()
        x = torch.randn(2, 64, dim, device=DEVICE)
        assert model(x).shape == x.shape


# ---------------------------------------------------------------------------
# 2. Numerical sanity
# ---------------------------------------------------------------------------

class TestNumericalSanity:

    @pytest.mark.unimamba
    def test_no_nan_uni(self, uni_eval):
        x = torch.randn(2, 128, DIM, device=DEVICE)
        assert not torch.isnan(uni_eval(x)).any()

    @pytest.mark.bimamba
    def test_no_nan_bi(self, bi_eval):
        x = torch.randn(2, 128, DIM, device=DEVICE)
        assert not torch.isnan(bi_eval(x)).any()

    @pytest.mark.unimamba
    def test_no_inf_uni(self, uni_eval):
        x = torch.randn(2, 128, DIM, device=DEVICE)
        assert not torch.isinf(uni_eval(x)).any()

    @pytest.mark.bimamba
    def test_no_inf_bi(self, bi_eval):
        x = torch.randn(2, 128, DIM, device=DEVICE)
        assert not torch.isinf(bi_eval(x)).any()

    @pytest.mark.unimamba
    def test_not_all_zeros_uni(self, uni_eval):
        x = torch.randn(2, 128, DIM, device=DEVICE)
        assert uni_eval(x).abs().max() > 1e-6

    @pytest.mark.bimamba
    def test_not_all_zeros_bi(self, bi_eval):
        x = torch.randn(2, 128, DIM, device=DEVICE)
        assert bi_eval(x).abs().max() > 1e-6

    @pytest.mark.unimamba
    def test_magnitude_reasonable_uni(self, uni_eval):
        x = torch.randn(4, 64, DIM, device=DEVICE)
        assert uni_eval(x).abs().mean() < 100.0

    @pytest.mark.bimamba
    def test_magnitude_reasonable_bi(self, bi_eval):
        x = torch.randn(4, 64, DIM, device=DEVICE)
        assert bi_eval(x).abs().mean() < 100.0

    @pytest.mark.unimamba
    def test_different_inputs_uni(self, uni_eval):
        x1 = torch.randn(2, 64, DIM, device=DEVICE)
        x2 = torch.randn(2, 64, DIM, device=DEVICE)
        assert not torch.allclose(uni_eval(x1), uni_eval(x2))

    @pytest.mark.bimamba
    def test_different_inputs_bi(self, bi_eval):
        x1 = torch.randn(2, 64, DIM, device=DEVICE)
        x2 = torch.randn(2, 64, DIM, device=DEVICE)
        assert not torch.allclose(bi_eval(x1), bi_eval(x2))

    @pytest.mark.unimamba
    def test_deterministic_uni(self, uni_eval):
        x = torch.randn(2, 64, DIM, device=DEVICE)
        torch.testing.assert_close(uni_eval(x), uni_eval(x))

    @pytest.mark.bimamba
    def test_deterministic_bi(self, bi_eval):
        x = torch.randn(2, 64, DIM, device=DEVICE)
        torch.testing.assert_close(bi_eval(x), bi_eval(x))

    @pytest.mark.bimamba
    def test_bidirectional_not_causal(self, bi_eval):
        """
        BiMamba-specific: output at position 0 must depend on tokens at later
        positions. We verify this by checking that changing token at position L-1
        changes the output at position 0.
        If BiMamba were purely causal (like UniMamba) this test would fail.
        """
        torch.manual_seed(42)
        x1 = torch.randn(1, 32, DIM, device=DEVICE)
        x2 = x1.clone()
        x2[:, -1, :] += 1.0     # perturb only the last token

        out1 = bi_eval(x1)
        out2 = bi_eval(x2)

        # Position 0 output should differ — future token influenced it
        assert not torch.allclose(out1[:, 0, :], out2[:, 0, :]), (
            "Output at position 0 did not change when last token was perturbed — "
            "BiMamba may not be truly bidirectional"
        )

    @pytest.mark.unimamba
    def test_causal_uni(self, uni_eval):
        """
        UniMamba-specific: output at position 0 must NOT depend on later tokens.
        Changing the last token should leave position 0 unchanged.
        """
        torch.manual_seed(42)
        x1 = torch.randn(1, 32, DIM, device=DEVICE)
        x2 = x1.clone()
        x2[:, -1, :] += 1.0

        out1 = uni_eval(x1)
        out2 = uni_eval(x2)

        torch.testing.assert_close(
            out1[:, 0, :], out2[:, 0, :],
            msg="UniMamba is not causal — position 0 changed when last token was perturbed"
        )


# ---------------------------------------------------------------------------
# 3. Gradient tests
# ---------------------------------------------------------------------------

class TestGradients:

    @pytest.mark.unimamba
    def test_backward_runs_uni(self, uni_train):
        x = torch.randn(2, 64, DIM, device=DEVICE)
        uni_train(x).sum().backward()

    @pytest.mark.bimamba
    def test_backward_runs_bi(self, bi_train):
        x = torch.randn(2, 64, DIM, device=DEVICE)
        bi_train(x).sum().backward()

    @pytest.mark.unimamba
    def test_no_nan_grads_uni(self, uni_train):
        torch.randn(2, 64, DIM, device=DEVICE).requires_grad_(False)
        x = torch.randn(2, 64, DIM, device=DEVICE)
        uni_train(x).sum().backward()
        for name, p in uni_train.named_parameters():
            if p.grad is not None:
                assert not torch.isnan(p.grad).any(), f"NaN grad in {name}"

    @pytest.mark.bimamba
    def test_no_nan_grads_bi(self, bi_train):
        x = torch.randn(2, 64, DIM, device=DEVICE)
        bi_train(x).sum().backward()
        for name, p in bi_train.named_parameters():
            if p.grad is not None:
                assert not torch.isnan(p.grad).any(), f"NaN grad in {name}"

    @pytest.mark.unimamba
    def test_all_params_get_grad_uni(self, uni_train):
        x = torch.randn(2, 64, DIM, device=DEVICE)
        uni_train(x).sum().backward()
        no_grad = [n for n, p in uni_train.named_parameters() if p.grad is None]
        assert not no_grad, f"No gradient for: {no_grad}"

    @pytest.mark.bimamba
    def test_all_params_get_grad_bi(self, bi_train):
        x = torch.randn(2, 64, DIM, device=DEVICE)
        bi_train(x).sum().backward()
        no_grad = [n for n, p in bi_train.named_parameters() if p.grad is None]
        assert not no_grad, f"No gradient for: {no_grad}"

    @pytest.mark.unimamba
    def test_input_grad_uni(self, uni_train):
        x = torch.randn(2, 64, DIM, device=DEVICE, requires_grad=True)
        uni_train(x).sum().backward()
        assert x.grad is not None and not torch.isnan(x.grad).any()

    @pytest.mark.bimamba
    def test_input_grad_bi(self, bi_train):
        x = torch.randn(2, 64, DIM, device=DEVICE, requires_grad=True)
        bi_train(x).sum().backward()
        assert x.grad is not None and not torch.isnan(x.grad).any()

    @pytest.mark.unimamba
    def test_loss_changes_after_step_uni(self, uni_train):
        opt = torch.optim.SGD(uni_train.parameters(), lr=1e-3)
        x = torch.randn(2, 64, DIM, device=DEVICE)
        target = torch.randn(2, 64, DIM, device=DEVICE)
        loss_before = F.mse_loss(uni_train(x), target).item()
        opt.zero_grad()
        F.mse_loss(uni_train(x), target).backward()
        opt.step()
        loss_after = F.mse_loss(uni_train(x), target).item()
        assert loss_before != loss_after

    @pytest.mark.bimamba
    def test_loss_changes_after_step_bi(self, bi_train):
        opt = torch.optim.SGD(bi_train.parameters(), lr=1e-3)
        x = torch.randn(2, 64, DIM, device=DEVICE)
        target = torch.randn(2, 64, DIM, device=DEVICE)
        loss_before = F.mse_loss(bi_train(x), target).item()
        opt.zero_grad()
        F.mse_loss(bi_train(x), target).backward()
        opt.step()
        loss_after = F.mse_loss(bi_train(x), target).item()
        assert loss_before != loss_after


# ---------------------------------------------------------------------------
# 4. Internal component tests  (shared — same primitives in both models)
# ---------------------------------------------------------------------------

class TestSubModules:

    def test_rmsnorm_shape(self):
        norm = RMSNorm(64).to(DEVICE)
        x = torch.randn(2, 32, 64, device=DEVICE)
        assert norm(x).shape == x.shape

    def test_rmsnorm_unit_norm(self):
        norm = RMSNorm(64).to(DEVICE)
        nn.init.ones_(norm.weight)
        x = torch.randn(4, 16, 64, device=DEVICE)
        out = norm(x)
        rms = out.pow(2).mean(-1).sqrt()
        torch.testing.assert_close(rms, torch.ones_like(rms), atol=1e-4, rtol=1e-4)

    def test_rope_shape(self):
        rope = RoPE(dim=128).to(DEVICE)
        x = torch.randn(2, 64, 128, device=DEVICE)
        assert rope(x).shape == x.shape

    def test_rope_preserves_norm(self):
        rope = RoPE(dim=128).to(DEVICE)
        x = torch.randn(2, 32, 128, device=DEVICE)
        torch.testing.assert_close(
            x.norm(dim=-1), rope(x).norm(dim=-1), atol=1e-4, rtol=1e-4
        )

    def test_mimo_shape(self):
        mimo = MIMO(128).to(DEVICE)
        x = torch.randn(2, 64, 128, device=DEVICE)
        assert mimo(x).shape == x.shape

    def test_mimo_is_linear(self):
        mimo = MIMO(64).to(DEVICE)
        x = torch.randn(1, 16, 64, device=DEVICE)
        y = torch.randn(1, 16, 64, device=DEVICE)
        a, b = 2.3, -1.1
        torch.testing.assert_close(
            mimo(a * x + b * y), a * mimo(x) + b * mimo(y), atol=1e-5, rtol=1e-5
        )

    @pytest.mark.unimamba
    def test_a_negative_uni(self, uni_eval):
        A = -torch.exp(uni_eval.A_log.float())
        assert (A < 0).all(), "All A values must be negative for stable SSM dynamics"

    @pytest.mark.bimamba
    def test_a_negative_bi(self, bi_eval):
        A = -torch.exp(bi_eval.A_log.float())
        assert (A < 0).all(), "All A values must be negative for stable SSM dynamics"

    @pytest.mark.bimamba
    def test_dt_bias_is_standalone_param_bi(self, bi_eval):
        """
        BiMamba must store dt_bias as a standalone nn.Parameter, not inside
        dt_proj.bias — matching UniMamba's convention and the kernel's expectations.
        """
        assert hasattr(bi_eval, "dt_bias"), "BiMamba must have a standalone dt_bias parameter"
        assert isinstance(bi_eval.dt_bias, nn.Parameter), "dt_bias must be nn.Parameter"
        assert bi_eval.dt_proj.bias is None, (
            "dt_proj must have bias=False — dt_bias should be standalone"
        )

    @pytest.mark.bimamba
    def test_bimamba_has_all_unimamba_branches(self, bi_eval):
        """
        Verify BiMamba has every sub-module that UniMamba has in its SSM branch
        and gate branch (in_mimo, ssm_norm, rope, out_mimo, gate_mimo).
        """
        required = ["in_proj", "in_mimo", "ssm_norm", "rope",
                    "dt_proj", "dt_bias", "A_log", "D",
                    "out_mimo", "out_proj",
                    "gate_proj", "gate_mimo"]
        missing = [attr for attr in required if not hasattr(bi_eval, attr)]
        assert not missing, f"BiMamba is missing these attributes: {missing}"


# ---------------------------------------------------------------------------
# 5. Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:

    @pytest.mark.unimamba
    def test_single_token_uni(self, uni_eval):
        x = torch.randn(1, 1, DIM, device=DEVICE)
        out = uni_eval(x)
        assert out.shape == x.shape and not torch.isnan(out).any()

    @pytest.mark.bimamba
    def test_single_token_bi(self, bi_eval):
        x = torch.randn(1, 1, DIM, device=DEVICE)
        out = bi_eval(x)
        assert out.shape == x.shape and not torch.isnan(out).any()

    @pytest.mark.unimamba
    def test_fp16_uni(self):
        if DEVICE == "cpu":
            pytest.skip("fp16 on CPU is unreliable")
        model = make_unimamba().half().eval()
        x = torch.randn(2, 64, DIM, device=DEVICE, dtype=torch.float16)
        out = model(x)
        assert out.dtype == torch.float16 and not torch.isnan(out).any()

    @pytest.mark.bimamba
    def test_fp16_bi(self):
        if DEVICE == "cpu":
            pytest.skip("fp16 on CPU is unreliable")
        model = make_bimamba().half().eval()
        x = torch.randn(2, 64, DIM, device=DEVICE, dtype=torch.float16)
        out = model(x)
        assert out.dtype == torch.float16 and not torch.isnan(out).any()

    @pytest.mark.unimamba
    def test_zero_input_uni(self, uni_eval):
        x = torch.zeros(2, 64, DIM, device=DEVICE)
        out = uni_eval(x)
        assert not torch.isnan(out).any() and not torch.isinf(out).any()

    @pytest.mark.bimamba
    def test_zero_input_bi(self, bi_eval):
        x = torch.zeros(2, 64, DIM, device=DEVICE)
        out = bi_eval(x)
        assert not torch.isnan(out).any() and not torch.isinf(out).any()

    @pytest.mark.unimamba
    def test_large_values_uni(self, uni_eval):
        x = torch.randn(2, 64, DIM, device=DEVICE) * 10.0
        out = uni_eval(x)
        assert not torch.isnan(out).any() and not torch.isinf(out).any()

    @pytest.mark.bimamba
    def test_large_values_bi(self, bi_eval):
        x = torch.randn(2, 64, DIM, device=DEVICE) * 10.0
        out = bi_eval(x)
        assert not torch.isnan(out).any() and not torch.isinf(out).any()

    @pytest.mark.unimamba
    def test_serialization_uni(self, tmp_path, uni_eval):
        path = tmp_path / "uni_mamba.pt"
        torch.save(uni_eval.state_dict(), path)
        loaded = make_unimamba().eval()
        loaded.load_state_dict(torch.load(path, map_location=DEVICE))
        x = torch.randn(2, 64, DIM, device=DEVICE)
        with torch.no_grad():
            torch.testing.assert_close(uni_eval(x), loaded(x))

    @pytest.mark.bimamba
    def test_serialization_bi(self, tmp_path, bi_eval):
        path = tmp_path / "bi_mamba.pt"
        torch.save(bi_eval.state_dict(), path)
        loaded = make_bimamba().eval()
        loaded.load_state_dict(torch.load(path, map_location=DEVICE))
        x = torch.randn(2, 64, DIM, device=DEVICE)
        with torch.no_grad():
            torch.testing.assert_close(bi_eval(x), loaded(x))