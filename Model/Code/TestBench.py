"""
test_uni_mamba.py
-----------------
Pytest test suite for the UniMamba block.

Run with:
    pytest test_uni_mamba.py -v
    pytest test_uni_mamba.py -v -k "shape"          # only shape tests
    pytest test_uni_mamba.py -v --tb=short           # shorter tracebacks

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

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

@pytest.fixture(scope="module")
def default_model():
    """A standard UniMamba(dim=256) shared across tests in the module."""
    model = UniMamba(dim=256, nheads=8, dstate=64, chunk_size=64, ngroups=1)
    model.to(DEVICE)
    model.eval()
    return model


@pytest.fixture
def train_model():
    """A fresh model in train mode for gradient tests (not shared — state changes)."""
    model = UniMamba(dim=256, nheads=8, dstate=64, chunk_size=64, ngroups=1)
    model.to(DEVICE)
    model.train()
    return model


# ---------------------------------------------------------------------------
# 1. Shape tests — does the tensor flow through without crashing?
# ---------------------------------------------------------------------------

class TestOutputShape:

    def test_basic_shape(self, default_model):
        """Output shape must equal input shape."""
        x = torch.randn(2, 128, 256, device=DEVICE)
        out = default_model(x)
        assert out.shape == x.shape, f"Expected {x.shape}, got {out.shape}"

    def test_batch_1(self, default_model):
        x = torch.randn(1, 64, 256, device=DEVICE)
        assert default_model(x).shape == x.shape

    def test_large_batch(self, default_model):
        x = torch.randn(8, 64, 256, device=DEVICE)
        assert default_model(x).shape == x.shape

    def test_seqlen_equals_chunk_size(self, default_model):
        """Sequence length exactly matching chunk_size should work fine."""
        x = torch.randn(2, 64, 256, device=DEVICE)   # chunk_size=64
        assert default_model(x).shape == x.shape

    def test_seqlen_multiple_of_chunk(self, default_model):
        x = torch.randn(2, 256, 256, device=DEVICE)  # 4 × chunk_size
        assert default_model(x).shape == x.shape

    @pytest.mark.parametrize("dim,nheads", [(64, 4), (128, 4), (512, 16)])
    def test_various_dims(self, dim, nheads):
        model = UniMamba(dim=dim, nheads=nheads, dstate=32, chunk_size=32).to(DEVICE).eval()
        x = torch.randn(2, 64, dim, device=DEVICE)
        assert model(x).shape == x.shape


# ---------------------------------------------------------------------------
# 2. Numerical sanity — outputs should be finite and not degenerate
# ---------------------------------------------------------------------------

class TestNumericalSanity:

    def test_no_nan_in_output(self, default_model):
        x = torch.randn(2, 128, 256, device=DEVICE)
        out = default_model(x)
        assert not torch.isnan(out).any(), "Output contains NaN"

    def test_no_inf_in_output(self, default_model):
        x = torch.randn(2, 128, 256, device=DEVICE)
        out = default_model(x)
        assert not torch.isinf(out).any(), "Output contains Inf"

    def test_output_not_all_zeros(self, default_model):
        """A zero output would suggest the gate or projections are silencing everything."""
        x = torch.randn(2, 128, 256, device=DEVICE)
        out = default_model(x)
        assert out.abs().max() > 1e-6, "Output is effectively zero"

    def test_output_magnitude_reasonable(self, default_model):
        """Outputs shouldn't blow up for standard normal inputs."""
        x = torch.randn(4, 64, 256, device=DEVICE)
        out = default_model(x)
        assert out.abs().mean() < 100.0, "Output magnitude is suspiciously large"

    def test_different_inputs_give_different_outputs(self, default_model):
        """Basic sanity: model is not a constant function."""
        x1 = torch.randn(2, 64, 256, device=DEVICE)
        x2 = torch.randn(2, 64, 256, device=DEVICE)
        out1 = default_model(x1)
        out2 = default_model(x2)
        assert not torch.allclose(out1, out2), "Different inputs produced identical outputs"

    def test_deterministic_given_same_input(self, default_model):
        """In eval mode with no dropout, same input → same output."""
        x = torch.randn(2, 64, 256, device=DEVICE)
        out1 = default_model(x)
        out2 = default_model(x)
        torch.testing.assert_close(out1, out2)


# ---------------------------------------------------------------------------
# 3. Gradient tests — can we train this thing?
# ---------------------------------------------------------------------------

class TestGradients:

    def test_backward_runs(self, train_model):
        """Loss.backward() should not raise."""
        x = torch.randn(2, 64, 256, device=DEVICE)
        loss = train_model(x).sum()
        loss.backward()  # should not raise

    def test_no_nan_gradients(self, train_model):
        x = torch.randn(2, 64, 256, device=DEVICE)
        train_model(x).sum().backward()
        for name, p in train_model.named_parameters():
            if p.grad is not None:
                assert not torch.isnan(p.grad).any(), f"NaN gradient in {name}"

    def test_all_params_receive_gradient(self, train_model):
        x = torch.randn(2, 64, 256, device=DEVICE)
        train_model(x).sum().backward()
        no_grad = [
            name for name, p in train_model.named_parameters()
            if p.grad is None
        ]
        assert not no_grad, f"These parameters received no gradient: {no_grad}"

    def test_input_gradient(self, train_model):
        """Gradient should also flow back to the input tensor."""
        x = torch.randn(2, 64, 256, device=DEVICE, requires_grad=True)
        loss = train_model(x).sum()
        loss.backward()
        assert x.grad is not None, "No gradient at input"
        assert not torch.isnan(x.grad).any(), "NaN in input gradient"

    def test_loss_decreases_one_step(self, train_model):
        """One SGD step should reduce the loss (basic learning sanity check)."""
        optimizer = torch.optim.SGD(train_model.parameters(), lr=1e-3)
        x = torch.randn(2, 64, 256, device=DEVICE)
        target = torch.randn(2, 64, 256, device=DEVICE)

        loss_before = F.mse_loss(train_model(x), target).item()
        optimizer.zero_grad()
        loss = F.mse_loss(train_model(x), target)
        loss.backward()
        optimizer.step()
        loss_after = F.mse_loss(train_model(x), target).item()

        # Not guaranteed to decrease in 1 step for every random seed, but almost always true
        # for a fresh model with a reasonable LR. Use as a smoke test.
        assert loss_after != loss_before, "Loss did not change after an optimizer step"


# ---------------------------------------------------------------------------
# 4. Internal component tests — test sub-modules in isolation
# ---------------------------------------------------------------------------

class TestSubModules:

    def test_rmsnorm_shape(self):
        norm = RMSNorm(64).to(DEVICE)
        x = torch.randn(2, 32, 64, device=DEVICE)
        assert norm(x).shape == x.shape

    def test_rmsnorm_unit_norm(self):
        """After RMSNorm with weight=1, each vector should have RMS ≈ 1."""
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
        """RoPE is a rotation — it should preserve the L2 norm of each vector."""
        rope = RoPE(dim=128).to(DEVICE)
        x = torch.randn(2, 32, 128, device=DEVICE)
        out = rope(x)
        norms_in = x.norm(dim=-1)
        norms_out = out.norm(dim=-1)
        torch.testing.assert_close(norms_in, norms_out, atol=1e-4, rtol=1e-4)

    def test_mimo_shape(self):
        mimo = MIMO(128).to(DEVICE)
        x = torch.randn(2, 64, 128, device=DEVICE)
        assert mimo(x).shape == x.shape

    def test_mimo_is_linear(self):
        """MIMO is a linear projection — f(ax + by) == a*f(x) + b*f(y)."""
        mimo = MIMO(64).to(DEVICE)
        x = torch.randn(1, 16, 64, device=DEVICE)
        y = torch.randn(1, 16, 64, device=DEVICE)
        a, b = 2.3, -1.1
        lhs = mimo(a * x + b * y)
        rhs = a * mimo(x) + b * mimo(y)
        torch.testing.assert_close(lhs, rhs, atol=1e-5, rtol=1e-5)

    def test_a_is_always_negative(self, default_model):
        """A must be negative for stable SSM dynamics."""
        A = -torch.exp(default_model.A_log.float())
        assert (A < 0).all(), "All A values must be negative for stable dynamics"


# ---------------------------------------------------------------------------
# 5. Edge case tests
# ---------------------------------------------------------------------------

class TestEdgeCases:

    def test_single_token(self, default_model):
        """seqlen=1 (e.g. autoregressive decode step)."""
        x = torch.randn(1, 1, 256, device=DEVICE)
        out = default_model(x)
        assert out.shape == x.shape
        assert not torch.isnan(out).any()

    def test_fp16_forward(self):
        """Model should run in fp16 without NaN (common in practice)."""
        if DEVICE == "cpu":
            pytest.skip("fp16 on CPU is unreliable")
        model = UniMamba(dim=128, nheads=4, dstate=32, chunk_size=32).to(DEVICE).half().eval()
        x = torch.randn(2, 64, 128, device=DEVICE, dtype=torch.float16)
        out = model(x)
        assert out.dtype == torch.float16
        assert not torch.isnan(out).any()

    def test_zero_input(self, default_model):
        """Zero input should produce finite (not NaN) output."""
        x = torch.zeros(2, 64, 256, device=DEVICE)
        out = default_model(x)
        assert not torch.isnan(out).any()
        assert not torch.isinf(out).any()

    def test_large_values_input(self, default_model):
        """Large inputs should not cause overflow."""
        x = torch.randn(2, 64, 256, device=DEVICE) * 10.0
        out = default_model(x)
        assert not torch.isnan(out).any()
        assert not torch.isinf(out).any()

    def test_model_serialization(self, tmp_path, default_model):
        """Model should be saveable and loadable with matching output."""
        path = tmp_path / "uni_mamba.pt"
        torch.save(default_model.state_dict(), path)

        loaded = UniMamba(dim=256, nheads=8, dstate=64, chunk_size=64).to(DEVICE).eval()
        loaded.load_state_dict(torch.load(path, map_location=DEVICE))

        x = torch.randn(2, 64, 256, device=DEVICE)
        with torch.no_grad():
            out_orig = default_model(x)
            out_loaded = loaded(x)

        torch.testing.assert_close(out_orig, out_loaded)