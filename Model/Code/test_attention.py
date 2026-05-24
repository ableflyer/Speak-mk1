"""
test_attention.py
-----------------
Pytest test suite for LocalWindowAttention and CrossModelSparseAttention.

Run with:
    pytest test_attention.py -v
    pytest test_attention.py -v -k "LocalWindow"
    pytest test_attention.py -v -k "CrossModel"
    pytest test_attention.py -v -k "window"
    pytest test_attention.py -v -k "cross"
"""

import pytest
import torch
import torch.nn as nn

from local_window_attn import LocalWindowAttention
from cross_model_sparse_attn import CrossModelSparseAttention


DEVICE  = "cuda" if torch.cuda.is_available() else "cpu"
D_MODEL = 128
NHEADS  = 4
WINDOW  = 4
BATCH   = 2
SEQ_LEN = 32


def make_attn(**kwargs) -> LocalWindowAttention:
    cfg = dict(d_model=D_MODEL, nheads=NHEADS, window=WINDOW, dropout=0.0, causal=False)
    cfg.update(kwargs)
    return LocalWindowAttention(**cfg).to(DEVICE).eval()


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def attn_eval():
    return make_attn()


@pytest.fixture(scope="module")
def attn_causal():
    return make_attn(causal=True)


@pytest.fixture
def attn_train():
    return make_attn().train()


# ---------------------------------------------------------------------------
# 1. Shape tests
# ---------------------------------------------------------------------------

class TestShape:

    def test_output_shape(self, attn_eval):
        x = torch.randn(BATCH, SEQ_LEN, D_MODEL, device=DEVICE)
        assert attn_eval(x).shape == x.shape

    def test_batch_1(self, attn_eval):
        x = torch.randn(1, SEQ_LEN, D_MODEL, device=DEVICE)
        assert attn_eval(x).shape == x.shape

    def test_single_token(self, attn_eval):
        x = torch.randn(BATCH, 1, D_MODEL, device=DEVICE)
        assert attn_eval(x).shape == x.shape

    def test_seqlen_shorter_than_window(self):
        """Window larger than sequence length should still work."""
        attn = make_attn(window=64)
        x = torch.randn(BATCH, 8, D_MODEL, device=DEVICE)
        assert attn(x).shape == x.shape

    def test_various_nheads(self):
        for nheads in [1, 2, 4, 8]:
            attn = make_attn(d_model=128, nheads=nheads)
            x = torch.randn(BATCH, SEQ_LEN, D_MODEL, device=DEVICE)
            assert attn(x).shape == x.shape


# ---------------------------------------------------------------------------
# 2. Numerical sanity
# ---------------------------------------------------------------------------

class TestNumericalSanity:

    def test_no_nan(self, attn_eval):
        x = torch.randn(BATCH, SEQ_LEN, D_MODEL, device=DEVICE)
        assert not torch.isnan(attn_eval(x)).any()

    def test_no_inf(self, attn_eval):
        x = torch.randn(BATCH, SEQ_LEN, D_MODEL, device=DEVICE)
        assert not torch.isinf(attn_eval(x)).any()

    def test_not_all_zeros(self, attn_eval):
        x = torch.randn(BATCH, SEQ_LEN, D_MODEL, device=DEVICE)
        assert attn_eval(x).abs().max() > 1e-6

    def test_deterministic(self, attn_eval):
        x = torch.randn(BATCH, SEQ_LEN, D_MODEL, device=DEVICE)
        torch.testing.assert_close(attn_eval(x), attn_eval(x))

    def test_zero_input(self, attn_eval):
        x = torch.zeros(BATCH, SEQ_LEN, D_MODEL, device=DEVICE)
        out = attn_eval(x)
        assert not torch.isnan(out).any()
        assert not torch.isinf(out).any()

    def test_large_input(self, attn_eval):
        x = torch.randn(BATCH, SEQ_LEN, D_MODEL, device=DEVICE) * 10.0
        out = attn_eval(x)
        assert not torch.isnan(out).any()
        assert not torch.isinf(out).any()


# ---------------------------------------------------------------------------
# 3. THE KEY TESTS — window locality
#
# These are the tests that prove your implementation is actually doing
# local window attention and not just full attention.
# ---------------------------------------------------------------------------

class TestWindowLocality:

    def test_far_token_has_no_influence(self, attn_eval):
        """
        THE most important test.

        Token at position 0 has a window of WINDOW=4, so it can only attend
        to positions [0, 4]. Changing a token at position 20 (well outside
        the window) must NOT change the output at position 0.

        If this fails → your mask is wrong or not being applied.
        """
        torch.manual_seed(0)
        x1 = torch.randn(1, SEQ_LEN, D_MODEL, device=DEVICE)
        x2 = x1.clone()
        x2[:, 20, :] += 10.0       # large perturbation far outside window

        out1 = attn_eval(x1)
        out2 = attn_eval(x2)

        torch.testing.assert_close(
            out1[:, 0, :], out2[:, 0, :],
            atol=1e-5, rtol=1e-5,
            msg="Position 0 output changed when a token at position 20 was perturbed "
                "— window mask is not working"
        )

    def test_near_token_does_influence(self, attn_eval):
        """
        Complementary to the test above.

        A token within the window MUST influence the output.
        Changing position 2 (within window=4 of position 0) SHOULD
        change the output at position 0.

        If this fails → your attention weights are zero everywhere (dead attention).
        """
        torch.manual_seed(0)
        x1 = torch.randn(1, SEQ_LEN, D_MODEL, device=DEVICE)
        x2 = x1.clone()
        x2[:, 2, :] += 10.0        # within window of position 0

        out1 = attn_eval(x1)
        out2 = attn_eval(x2)

        assert not torch.allclose(out1[:, 0, :], out2[:, 0, :], atol=1e-5), (
            "Position 0 output did NOT change when position 2 (within window) "
            "was perturbed — attention weights may be dead"
        )

    def test_window_boundary_is_exact(self, attn_eval):
        """
        Test the exact boundary — position WINDOW should be included,
        position WINDOW+1 should be excluded, for query at position 0.
        """
        torch.manual_seed(1)
        x_base = torch.randn(1, SEQ_LEN, D_MODEL, device=DEVICE)

        # Perturb exactly at boundary (should influence)
        x_at_boundary = x_base.clone()
        x_at_boundary[:, WINDOW, :] += 10.0
        out_base     = attn_eval(x_base)
        out_boundary = attn_eval(x_at_boundary)
        assert not torch.allclose(out_base[:, 0, :], out_boundary[:, 0, :], atol=1e-5), (
            f"Position {WINDOW} (exactly at window boundary) did not influence "
            f"position 0 — boundary is off by one"
        )

        # Perturb just outside boundary (should NOT influence)
        x_outside = x_base.clone()
        x_outside[:, WINDOW + 1, :] += 10.0
        out_outside = attn_eval(x_outside)
        torch.testing.assert_close(
            out_base[:, 0, :], out_outside[:, 0, :],
            atol=1e-5, rtol=1e-5,
            msg=f"Position {WINDOW + 1} (just outside window) influenced position 0 "
                f"— boundary is off by one"
        )

    def test_window_is_symmetric_in_encoder(self, attn_eval):
        """
        In encoder mode (causal=False), the window is symmetric:
        position t can attend to both past AND future within the window.

        Verify: changing a FUTURE token within window changes current output.
        Test: perturb position 5 and check position 3 changes (5 is 2 ahead of 3,
        within window=4).
        """
        torch.manual_seed(2)
        x1 = torch.randn(1, SEQ_LEN, D_MODEL, device=DEVICE)
        x2 = x1.clone()
        x2[:, 5, :] += 10.0        # future token, within window of position 3

        out1 = attn_eval(x1)
        out2 = attn_eval(x2)

        assert not torch.allclose(out1[:, 3, :], out2[:, 3, :], atol=1e-5), (
            "Future token within window did not influence current position in "
            "encoder mode — window may be causal-only by mistake"
        )

    def test_causal_blocks_future_tokens(self, attn_causal):
        """
        In causal mode, future tokens must NEVER influence past positions,
        even if they are within the window.

        Perturb position 5 and verify position 3 does NOT change.
        (5 is within window=4 of position 3, but it's in the future.)
        """
        torch.manual_seed(3)
        x1 = torch.randn(1, SEQ_LEN, D_MODEL, device=DEVICE)
        x2 = x1.clone()
        x2[:, 5, :] += 10.0        # future token within window

        out1 = attn_causal(x1)
        out2 = attn_causal(x2)

        torch.testing.assert_close(
            out1[:, 3, :], out2[:, 3, :],
            atol=1e-5, rtol=1e-5,
            msg="Causal mode: future token within window influenced a past position "
                "— causal mask is not working"
        )

    def test_causal_allows_past_tokens(self, attn_causal):
        """
        In causal mode, past tokens within the window must still influence
        the current position.
        """
        torch.manual_seed(4)
        x1 = torch.randn(1, SEQ_LEN, D_MODEL, device=DEVICE)
        x2 = x1.clone()
        x2[:, 3, :] += 10.0        # past token, within window of position 5

        out1 = attn_causal(x1)
        out2 = attn_causal(x2)

        assert not torch.allclose(out1[:, 5, :], out2[:, 5, :], atol=1e-5), (
            "Causal mode: past token within window did not influence current position "
            "— past attention is dead"
        )

    def test_mask_shape_is_correct(self):
        """Directly inspect the window mask for correctness."""
        attn = make_attn(window=2)
        L = 7
        mask = attn._build_window_mask(L, torch.device("cpu"))  # True = blocked

        # mask[i, j] should be True when |i - j| > window
        for i in range(L):
            for j in range(L):
                expected_blocked = abs(i - j) > 2
                assert mask[i, j].item() == expected_blocked, (
                    f"mask[{i},{j}] = {mask[i,j].item()}, "
                    f"expected blocked={expected_blocked} (|{i}-{j}|={abs(i-j)}, window=2)"
                )

    def test_causal_mask_shape_is_correct(self):
        """Directly inspect the causal window mask for correctness."""
        attn = make_attn(window=2, causal=True)
        L = 7
        mask = attn._build_window_mask(L, torch.device("cpu"))  # True = blocked

        for i in range(L):
            for j in range(L):
                expected_blocked = (abs(i - j) > 2) or (j > i)
                assert mask[i, j].item() == expected_blocked, (
                    f"causal mask[{i},{j}] = {mask[i,j].item()}, "
                    f"expected blocked={expected_blocked}"
                )


# ---------------------------------------------------------------------------
# 4. Padding mask tests
# ---------------------------------------------------------------------------

class TestPaddingMask:

    def test_padded_positions_do_not_influence_output(self):
        """
        Padded positions (key_padding_mask=True) must not influence any output,
        regardless of whether they are within the window.

        Correct setup: both calls use the SAME padding mask so the only
        difference between them is the value at the padded position itself.
        The residual path carries the raw x, so we must keep the mask
        consistent — not compare a masked call to an unmasked call.

          Call 1: x,            pad_mask active  → out_original
          Call 2: x + perturb,  pad_mask active  → out_perturbed

        If the mask is working, position 2 is blocked as a key in both calls,
        so position 0's output must be identical despite the large perturbation.
        """
        torch.manual_seed(5)
        attn = make_attn()
        x = torch.randn(1, SEQ_LEN, D_MODEL, device=DEVICE)

        # Mark position 2 as padded — applied to BOTH calls
        pad_mask = torch.zeros(1, SEQ_LEN, dtype=torch.bool, device=DEVICE)
        pad_mask[:, 2] = True

        # Call 1: original x with padding mask
        out_original = attn(x, key_padding_mask=pad_mask)

        # Call 2: same mask, but perturb the padded position by a large amount
        x_perturbed = x.clone()
        x_perturbed[:, 2, :] += 100.0
        out_perturbed = attn(x_perturbed, key_padding_mask=pad_mask)

        torch.testing.assert_close(
            out_original[:, 0, :], out_perturbed[:, 0, :],
            atol=1e-4, rtol=1e-4,
            msg="Padded position within window still influenced the output — "
                "key_padding_mask is not being applied correctly"
        )

    def test_all_padded_produces_finite_output(self):
        """If all keys are masked, softmax produces NaN — nan_to_num should handle it."""
        attn = make_attn()
        x = torch.randn(BATCH, SEQ_LEN, D_MODEL, device=DEVICE)
        pad_mask = torch.ones(BATCH, SEQ_LEN, dtype=torch.bool, device=DEVICE)
        out = attn(x, key_padding_mask=pad_mask)
        assert not torch.isnan(out).any(), "All-padded input produced NaN"
        assert not torch.isinf(out).any()


# ---------------------------------------------------------------------------
# 5. Gradient tests
# ---------------------------------------------------------------------------

class TestGradients:

    def test_backward_runs(self, attn_train):
        x = torch.randn(BATCH, SEQ_LEN, D_MODEL, device=DEVICE)
        attn_train(x).sum().backward()

    def test_no_nan_gradients(self, attn_train):
        x = torch.randn(BATCH, SEQ_LEN, D_MODEL, device=DEVICE)
        attn_train(x).sum().backward()
        for name, p in attn_train.named_parameters():
            if p.grad is not None:
                assert not torch.isnan(p.grad).any(), f"NaN gradient in {name}"

    def test_all_params_receive_gradient(self, attn_train):
        x = torch.randn(BATCH, SEQ_LEN, D_MODEL, device=DEVICE)
        attn_train(x).sum().backward()
        no_grad = [n for n, p in attn_train.named_parameters() if p.grad is None]
        assert not no_grad, f"No gradient for: {no_grad}"

    def test_input_receives_gradient(self, attn_train):
        x = torch.randn(BATCH, SEQ_LEN, D_MODEL, device=DEVICE, requires_grad=True)
        attn_train(x).sum().backward()
        assert x.grad is not None
        assert not torch.isnan(x.grad).any()


# ---------------------------------------------------------------------------
# 6. Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:

    def test_window_1(self):
        """Window of 1 — each token only sees immediate neighbours."""
        attn = make_attn(window=1)
        x = torch.randn(BATCH, SEQ_LEN, D_MODEL, device=DEVICE)
        out = attn(x)
        assert out.shape == x.shape
        assert not torch.isnan(out).any()

    def test_window_equals_seqlen(self):
        """Window >= L/2 is equivalent to full attention — should still work."""
        attn = make_attn(window=SEQ_LEN)
        x = torch.randn(BATCH, SEQ_LEN, D_MODEL, device=DEVICE)
        out = attn(x)
        assert out.shape == x.shape
        assert not torch.isnan(out).any()

    def test_fp16(self):
        if DEVICE == "cpu":
            pytest.skip("fp16 on CPU is unreliable")
        attn = make_attn().half()
        x = torch.randn(BATCH, SEQ_LEN, D_MODEL, device=DEVICE, dtype=torch.float16)
        out = attn(x)
        assert out.dtype == torch.float16
        assert not torch.isnan(out).any()

    def test_serialization(self, tmp_path):
        attn = make_attn()
        path = tmp_path / "local_window_attn.pt"
        torch.save(attn.state_dict(), path)
        loaded = make_attn()
        loaded.load_state_dict(torch.load(path, map_location=DEVICE))
        loaded.eval()
        x = torch.randn(BATCH, SEQ_LEN, D_MODEL, device=DEVICE)
        with torch.no_grad():
            torch.testing.assert_close(attn(x), loaded(x))

    def test_d_model_not_divisible_by_nheads_raises(self):
        with pytest.raises(AssertionError):
            LocalWindowAttention(d_model=100, nheads=3, window=4)


# ===========================================================================
# CROSS MODEL SPARSE ATTENTION TESTS
# ===========================================================================

# Shared config for cross-attn tests — deliberately different from LWA config
# so fixtures don't collide
D_MODEL_C  = 128
NHEADS_C   = 4
TOP_K      = 8
BATCH_C    = 2
T_TEXT     = 16     # decoder text token count
T_AUDIO    = 64     # encoder audio frame count


def make_cross(**kwargs) -> CrossModelSparseAttention:
    cfg = dict(d_model=D_MODEL_C, nheads=NHEADS_C, top_k=TOP_K,
               dropout=0.0, use_gate=True)
    cfg.update(kwargs)
    return CrossModelSparseAttention(**cfg).to(DEVICE).eval()


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def cross_eval() -> CrossModelSparseAttention:
    return make_cross()


@pytest.fixture(scope="module")
def cross_eval_no_gate() -> CrossModelSparseAttention:
    return make_cross(use_gate=False)


@pytest.fixture
def cross_train() -> CrossModelSparseAttention:
    return make_cross().train()


# ---------------------------------------------------------------------------
# 1. Shape tests
# ---------------------------------------------------------------------------

class TestCrossModelShape:

    def test_output_shape(self, cross_eval):
        txt = torch.randn(BATCH_C, T_TEXT,  D_MODEL_C, device=DEVICE)
        aud = torch.randn(BATCH_C, T_AUDIO, D_MODEL_C, device=DEVICE)
        out = cross_eval(txt, aud)
        assert out.shape == txt.shape, f"Expected {txt.shape}, got {out.shape}"

    def test_output_matches_text_shape_not_audio(self, cross_eval):
        """Output must always have T_text timesteps, never T_audio."""
        txt = torch.randn(BATCH_C, T_TEXT,  D_MODEL_C, device=DEVICE)
        aud = torch.randn(BATCH_C, T_AUDIO, D_MODEL_C, device=DEVICE)
        out = cross_eval(txt, aud)
        assert out.shape[1] == T_TEXT
        assert out.shape[1] != T_AUDIO

    def test_batch_1(self, cross_eval):
        txt = torch.randn(1, T_TEXT,  D_MODEL_C, device=DEVICE)
        aud = torch.randn(1, T_AUDIO, D_MODEL_C, device=DEVICE)
        assert cross_eval(txt, aud).shape == txt.shape

    def test_single_text_token(self, cross_eval):
        txt = torch.randn(BATCH_C, 1,      D_MODEL_C, device=DEVICE)
        aud = torch.randn(BATCH_C, T_AUDIO, D_MODEL_C, device=DEVICE)
        out = cross_eval(txt, aud)
        assert out.shape == txt.shape

    def test_top_k_larger_than_audio_clamps(self):
        """top_k > T_audio must not crash — should clamp to T_audio."""
        model = make_cross(top_k=200)   # way more than T_AUDIO=64
        txt = torch.randn(BATCH_C, T_TEXT,  D_MODEL_C, device=DEVICE)
        aud = torch.randn(BATCH_C, T_AUDIO, D_MODEL_C, device=DEVICE)
        out = model(txt, aud)
        assert out.shape == txt.shape
        assert not torch.isnan(out).any()

    def test_various_nheads(self):
        for nheads in [1, 2, 4, 8]:
            model = make_cross(d_model=128, nheads=nheads)
            txt = torch.randn(BATCH_C, T_TEXT,  D_MODEL_C, device=DEVICE)
            aud = torch.randn(BATCH_C, T_AUDIO, D_MODEL_C, device=DEVICE)
            assert model(txt, aud).shape == txt.shape

    def test_d_model_not_divisible_by_nheads_raises(self):
        with pytest.raises(AssertionError):
            CrossModelSparseAttention(d_model=100, nheads=3, top_k=8)


# ---------------------------------------------------------------------------
# 2. Numerical sanity
# ---------------------------------------------------------------------------

class TestCrossModelNumericalSanity:

    def test_no_nan(self, cross_eval):
        txt = torch.randn(BATCH_C, T_TEXT,  D_MODEL_C, device=DEVICE)
        aud = torch.randn(BATCH_C, T_AUDIO, D_MODEL_C, device=DEVICE)
        assert not torch.isnan(cross_eval(txt, aud)).any()

    def test_no_inf(self, cross_eval):
        txt = torch.randn(BATCH_C, T_TEXT,  D_MODEL_C, device=DEVICE)
        aud = torch.randn(BATCH_C, T_AUDIO, D_MODEL_C, device=DEVICE)
        assert not torch.isinf(cross_eval(txt, aud)).any()

    def test_not_all_zeros(self, cross_eval_no_gate):
        """Use no-gate model — gate=0 at init zeros out audio contribution."""
        txt = torch.randn(BATCH_C, T_TEXT,  D_MODEL_C, device=DEVICE)
        aud = torch.randn(BATCH_C, T_AUDIO, D_MODEL_C, device=DEVICE)
        out = cross_eval_no_gate(txt, aud)
        assert out.abs().max() > 1e-6

    def test_zero_audio_input(self, cross_eval):
        txt = torch.randn(BATCH_C, T_TEXT,  D_MODEL_C, device=DEVICE)
        aud = torch.zeros(BATCH_C, T_AUDIO, D_MODEL_C, device=DEVICE)
        out = cross_eval(txt, aud)
        assert not torch.isnan(out).any()
        assert not torch.isinf(out).any()

    def test_deterministic(self, cross_eval):
        txt = torch.randn(BATCH_C, T_TEXT,  D_MODEL_C, device=DEVICE)
        aud = torch.randn(BATCH_C, T_AUDIO, D_MODEL_C, device=DEVICE)
        torch.testing.assert_close(
            cross_eval(txt, aud), cross_eval(txt, aud)
        )


# ---------------------------------------------------------------------------
# 3. THE KEY TESTS — cross-modal sparsity and directionality
# ---------------------------------------------------------------------------

class TestCrossModelSparsity:

    def test_audio_influences_text_output(self, cross_eval_no_gate):
        """
        Changing audio frames must change the text output.
        Uses no-gate model so the gate=0 init doesn't suppress the signal.
        """
        torch.manual_seed(10)
        txt = torch.randn(1, T_TEXT,  D_MODEL_C, device=DEVICE)
        aud1 = torch.randn(1, T_AUDIO, D_MODEL_C, device=DEVICE)
        aud2 = aud1.clone()
        aud2 += 5.0     # perturb ALL audio frames

        out1 = cross_eval_no_gate(txt, aud1)
        out2 = cross_eval_no_gate(txt, aud2)

        assert not torch.allclose(out1, out2, atol=1e-5), (
            "Changing audio did not change text output — "
            "audio is not flowing into the cross-attention"
        )

    def test_text_change_does_not_affect_unrelated_positions(self, cross_eval_no_gate):
        """
        Cross-attention queries come from text. Changing a text token changes
        WHICH audio frames that position attends to, so its own output changes.
        But other text positions query independently — their outputs may or may
        not change depending on whether they share selected audio frames.
        We only verify that text position 0 changes when text position 0 changes.
        """
        torch.manual_seed(11)
        txt1 = torch.randn(1, T_TEXT,  D_MODEL_C, device=DEVICE)
        txt2 = txt1.clone()
        txt2[:, 0, :] += 10.0

        aud = torch.randn(1, T_AUDIO, D_MODEL_C, device=DEVICE)

        out1 = cross_eval_no_gate(txt1, aud)
        out2 = cross_eval_no_gate(txt2, aud)

        assert not torch.allclose(out1[:, 0, :], out2[:, 0, :], atol=1e-5), (
            "Changing text position 0 did not change output at position 0"
        )

    def test_output_is_residual_of_text_not_audio(self, cross_eval_no_gate):
        """
        The output has shape (B, T_text, D) and is a residual on top of
        text_hidden. Completely zeroing the audio should leave the output
        equal to text_hidden (pure residual passthrough), because
        V=0 → weighted sum=0 → out_proj(0)=0 → output = text_hidden + 0.
        """
        torch.manual_seed(12)
        txt = torch.randn(1, T_TEXT,  D_MODEL_C, device=DEVICE)
        aud = torch.zeros(1, T_AUDIO, D_MODEL_C, device=DEVICE)

        out = cross_eval_no_gate(txt, aud)

        torch.testing.assert_close(
            out, txt,
            atol=1e-5, rtol=1e-5,
            msg="Zero audio did not produce pure residual passthrough — "
                "output should equal text_hidden when audio is all zeros"
        )

    def test_only_top_k_audio_frames_matter(self, cross_eval_no_gate):
        """
        THE key sparsity test.

        Strategy: find which k audio frames a text token selected, then
        perturb ONLY frames that were NOT selected — output must not change.

        We extract the top-k indices by monkey-patching the forward pass.
        """
        torch.manual_seed(13)
        model = make_cross(use_gate=False)
        model.eval()

        txt = torch.randn(1, T_TEXT,  D_MODEL_C, device=DEVICE)
        aud = torch.randn(1, T_AUDIO, D_MODEL_C, device=DEVICE)

        # Capture top-k indices during forward
        captured = {}
        original_forward = model.forward

        def patched_forward(text_hidden, audio_out, audio_padding_mask=None):
            B, T_t, D = text_hidden.shape
            _, T_a, _  = audio_out.shape
            k = min(model.top_k, T_a)

            q_in  = model.norm_text(text_hidden)
            kv_in = model.norm_audio(audio_out)
            Q = model.q_proj(q_in)
            K = model.k_proj(kv_in)

            Q = Q.view(B, T_t, model.nheads, model.headdim).transpose(1, 2)
            K = K.view(B, T_a, model.nheads, model.headdim).transpose(1, 2)

            scores = torch.matmul(Q, K.transpose(-2, -1)) * model.scale
            _, topk_indices = scores.topk(k, dim=-1)
            captured["indices"] = topk_indices   # (B, H, T_text, k)

            return original_forward(text_hidden, audio_out, audio_padding_mask)

        model.forward = patched_forward
        out1 = model(txt, aud)
        model.forward = original_forward

        # Collect the UNION of selected frames across ALL heads for text token 0.
        # captured["indices"] shape: (B=1, H, T_text, k)
        # Each head selects a different set of k frames — the final output at
        # position 0 is the merge of all heads, so a frame is "used" if ANY
        # head selected it. We must only perturb frames no head selected.
        selected_union = set()
        for h in range(model.nheads):
            selected_union.update(captured["indices"][0, h, 0].tolist())

        all_indices = set(range(T_AUDIO))
        unselected = list(all_indices - selected_union)

        assert len(unselected) > 0, (
            "Every audio frame was selected by at least one head — "
            "increase T_AUDIO or decrease top_k to make the test meaningful"
        )

        # Perturb only frames that NO head selected
        aud2 = aud.clone()
        for idx in unselected:
            aud2[:, idx, :] += 100.0

        out2 = model(txt, aud2)

        # Output at position 0 must be unchanged — the frames that changed
        # were not selected by any head, so they never entered the computation
        torch.testing.assert_close(
            out1[:, 0, :], out2[:, 0, :],
            atol=1e-3, rtol=1e-3,
            msg="Perturbing only non-selected audio frames changed the output — "
                "sparsity is not working correctly"
        )


# ---------------------------------------------------------------------------
# 4. Gate tests
# ---------------------------------------------------------------------------

class TestCrossModelGate:

    def test_gate_zero_at_init_means_no_audio_influence(self):
        """
        At init gate=0 → tanh(0)=0 → audio contribution is fully suppressed.
        Output must equal text_hidden exactly.
        """
        torch.manual_seed(20)
        model = make_cross(use_gate=True)
        model.eval()

        # Confirm gate is initialised to 0
        assert model.gate.item() == 0.0, "Gate should be initialised to 0"

        txt = torch.randn(1, T_TEXT,  D_MODEL_C, device=DEVICE)
        aud = torch.randn(1, T_AUDIO, D_MODEL_C, device=DEVICE)
        out = model(txt, aud)

        torch.testing.assert_close(
            out, txt,
            atol=1e-5, rtol=1e-5,
            msg="Gate=0 at init should suppress all audio influence — "
                "output should equal text_hidden"
        )

    def test_gate_open_allows_audio_influence(self):
        """After manually opening the gate, audio must influence the output."""
        torch.manual_seed(21)
        model = make_cross(use_gate=True)
        model.eval()

        with torch.no_grad():
            model.gate.fill_(2.0)   # tanh(2) ≈ 0.96 — almost fully open

        txt = torch.randn(1, T_TEXT,  D_MODEL_C, device=DEVICE)
        aud1 = torch.randn(1, T_AUDIO, D_MODEL_C, device=DEVICE)
        aud2 = aud1.clone()
        aud2 += 5.0

        out1 = model(txt, aud1)
        out2 = model(txt, aud2)

        assert not torch.allclose(out1, out2, atol=1e-5), (
            "With gate open, changing audio should change output"
        )

    def test_no_gate_model_has_no_gate_param(self, cross_eval_no_gate):
        """use_gate=False must not create a gate parameter."""
        assert not hasattr(cross_eval_no_gate, "gate"), (
            "use_gate=False model should not have a gate parameter"
        )


# ---------------------------------------------------------------------------
# 5. Padding mask tests
# ---------------------------------------------------------------------------

class TestCrossModelPaddingMask:

    def test_padded_audio_does_not_influence_output(self):
        """
        Padded audio frames (audio_padding_mask=True) must be excluded from
        top-k selection entirely — perturbing them should not change the output.

        Same correct pattern as the LocalWindowAttention padding test:
        both calls use the SAME mask, only the values differ.
        """
        torch.manual_seed(30)
        model = make_cross(use_gate=False, top_k=4)
        model.eval()

        txt = torch.randn(1, T_TEXT,  D_MODEL_C, device=DEVICE)
        aud = torch.randn(1, T_AUDIO, D_MODEL_C, device=DEVICE)

        # Pad the last 32 audio frames — well more than top_k=4
        pad_mask = torch.zeros(1, T_AUDIO, dtype=torch.bool, device=DEVICE)
        pad_mask[:, 32:] = True

        out1 = model(txt, aud,  audio_padding_mask=pad_mask)

        aud2 = aud.clone()
        aud2[:, 32:] += 100.0   # large perturbation of padded frames only
        out2 = model(txt, aud2, audio_padding_mask=pad_mask)

        torch.testing.assert_close(
            out1, out2,
            atol=1e-4, rtol=1e-4,
            msg="Padded audio frames influenced the output — "
                "audio_padding_mask is not working"
        )

    def test_all_audio_padded_produces_finite_output(self):
        """All audio masked → all scores are -inf → softmax NaN → nan_to_num handles it."""
        model = make_cross(use_gate=False)
        model.eval()
        txt = torch.randn(BATCH_C, T_TEXT,  D_MODEL_C, device=DEVICE)
        aud = torch.randn(BATCH_C, T_AUDIO, D_MODEL_C, device=DEVICE)
        pad_mask = torch.ones(BATCH_C, T_AUDIO, dtype=torch.bool, device=DEVICE)
        out = model(txt, aud, audio_padding_mask=pad_mask)
        assert not torch.isnan(out).any()
        assert not torch.isinf(out).any()


# ---------------------------------------------------------------------------
# 6. Gradient tests
# ---------------------------------------------------------------------------

class TestCrossModelGradients:

    def test_backward_runs(self, cross_train):
        txt = torch.randn(BATCH_C, T_TEXT,  D_MODEL_C, device=DEVICE)
        aud = torch.randn(BATCH_C, T_AUDIO, D_MODEL_C, device=DEVICE)
        cross_train(txt, aud).sum().backward()

    def test_no_nan_gradients(self, cross_train):
        txt = torch.randn(BATCH_C, T_TEXT,  D_MODEL_C, device=DEVICE)
        aud = torch.randn(BATCH_C, T_AUDIO, D_MODEL_C, device=DEVICE)
        cross_train(txt, aud).sum().backward()
        for name, p in cross_train.named_parameters():
            if p.grad is not None:
                assert not torch.isnan(p.grad).any(), f"NaN gradient in {name}"

    def test_all_params_receive_gradient(self, cross_train):
        txt = torch.randn(BATCH_C, T_TEXT,  D_MODEL_C, device=DEVICE)
        aud = torch.randn(BATCH_C, T_AUDIO, D_MODEL_C, device=DEVICE)
        cross_train(txt, aud).sum().backward()
        no_grad = [n for n, p in cross_train.named_parameters() if p.grad is None]
        assert not no_grad, f"No gradient for: {no_grad}"

    def test_gradient_flows_to_audio_input(self, cross_train):
        """Gradient must flow back into audio_out — encoder needs to be trainable."""
        txt = torch.randn(BATCH_C, T_TEXT,  D_MODEL_C, device=DEVICE)
        aud = torch.randn(BATCH_C, T_AUDIO, D_MODEL_C, device=DEVICE,
                          requires_grad=True)
        cross_train(txt, aud).sum().backward()
        assert aud.grad is not None, "No gradient flowed back to audio_out"
        assert not torch.isnan(aud.grad).any()

    def test_gradient_flows_to_text_input(self, cross_train):
        txt = torch.randn(BATCH_C, T_TEXT,  D_MODEL_C, device=DEVICE,
                          requires_grad=True)
        aud = torch.randn(BATCH_C, T_AUDIO, D_MODEL_C, device=DEVICE)
        cross_train(txt, aud).sum().backward()
        assert txt.grad is not None, "No gradient flowed back to text_hidden"
        assert not torch.isnan(txt.grad).any()

    def test_gate_receives_gradient(self):
        """Gate parameter must receive gradient so it can learn to open."""
        model = make_cross(use_gate=True).train()
        txt = torch.randn(BATCH_C, T_TEXT,  D_MODEL_C, device=DEVICE)
        aud = torch.randn(BATCH_C, T_AUDIO, D_MODEL_C, device=DEVICE)
        model(txt, aud).sum().backward()
        assert model.gate.grad is not None, "Gate parameter received no gradient"
        assert not torch.isnan(model.gate.grad).any()


# ---------------------------------------------------------------------------
# 7. Edge cases
# ---------------------------------------------------------------------------

class TestCrossModelEdgeCases:

    def test_top_k_1(self):
        """top_k=1 — each text token attends to exactly one audio frame."""
        model = make_cross(top_k=1)
        txt = torch.randn(BATCH_C, T_TEXT,  D_MODEL_C, device=DEVICE)
        aud = torch.randn(BATCH_C, T_AUDIO, D_MODEL_C, device=DEVICE)
        out = model(txt, aud)
        assert out.shape == txt.shape
        assert not torch.isnan(out).any()

    def test_top_k_equals_audio_len(self):
        """top_k == T_audio — equivalent to full cross-attention."""
        model = make_cross(top_k=T_AUDIO)
        txt = torch.randn(BATCH_C, T_TEXT,  D_MODEL_C, device=DEVICE)
        aud = torch.randn(BATCH_C, T_AUDIO, D_MODEL_C, device=DEVICE)
        out = model(txt, aud)
        assert out.shape == txt.shape
        assert not torch.isnan(out).any()

    def test_fp16(self):
        if DEVICE == "cpu":
            pytest.skip("fp16 on CPU is unreliable")
        model = make_cross().half()
        txt = torch.randn(BATCH_C, T_TEXT,  D_MODEL_C, device=DEVICE,
                          dtype=torch.float16)
        aud = torch.randn(BATCH_C, T_AUDIO, D_MODEL_C, device=DEVICE,
                          dtype=torch.float16)
        out = model(txt, aud)
        assert out.dtype == torch.float16
        assert not torch.isnan(out).any()

    def test_serialization(self, tmp_path):
        model = make_cross()
        path = tmp_path / "cross_model_sparse_attn.pt"
        torch.save(model.state_dict(), path)
        loaded = make_cross()
        loaded.load_state_dict(torch.load(path, map_location=DEVICE))
        loaded.eval()
        txt = torch.randn(BATCH_C, T_TEXT,  D_MODEL_C, device=DEVICE)
        aud = torch.randn(BATCH_C, T_AUDIO, D_MODEL_C, device=DEVICE)
        with torch.no_grad():
            torch.testing.assert_close(model(txt, aud), loaded(txt, aud))

    def test_different_text_and_audio_seq_lengths(self):
        """T_text and T_audio can be completely independent lengths."""
        model = make_cross()
        txt = torch.randn(BATCH_C, 7,   D_MODEL_C, device=DEVICE)
        aud = torch.randn(BATCH_C, 200, D_MODEL_C, device=DEVICE)
        out = model(txt, aud)
        assert out.shape == txt.shape