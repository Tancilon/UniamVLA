"""LatentDepthHead: shapes, ZeRO alignment, R^2 semantics, probe isolation."""
from __future__ import annotations

import pytest
import torch

from starVLA.model.modules.uamvla.aux_heads.latent_depth_head import LatentDepthHead

HIDDEN, PPV, GRID, PATCH, LATENT_C = 64, 49, 7, 16, 64
IMAGE_TOKEN_ID, VIEW_IDX, NUM_VIEWS = 999, 2, 3


def make_head(**kw):
    return LatentDepthHead(
        hidden_size=HIDDEN, image_token_id=IMAGE_TOKEN_ID, patches_per_view=PPV,
        view_idx=VIEW_IDX, latent_channels=LATENT_C, depth_px_size=GRID * PATCH, **kw,
    )


def make_batch(B=2, seq_pad=5):
    """input_ids with NUM_VIEWS contiguous image-token blocks, view 2 last."""
    L = seq_pad + PPV * NUM_VIEWS
    input_ids = torch.zeros(B, L, dtype=torch.long)
    input_ids[:, seq_pad:] = IMAGE_TOKEN_ID
    hidden = torch.randn(B, L, HIDDEN)
    return hidden, {
        "input_ids": input_ids,
        "depth_latent": torch.randn(B, LATENT_C, GRID, GRID),
        "depth_px": torch.rand(B, GRID * PATCH, GRID * PATCH),
    }


def test_slices_view2_and_produces_aligned_shapes():
    head = make_head()
    hidden, batch = make_batch()
    out = head.compute_loss(hidden, batch, mask=torch.ones(2, dtype=torch.bool))
    assert out.loss.ndim == 0 and torch.isfinite(out.loss)
    assert {"r2", "var_ratio", "cosine", "probe_r2_pixel", "head_mse", "probe_mse", "valid_ratio"} <= set(out.metrics)


def test_slice_targets_the_third_block_not_the_first():
    """view_idx must select block 2. Zeroing block 2 alone must change the loss."""
    head = make_head()
    hidden, batch = make_batch(B=1)
    mask = torch.ones(1, dtype=torch.bool)
    base = head.compute_loss(hidden, batch, mask).loss.item()

    h_block0 = hidden.clone()
    h_block0[:, 5:5 + PPV] = 0.0                     # perturb block 0 only
    assert head.compute_loss(h_block0, batch, mask).loss.item() == pytest.approx(base)

    h_block2 = hidden.clone()
    h_block2[:, 5 + 2 * PPV:] = 0.0                  # perturb block 2 only
    assert head.compute_loss(h_block2, batch, mask).loss.item() != pytest.approx(base)


def test_r2_is_one_on_perfect_prediction_and_zero_on_mean_prediction():
    """R^2 == 1 when z_pred == z_gt, and == 0 for the constant-mean baseline.

    _r2 normalises by the *global* variance, so the zero point is the single
    scalar mean of the batch -- "no better than predicting the batch mean".
    """
    head = make_head()
    z_gt = torch.randn(4, PPV, LATENT_C)
    assert head._r2(z_gt, z_gt) == pytest.approx(1.0, abs=1e-5)
    assert head._r2(torch.full_like(z_gt, float(z_gt.mean())), z_gt) == pytest.approx(0.0, abs=1e-5)


def test_probe_gradient_never_reaches_the_backbone():
    """The detached probe must not shape the backbone.

    Note: freezing the training head's *parameters* would not isolate the probe
    -- gradients still flow through a frozen Linear to its input.  Zeroing
    loss_weight removes the training-head term from the loss entirely.
    """
    head = make_head(head_kind="linear", loss_weight=0.0)
    hidden, batch = make_batch(B=1)
    hidden.requires_grad_(True)

    out = head.compute_loss(hidden, batch, torch.ones(1, dtype=torch.bool))
    out.loss.backward()

    assert hidden.grad is None or torch.count_nonzero(hidden.grad) == 0, \
        "probe leaked gradient into the backbone hidden states"
    assert any(p.grad is not None and torch.count_nonzero(p.grad) for p in head.probe.parameters()), \
        "probe parameters received no gradient"


def test_training_head_does_reach_the_backbone():
    head = make_head(probe_enabled=False)
    hidden, batch = make_batch(B=1)
    hidden.requires_grad_(True)
    head.compute_loss(hidden, batch, torch.ones(1, dtype=torch.bool)).loss.backward()
    assert torch.count_nonzero(hidden.grad) > 0


def test_empty_mask_returns_grad_carrying_zero_loss():
    """ZeRO-3 collectives stay aligned only if every rank runs the same forward."""
    head = make_head()
    hidden, batch = make_batch(B=2)
    out = head.compute_loss(hidden, batch, mask=torch.zeros(2, dtype=torch.bool))
    assert out.loss.requires_grad and out.loss.item() == pytest.approx(0.0)
    out.loss.backward()
    touched = [n for n, p in head.named_parameters() if p.grad is not None]
    assert any("head." in n for n in touched) and any("probe." in n for n in touched), \
        f"dummy path skipped some params: {touched}"


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
def test_accepts_autocast_dtypes_from_the_backbone(dtype):
    """Qwen emits bf16 under autocast; aux heads run outside any autocast block."""
    head = make_head()
    hidden, batch = make_batch(B=2)
    out = head.compute_loss(hidden.to(dtype), batch, mask=torch.ones(2, dtype=torch.bool))
    assert out.loss.dtype == torch.float32 and torch.isfinite(out.loss)


def test_empty_mask_path_also_accepts_bfloat16():
    head = make_head()
    hidden, batch = make_batch(B=2)
    out = head.compute_loss(hidden.bfloat16(), batch, mask=torch.zeros(2, dtype=torch.bool))
    assert out.loss.requires_grad and torch.isfinite(out.loss)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA for autocast")
@pytest.mark.parametrize("hidden_dtype", [torch.float32, torch.bfloat16])
def test_bf16_head_parameters_accept_any_hidden_dtype(hidden_dtype):
    """DeepSpeed casts the head's Linear weights to bf16.

    nn.Linear rejects an fp32 activation against a bf16 weight, so compute_loss
    must run its projections under autocast(fp32).  A CPU-only test cannot see
    this: there the parameters stay fp32.
    """
    head = make_head().cuda().bfloat16()
    hidden, batch = make_batch(B=2)
    hidden = hidden.cuda().to(hidden_dtype)
    batch = {k: v.cuda() for k, v in batch.items()}

    out = head.compute_loss(hidden, batch, mask=torch.ones(2, dtype=torch.bool, device="cuda"))
    assert torch.isfinite(out.loss)

    empty = head.compute_loss(hidden, batch, mask=torch.zeros(2, dtype=torch.bool, device="cuda"))
    assert empty.loss.requires_grad and torch.isfinite(empty.loss)


def test_head_has_no_cross_token_mixing():
    """Hard constraint: perturbing token j must never change token i's output.

    Perturb a single channel: adding a constant to *all* channels of a token is
    a no-op under the non-affine LayerNorm and would prove nothing.
    """
    head = make_head()
    h = torch.randn(1, PPV, HIDDEN)
    z0 = head.head(head.ln_pre(h))
    h2 = h.clone()
    h2[:, 3, 0] += 10.0
    z1 = head.head(head.ln_pre(h2))
    changed = (z0 - z1).abs().amax(dim=-1)[0] > 1e-6
    assert changed[3], "perturbation had no effect at all"
    assert changed.sum() == 1, "head mixes information across tokens"


@pytest.mark.parametrize("kind,expected", [("linear", 2560 * 64 + 64), ("pointwise_mlp", None)])
def test_head_kind_switch(kind, expected):
    head = LatentDepthHead(
        hidden_size=2560, image_token_id=IMAGE_TOKEN_ID, patches_per_view=PPV,
        view_idx=VIEW_IDX, latent_channels=64, depth_px_size=112, head_kind=kind,
    )
    n = sum(p.numel() for p in head.head.parameters())
    if expected is not None:
        assert n == expected
    else:
        assert n == 2560 * 640 + 640 + 640 * 64 + 64
