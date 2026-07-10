"""M-RoPE layout guarantees for UamGR00T_LT's depth block.

The LT framework appends the primary image a second time so the depth tokens
land in a real vision block.  Everything downstream (the per-token depth head,
the 7x7 grid reshape) assumes depth token k corresponds to primary patch k.
That correspondence is not given by the token id -- it is given by
``get_rope_index`` finding a ``<|vision_start|>`` and consuming one
``image_grid_thw`` row.

These tests pin that behaviour.  They need no model weights: ``get_rope_index``
only reads ``self.config.{vision_config.spatial_merge_size, image_token_id,
video_token_id, vision_start_token_id}``.
"""
from __future__ import annotations

import json
import types
from pathlib import Path

import pytest
import torch

QWEN_CONFIG = Path("ckpt/Qwen3-VL-4B-Instruct/config.json")
GRID = 7               # 224 / 32
TOKENS_PER_BLOCK = GRID * GRID


def _load_ids():
    if not QWEN_CONFIG.exists():
        pytest.skip(f"{QWEN_CONFIG} not present")
    cfg = json.loads(QWEN_CONFIG.read_text())
    return cfg["image_token_id"], cfg["vision_start_token_id"], cfg["vision_end_token_id"], cfg["video_token_id"]


def _rope(input_ids: list[int], n_images: int, ids):
    """Call the unbound get_rope_index with a stand-in ``self``."""
    from transformers.models.qwen3_vl import modeling_qwen3_vl as m

    image_id, vstart_id, _, video_id = ids
    fake_self = types.SimpleNamespace(
        config=types.SimpleNamespace(
            vision_config=types.SimpleNamespace(spatial_merge_size=2),
            image_token_id=image_id,
            video_token_id=video_id,
            vision_start_token_id=vstart_id,
        )
    )
    grid_thw = torch.tensor([[1, GRID * 2, GRID * 2]] * n_images)
    position_ids, _ = m.Qwen3VLModel.get_rope_index(
        fake_self, torch.tensor([input_ids]), image_grid_thw=grid_thw
    )
    return position_ids[:, 0]  # (3, L) == (t, h, w)


def _image_blocks(input_ids: list[int], image_id: int) -> list[list[int]]:
    pos = [i for i, v in enumerate(input_ids) if v == image_id]
    return [pos[i:i + TOKENS_PER_BLOCK] for i in range(0, len(pos), TOKENS_PER_BLOCK)]


def _text(n: int, start: int) -> list[int]:
    return list(range(start, start + n))


def test_depth_block_registered_as_vision_block_gets_2d_grid():
    """Block 2 (the duplicated primary) must receive a real 7x7 M-RoPE grid."""
    image_id, vstart_id, vend_id, _ = ids = _load_ids()
    block = [vstart_id] + [image_id] * TOKENS_PER_BLOCK + [vend_id]
    input_ids = _text(6, 1000) + block + _text(4, 2000) + block + _text(6, 3000) + block + _text(3, 4000)

    pos = _rope(input_ids, n_images=3, ids=ids)
    h, w = pos[1], pos[2]

    blocks = _image_blocks(input_ids, image_id)
    assert len(blocks) == 3

    for b, blk in enumerate(blocks):
        hb = h[blk].reshape(GRID, GRID)
        wb = w[blk].reshape(GRID, GRID)
        # h varies down the rows, is constant across a row; w is the transpose.
        assert torch.equal(hb.diff(dim=0), torch.ones(GRID - 1, GRID, dtype=hb.dtype)), f"block {b} h not a grid"
        assert torch.equal(hb.diff(dim=1), torch.zeros(GRID, GRID - 1, dtype=hb.dtype)), f"block {b} h not a grid"
        assert torch.equal(wb.diff(dim=1), torch.ones(GRID, GRID - 1, dtype=wb.dtype)), f"block {b} w not a grid"
        assert torch.equal(wb.diff(dim=0), torch.zeros(GRID - 1, GRID, dtype=wb.dtype)), f"block {b} w not a grid"


def test_depth_to_primary_offset_is_constant():
    """depth[k] - view0[k] must be one single (dh, dw) for all 49 k.

    This is what lets RoPE's relative-position encoding express the
    depth-token-to-patch correspondence.  If it ever becomes non-constant the
    per-token head is regressing mismatched grid cells.
    """
    image_id, vstart_id, vend_id, _ = ids = _load_ids()
    block = [vstart_id] + [image_id] * TOKENS_PER_BLOCK + [vend_id]
    input_ids = _text(6, 1000) + block + _text(4, 2000) + block + _text(6, 3000) + block + _text(3, 4000)

    pos = _rope(input_ids, n_images=3, ids=ids)
    h, w = pos[1], pos[2]
    view0, depth = _image_blocks(input_ids, image_id)[0], _image_blocks(input_ids, image_id)[2]

    dh = {int(h[depth[k]] - h[view0[k]]) for k in range(TOKENS_PER_BLOCK)}
    dw = {int(w[depth[k]] - w[view0[k]]) for k in range(TOKENS_PER_BLOCK)}
    assert len(dh) == 1, f"dh must be constant across all 49 tokens, got {sorted(dh)}"
    assert len(dw) == 1, f"dw must be constant across all 49 tokens, got {sorted(dw)}"


def test_bare_image_pad_ids_do_not_get_a_grid():
    """Regression guard: reusing <|image_pad|> in a text suffix is NOT enough.

    Without a preceding <|vision_start|> and a matching image_grid_thw row,
    get_rope_index treats the tokens as plain text and assigns the diagonal
    t == h == w, destroying the 2D correspondence.  This test documents the
    trap so nobody "simplifies" _encode_qwen_hidden into a text_suffix.
    """
    image_id, vstart_id, vend_id, _ = ids = _load_ids()
    block = [vstart_id] + [image_id] * TOKENS_PER_BLOCK + [vend_id]
    input_ids = (
        _text(6, 1000) + block + _text(4, 2000) + block
        + _text(6, 3000) + [image_id] * TOKENS_PER_BLOCK + _text(3, 4000)
    )

    pos = _rope(input_ids, n_images=2, ids=ids)
    h, w = pos[1], pos[2]
    view0, fake_depth = _image_blocks(input_ids, image_id)[0], _image_blocks(input_ids, image_id)[2]

    dh = {int(h[fake_depth[k]] - h[view0[k]]) for k in range(TOKENS_PER_BLOCK)}
    assert len(dh) > 1, "text-suffix image_pad unexpectedly produced a constant offset"
    # and the block itself is a diagonal, not a grid
    assert torch.equal(h[fake_depth], w[fake_depth])
