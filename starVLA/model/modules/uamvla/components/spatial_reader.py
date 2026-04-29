"""SpatialReader: stateless utility to slice image-token hidden states from
the LLM output by locating <image> placeholder positions in input_ids.

Spec ref: docs/superpowers/specs/2026-04-24-recon-spatial-reader-design.md §2.1
"""
import torch


def slice_image_tokens(
    hidden_states: torch.Tensor,
    input_ids: torch.Tensor,
    image_token_id: int,
    patches_per_view: int,
    view_idx: int = 0,
) -> torch.Tensor:
    """Return flat image-token features for one view.

    Args:
        hidden_states: (B, L, H) LLM output.
        input_ids: (B, L) token ids. Positions where input_ids == image_token_id
            mark the per-sample image-token block.
        image_token_id: ID of the <image> placeholder token.
        patches_per_view: Number of image tokens per view (e.g. 729 for
            SigLIP so400m @ 384/patch14).
        view_idx: Which view to extract (0 = first view in image list).

    Returns:
        (B, patches_per_view, H) tensor of image-token hidden states.

    Raises:
        AssertionError: Per-sample image-token count is not a multiple of
            patches_per_view, or is insufficient for the requested view_idx.
    """
    B = hidden_states.shape[0]
    out = []
    start = view_idx * patches_per_view
    end = (view_idx + 1) * patches_per_view

    for b in range(B):
        positions = (input_ids[b] == image_token_id).nonzero(as_tuple=True)[0]
        n = positions.shape[0]
        assert n % patches_per_view == 0, (
            f"sample {b}: expected image-token count divisible by "
            f"patches_per_view={patches_per_view}, got {n}"
        )
        assert n >= end, (
            f"sample {b}: need at least {end} image tokens for view_idx="
            f"{view_idx} (patches_per_view={patches_per_view}), got {n}"
        )
        selected = positions[start:end]
        out.append(hidden_states[b, selected, :])

    return torch.stack(out, dim=0)
