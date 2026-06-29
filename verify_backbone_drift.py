#!/usr/bin/env python
"""Verify whether enabling the recon aux branch actually changed the Qwen3-VL backbone.

Two independent checks (run --mode weights first; it needs no GPU and is decisive
for the "did the backbone move at all" question = A1+A2):

  1. weights : diff the two checkpoints' parameter tensors directly. Reports, per
               top-level module (qwen_vl_interface / action_model / aux_heads / vae),
               the relative-L2 change and global cosine between baseline and recon
               weights. CPU-only.

  2. hidden  : load both models, feed the SAME real batch through each, and compare
               hidden_states[-1] (the exact tensor predict_action feeds the action
               head). Reports per-token cosine + relative-L2, split into image-token
               vs text-token positions. Needs one GPU.

Interpretation:
  - backbone weights ~identical  AND/OR  hidden drift ~0  -> A1+A2 (gradient barely
    moved the backbone; the aux DiT absorbed the loss).
  - backbone weights / hidden drift clearly nonzero, but eval SR unchanged -> A4
    (representation changed, but the change is not useful for the policy).

Usage:
  # CPU, no GPU needed:
  python verify_backbone_drift.py --mode weights \
      --base playground/Checkpoints/uamvla_gr00t_starvla26_calvin_d_baseline/final_model/pytorch_model.pt \
      --recon playground/Checkpoints/uamvla_gr00t_starvla26_calvin_d_recon/final_model/pytorch_model.pt

  # one idle GPU (check nvidia-smi first, then pick the free card):
  CUDA_VISIBLE_DEVICES=N python verify_backbone_drift.py --mode hidden \
      --base  .../uamvla_gr00t_starvla26_calvin_d_baseline/final_model/pytorch_model.pt \
      --recon .../uamvla_gr00t_starvla26_calvin_d_recon/final_model/pytorch_model.pt \
      --batch-size 8
"""
from __future__ import annotations

import argparse
from collections import defaultdict

import torch


# ----------------------------------------------------------------------------- #
#  Mode 1: weight-space diff (CPU only)
# ----------------------------------------------------------------------------- #
def _load_state_dict(path: str):
    try:
        return torch.load(path, map_location="cpu", mmap=True)  # torch>=2.1, low RAM
    except TypeError:
        return torch.load(path, map_location="cpu")


def _top_module(key: str) -> str:
    return key.split(".", 1)[0]


def run_weights(base_path: str, recon_path: str, topk: int) -> None:
    print(f"[weights] loading\n  base : {base_path}\n  recon: {recon_path}")
    sd_b = _load_state_dict(base_path)
    sd_r = _load_state_dict(recon_path)
    kb, kr = set(sd_b), set(sd_r)
    shared = kb & kr
    print(f"[weights] keys: shared={len(shared)} recon_only={len(kr - kb)} base_only={len(kb - kr)}")
    recon_only = sorted(kr - kb)
    if recon_only:
        print(f"[weights] recon-only keys (the new aux branch), first 8:")
        for k in recon_only[:8]:
            print(f"            + {k}")

    # per-bucket accumulators: [sumsq_diff, sumsq_base, sumsq_recon, dot, n_params, n_keys]
    agg = defaultdict(lambda: [0.0, 0.0, 0.0, 0.0, 0, 0])
    per_key = []  # (key, rel_l2, cosine)
    skipped = 0
    for k in sorted(shared):
        wb, wr = sd_b[k], sd_r[k]
        if not (torch.is_floating_point(wb) and torch.is_floating_point(wr)):
            skipped += 1
            continue
        if tuple(wb.shape) != tuple(wr.shape):
            skipped += 1
            continue
        wb = wb.detach().float().flatten()
        wr = wr.detach().float().flatten()
        sqd = torch.sum((wr - wb) ** 2).item()
        sqb = torch.sum(wb ** 2).item()
        sqr = torch.sum(wr ** 2).item()
        dot = torch.sum(wr * wb).item()
        rel = (sqd ** 0.5) / ((sqb ** 0.5) + 1e-12)
        cos = dot / (((sqb ** 0.5) * (sqr ** 0.5)) + 1e-12)
        per_key.append((k, rel, cos))
        a = agg[_top_module(k)]
        a[0] += sqd; a[1] += sqb; a[2] += sqr; a[3] += dot
        a[4] += wb.numel(); a[5] += 1

    print(f"\n[weights] per top-level module (relative-L2 = ||W_recon - W_base|| / ||W_base||):")
    print(f"  {'module':<22} {'rel_L2':>10} {'cosine':>10} {'#params':>14} {'#keys':>7}")
    for mod in sorted(agg):
        sqd, sqb, sqr, dot, nparam, nkeys = agg[mod]
        rel = (sqd ** 0.5) / ((sqb ** 0.5) + 1e-12)
        cos = dot / (((sqb ** 0.5) * (sqr ** 0.5)) + 1e-12)
        print(f"  {mod:<22} {rel:>10.6f} {cos:>10.6f} {nparam:>14,} {nkeys:>7}")
    if skipped:
        print(f"  (skipped {skipped} non-float / shape-mismatched keys)")

    bk = "qwen_vl_interface"
    bb = [pk for pk in per_key if pk[0].startswith(bk + ".")]
    if bb:
        bb.sort(key=lambda x: x[1], reverse=True)
        print(f"\n[weights] top {topk} most-changed BACKBONE ({bk}) params by rel-L2:")
        for k, rel, cos in bb[:topk]:
            print(f"  rel_L2={rel:>9.6f}  cos={cos:>9.6f}  {k}")

    print("\n[weights] READ THIS:")
    print("  * qwen_vl_interface rel_L2 ~ 0 (cos ~ 1.0) => backbone essentially unchanged")
    print("    by the recon branch => supports A1+A2 (aux gradient did not reshape the trunk).")
    print("  * qwen_vl_interface rel_L2 clearly > 0 => backbone DID move; run --mode hidden")
    print("    to see if the action-relevant representation changed, and if SR is still flat => A4.")
    print("  * NOTE: action_model also differs between runs (different optimization trajectory);")
    print("    the decisive number for A1+A2 is the qwen_vl_interface (backbone) row.")


# ----------------------------------------------------------------------------- #
#  Mode 2: activation drift on hidden_states[-1] (needs one GPU)
# ----------------------------------------------------------------------------- #
def _encode_hidden(model, examples):
    """Return hidden_states[-1] (B,T,H) and qwen input_ids for `examples`, on CPU."""
    model = model.to("cuda").eval()
    with torch.no_grad():
        _ex, qwen_inputs, hidden = model._encode_qwen_hidden(examples)
    out = hidden.detach().float().cpu()
    ids = qwen_inputs["input_ids"].detach().cpu()
    model.to("cpu")
    torch.cuda.empty_cache()
    return out, ids


def _drift_report(h_b, h_r, ids_b, img_token_id):
    assert h_b.shape == h_r.shape, f"hidden shape mismatch {h_b.shape} vs {h_r.shape}"
    B, T, H = h_b.shape
    flat_b = h_b.reshape(B * T, H)
    flat_r = h_r.reshape(B * T, H)
    cos = torch.nn.functional.cosine_similarity(flat_b, flat_r, dim=-1)  # (B*T,)
    rel_l2 = (flat_r - flat_b).norm(dim=-1) / (flat_b.norm(dim=-1) + 1e-12)

    def _line(name, m):
        print(f"  {name:<18} cos: mean={m_cos[m].mean():.6f} min={m_cos[m].min():.6f} "
              f"| rel_L2: mean={m_rel[m].mean():.6f} max={m_rel[m].max():.6f} | n={int(m.sum())}")

    m_cos, m_rel = cos, rel_l2
    print(f"\n[hidden] drift over ALL tokens (B={B}, T={T}, H={H}):")
    print(f"  all                cos: mean={cos.mean():.6f} min={cos.min():.6f} "
          f"| rel_L2: mean={rel_l2.mean():.6f} max={rel_l2.max():.6f} | n={B*T}")

    if img_token_id is not None:
        is_img = (ids_b.reshape(-1) == int(img_token_id))
        if is_img.any():
            _line("image tokens", is_img)
        if (~is_img).any():
            _line("text tokens", ~is_img)
    print("\n[hidden] READ THIS:")
    print("  * cos ~ 1.0 and rel_L2 ~ 0 on image tokens  => the recon-conditioned representation")
    print("    barely moved => A1+A2. If SR is flat, that's why.")
    print("  * cos clearly < 1.0 / rel_L2 clearly > 0, yet eval SR unchanged => A4 (changed but useless).")


def run_hidden(base_path: str, recon_path: str, batch_size: int, num_batches: int, seed: int) -> None:
    from starVLA.model.framework.base_framework import baseframework
    from starVLA.dataloader import build_dataloader

    assert torch.cuda.is_available(), "hidden mode needs a GPU. Pick an idle card via CUDA_VISIBLE_DEVICES."

    print(f"[hidden] loading recon model (its config drives the dataloader)...")
    recon_model = baseframework.from_pretrained(recon_path)
    cfg = recon_model.config
    try:
        cfg.datasets.vla_data.per_device_batch_size = int(batch_size)
    except Exception:
        pass

    print(f"[hidden] building dataloader (mix={cfg.datasets.vla_data.data_mix})...")
    torch.manual_seed(seed)
    loader = build_dataloader(cfg=cfg, dataset_py=cfg.datasets.vla_data.dataset_py)
    it = iter(loader)

    # Grab a FIXED set of example batches ONCE; reuse identically for both models.
    batches = [next(it) for _ in range(max(1, num_batches))]
    print(f"[hidden] captured {len(batches)} batch(es); sizes={[len(b) for b in batches]}")

    img_token_id = getattr(recon_model.qwen_vl_interface, "image_token_id", None)

    # recon first (already loaded), then base. One model on GPU at a time.
    hr, ids_r = [], []
    for b in batches:
        h, ids = _encode_hidden(recon_model, b)
        hr.append(h); ids_r.append(ids)
    del recon_model
    torch.cuda.empty_cache()

    print(f"[hidden] loading baseline model...")
    base_model = baseframework.from_pretrained(base_path)
    hb, ids_b = [], []
    for b in batches:
        h, ids = _encode_hidden(base_model, b)
        hb.append(h); ids_b.append(ids)
    del base_model
    torch.cuda.empty_cache()

    h_b = torch.cat(hb, dim=0)
    h_r = torch.cat(hr, dim=0)
    ids_all = torch.cat(ids_b, dim=0)
    _drift_report(h_b, h_r, ids_all, img_token_id)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=["weights", "hidden"], default="weights")
    ap.add_argument("--base", required=True, help="baseline .pt checkpoint")
    ap.add_argument("--recon", required=True, help="recon-branch .pt checkpoint")
    ap.add_argument("--topk", type=int, default=15, help="[weights] top-N most-changed backbone params")
    ap.add_argument("--batch-size", type=int, default=8, help="[hidden] examples per batch")
    ap.add_argument("--num-batches", type=int, default=1, help="[hidden] number of batches to average over")
    ap.add_argument("--seed", type=int, default=0, help="[hidden] dataloader seed (only picks which samples)")
    args = ap.parse_args()

    if args.mode == "weights":
        run_weights(args.base, args.recon, args.topk)
    else:
        run_hidden(args.base, args.recon, args.batch_size, args.num_batches, args.seed)


if __name__ == "__main__":
    main()
