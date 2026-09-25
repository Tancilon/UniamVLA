"""One real CALVIN sample: new DiT forward, optional backward, action inference.

No optimizer step or checkpoint writes. --offload-activations reduces peak
GPU memory by saving activations on CPU and discarding each parameter's
gradient after measuring it.
"""
import argparse
from contextlib import nullcontext
from pathlib import Path

import torch
from omegaconf import OmegaConf

from starVLA.dataloader.lerobot_datasets import make_LeRobotSingleDataset
from starVLA.model.framework.VLM4A.UamVLA_DiT import UamVLA_DiT


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--backward', action='store_true')
    parser.add_argument('--offload-activations', action='store_true')
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.manual_seed(42)
    cfg = OmegaConf.load('examples/calvin/train_files/run_uamvla_DiT_calvin.yaml')
    reader = make_LeRobotSingleDataset(
        Path(cfg.datasets.vla_data.data_root_dir), 'task_ABC_D_scene_A_lerobot_lt',
        'calvin_dit', data_cfg=cfg.datasets.vla_data,
    )
    sample = reader[0]
    assert 'state' not in sample and 'future_rgb' not in sample
    print('Real sample:', sample['lang'], sample['action'].shape, flush=True)
    model = UamVLA_DiT(cfg).to(dtype=torch.bfloat16, device='cuda')
    print('Model on GPU; params:', sum(p.numel() for p in model.parameters()), flush=True)
    gradients = {}
    handles = []
    if args.offload_activations:
        def record_and_release(name):
            def hook(param):
                grad = param.grad
                if not torch.isfinite(grad).all():
                    raise ValueError(f'Nonfinite gradient: {name}')
                group = name.split('.')[0]
                gradients[group] = gradients.get(group, 0) + float(grad.norm())
                param.grad = None
            return hook
        for name, param in model.named_parameters():
            if param.requires_grad:
                handles.append(param.register_post_accumulate_grad_hook(record_and_release(name)))
    model.train()
    context = torch.autograd.graph.save_on_cpu() if args.offload_activations else nullcontext()
    with context:
        loss = model([sample])['action_loss']
        print('loss:', float(loss), flush=True)
        assert torch.isfinite(loss)
        if args.backward:
            loss.backward()
            print('backward complete; gradient norms:', gradients, flush=True)
    for handle in handles:
        handle.remove()
    model.zero_grad(set_to_none=True)
    model.eval()
    output = model.predict_action([sample])['normalized_actions']
    assert output.shape == (1, 8, 7)
    assert torch.isfinite(torch.from_numpy(output)).all()
    print('inference:', output.shape, 'finite:', bool(torch.isfinite(torch.from_numpy(output)).all()), flush=True)


if __name__ == '__main__':
    main()
