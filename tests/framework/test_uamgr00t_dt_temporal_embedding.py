from __future__ import annotations

import contextlib
import importlib.util
import sys
import types
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml


class _AttrDict(dict):
    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError as exc:
            raise AttributeError(key) from exc


class _CountingEmbedding(nn.Embedding):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.forward_calls = 0

    def forward(self, input):
        self.forward_calls += 1
        return super().forward(input)


def _reference_compressed_inputs(
    input_ids,
    attention_mask,
    compressed_vis,
    embed_fn,
    n_orig,
    n_comp,
    num_views,
    state_embs,
    temporal_frame_embs,
):
    vis_start, vis_end = 151652, 151653
    batches = []
    for b in range(input_ids.shape[0]):
        ids = input_ids[b]
        mask = attention_mask[b]
        parts = []
        img_idx = 0
        i = 0
        while i < ids.shape[0]:
            if mask[i].item() == 0:
                i += 1
                continue
            if ids[i].item() != vis_start:
                parts.append(embed_fn(ids[i : i + 1]))
                i += 1
                continue

            frame_idx = img_idx // num_views
            frame_emb = temporal_frame_embs[frame_idx]
            if state_embs is not None and img_idx % num_views == 0 and frame_idx < state_embs.shape[1]:
                parts.append((state_embs[b, frame_idx] + frame_emb).unsqueeze(0))
            parts.append(embed_fn(ids[i : i + 1]))
            i += n_orig + 1
            parts.append(compressed_vis[b, img_idx] + frame_emb.unsqueeze(0))
            img_idx += 1
            if i < ids.shape[0] and ids[i].item() == vis_end:
                parts.append(embed_fn(ids[i : i + 1]))
                i += 1
        batches.append(torch.cat(parts))

    max_len = max(batch.shape[0] for batch in batches)
    padded = []
    masks = []
    for batch in batches:
        pad = max_len - batch.shape[0]
        padded.append(torch.cat([batch.new_zeros(pad, batch.shape[-1]), batch], dim=0))
        masks.append(torch.cat([torch.zeros(pad, dtype=torch.long), torch.ones(batch.shape[0], dtype=torch.long)]))
    return torch.stack(padded), torch.stack(masks)


def _load_dt_module(monkeypatch):
    class _Registry:
        def register(self, _name):
            return lambda cls: cls

    class _Base(nn.Module):
        def __init__(self, config):
            super().__init__()
            self.config = config
            model_config = types.SimpleNamespace(hidden_size=8)
            self.qwen_vl_interface = types.SimpleNamespace(model=types.SimpleNamespace(config=model_config))

    class _FutureDiTBranch(nn.Module):
        def __init__(self, **_kwargs):
            super().__init__()

    class _PerceiverResampler(nn.Module):
        def __init__(self, **_kwargs):
            super().__init__()

    base = types.ModuleType("starVLA.model.framework.VLM4A.UamGR00T")
    base.UamVLAGR00T = _Base
    monkeypatch.setitem(sys.modules, "starVLA.model.framework.VLM4A.UamGR00T", base)

    future = types.ModuleType("starVLA.model.modules.uamvla.components.future_dit_branch")
    future.FutureDiTBranch = _FutureDiTBranch
    monkeypatch.setitem(sys.modules, "starVLA.model.modules.uamvla.components.future_dit_branch", future)

    resampler = types.ModuleType("starVLA.model.modules.uamvla.components.perceiver_resampler")
    resampler.PerceiverResampler = _PerceiverResampler
    monkeypatch.setitem(sys.modules, "starVLA.model.modules.uamvla.components.perceiver_resampler", resampler)

    tools = types.ModuleType("starVLA.model.tools")
    tools.FRAMEWORK_REGISTRY = _Registry()
    monkeypatch.setitem(sys.modules, "starVLA.model.tools", tools)

    path = Path("starVLA/model/framework/VLM4A/UamGR00T_DT.py")
    spec = importlib.util.spec_from_file_location("uamgr00t_dt_temporal_under_test", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_visual_resampler_builds_zero_initialized_temporal_embeddings(monkeypatch):
    module = _load_dt_module(monkeypatch)
    config = _AttrDict(
        framework=_AttrDict(
            history_frames=2,
            state_dim=7,
            future_branch={},
            visual_resampler=_AttrDict(enabled=True, num_orig_tokens=2, num_compressed_tokens=1),
        ),
        datasets=_AttrDict(vla_data=_AttrDict(obs=["video.primary", "video.wrist"])),
        trainer=_AttrDict(
            learning_rate=_AttrDict(base=2.5e-5, temporal_frame_embedding=1.0e-4),
            freeze_modules="",
        ),
    )

    model = module.UamVLAGR00T_DT(config)

    assert model.temporal_frame_embedding.num_embeddings == 2
    assert model.temporal_frame_embedding.embedding_dim == 8
    assert model.temporal_frame_embedding.weight.requires_grad
    torch.testing.assert_close(model.temporal_frame_embedding.weight, torch.zeros(2, 8))

    accelerate = types.ModuleType("accelerate")
    accelerate_logging = types.ModuleType("accelerate.logging")
    accelerate_logging.get_logger = lambda _name: types.SimpleNamespace()
    monkeypatch.setitem(sys.modules, "accelerate", accelerate)
    monkeypatch.setitem(sys.modules, "accelerate.logging", accelerate_logging)
    from starVLA.training.trainer_utils.trainer_tools import build_param_lr_groups

    groups = build_param_lr_groups(model, config)
    temporal_group = next(group for group in groups if group["name"] == "temporal_frame_embedding")
    assert temporal_group["lr"] == 1.0e-4
    assert temporal_group["params"] == [model.temporal_frame_embedding.weight]


def test_compressed_sequence_broadcasts_temporal_embedding_only_to_frame_content(monkeypatch):
    module = _load_dt_module(monkeypatch)
    vis_start, vis_end = 151652, 151653
    input_ids = torch.tensor(
        [[10, vis_start, 99, 99, vis_end, vis_start, 99, 99, vis_end,
          vis_start, 99, 99, vis_end, vis_start, 99, 99, vis_end, 11]]
    )
    attention_mask = torch.ones_like(input_ids)
    embed_fn = nn.Embedding(151700, 2)
    with torch.no_grad():
        embed_fn.weight.zero_()
        embed_fn.weight[10] = torch.tensor([10.0, 10.0])
        embed_fn.weight[11] = torch.tensor([11.0, 11.0])
        embed_fn.weight[vis_start] = torch.tensor([1.0, 1.0])
        embed_fn.weight[vis_end] = torch.tensor([2.0, 2.0])

    compressed = torch.tensor(
        [[[[100.0, 100.0]], [[200.0, 200.0]], [[300.0, 300.0]], [[400.0, 400.0]]]]
    )
    states = torch.tensor([[[10.0, 20.0], [30.0, 40.0]]])
    temporal = nn.Embedding(2, 2)
    with torch.no_grad():
        temporal.weight.copy_(torch.tensor([[1.0, 2.0], [3.0, 4.0]]))

    output, mask = module.UamVLAGR00T_DT._build_compressed_inputs_embeds(
        input_ids=input_ids,
        attention_mask=attention_mask,
        compressed_vis=compressed,
        embed_fn=embed_fn,
        n_orig=2,
        n_comp=1,
        num_views=2,
        state_embs=states,
        temporal_frame_embs=temporal.weight,
        device=torch.device("cpu"),
    )

    expected = torch.tensor(
        [
            [
                [10.0, 10.0],
                [11.0, 22.0],
                [1.0, 1.0],
                [101.0, 102.0],
                [2.0, 2.0],
                [1.0, 1.0],
                [201.0, 202.0],
                [2.0, 2.0],
                [33.0, 44.0],
                [1.0, 1.0],
                [303.0, 304.0],
                [2.0, 2.0],
                [1.0, 1.0],
                [403.0, 404.0],
                [2.0, 2.0],
                [11.0, 11.0],
            ]
        ]
    )
    torch.testing.assert_close(output, expected)
    torch.testing.assert_close(mask, torch.ones_like(mask))

    output.sum().backward()
    torch.testing.assert_close(temporal.weight.grad, torch.full((2, 2), 3.0))


def test_compressed_sequence_matches_reference_with_one_embedding_call(monkeypatch):
    module = _load_dt_module(monkeypatch)
    vis_start, vis_end = 151652, 151653
    span = [vis_start, 99, 99, vis_end]
    input_ids = torch.tensor(
        [
            [0, 0, 10, *span, *span, *span, *span, 11],
            [20, 21, 22, *span, *span, *span, *span, 23],
        ]
    )
    attention_mask = torch.tensor(
        [
            [0, 0, *([1] * 18)],
            [*([1] * 20)],
        ]
    )
    torch.manual_seed(7)
    reference_embed = nn.Embedding(151700, 3)
    optimized_embed = _CountingEmbedding(151700, 3)
    optimized_embed.load_state_dict(reference_embed.state_dict())

    compressed = torch.randn(2, 4, 1, 3, requires_grad=True)
    states = torch.randn(2, 2, 3, requires_grad=True)
    temporal = torch.randn(2, 3, requires_grad=True)
    optimized_compressed = compressed.detach().clone().requires_grad_()
    optimized_states = states.detach().clone().requires_grad_()
    optimized_temporal = temporal.detach().clone().requires_grad_()

    expected, expected_mask = _reference_compressed_inputs(
        input_ids,
        attention_mask,
        compressed,
        reference_embed,
        n_orig=2,
        n_comp=1,
        num_views=2,
        state_embs=states,
        temporal_frame_embs=temporal,
    )
    actual, actual_mask = module.UamVLAGR00T_DT._build_compressed_inputs_embeds(
        input_ids=input_ids,
        attention_mask=attention_mask,
        compressed_vis=optimized_compressed,
        embed_fn=optimized_embed,
        n_orig=2,
        n_comp=1,
        num_views=2,
        state_embs=optimized_states,
        temporal_frame_embs=optimized_temporal,
        device=torch.device("cpu"),
    )

    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(actual_mask, expected_mask)
    assert optimized_embed.forward_calls == 1

    weights = torch.randn_like(expected)
    (expected * weights).sum().backward()
    (actual * weights).sum().backward()
    torch.testing.assert_close(optimized_embed.weight.grad, reference_embed.weight.grad)
    torch.testing.assert_close(optimized_compressed.grad, compressed.grad)
    torch.testing.assert_close(optimized_states.grad, states.grad)
    torch.testing.assert_close(optimized_temporal.grad, temporal.grad)


def test_future_gt_tokens_batch_views_into_one_processor_and_visual_call(monkeypatch):
    module = _load_dt_module(monkeypatch)

    class _ImageProcessor:
        def __init__(self):
            self.calls = 0

        def __call__(self, images, return_tensors):
            self.calls += 1
            assert return_tensors == "pt"
            values = torch.tensor([np.asarray(image)[0, 0, 0] for image in images], dtype=torch.float32)
            return {
                "pixel_values": values[:, None],
                "image_grid_thw": torch.tensor([[1, 14, 14]] * len(images)),
            }

    class _Visual(nn.Module):
        spatial_merge_size = 2

        def __init__(self):
            super().__init__()
            self.dtype_anchor = nn.Parameter(torch.zeros(()), requires_grad=False)
            self.calls = 0

        def forward(self, pixel_values, grid_thw):
            self.calls += 1
            tokens = pixel_values[:, None, :].expand(-1, 49, -1).reshape(-1, 1)
            return tokens, []

    processor = _ImageProcessor()
    visual = _Visual()
    model = types.SimpleNamespace(
        qwen_vl_interface=types.SimpleNamespace(
            model=types.SimpleNamespace(model=types.SimpleNamespace(visual=visual)),
            processor=types.SimpleNamespace(image_processor=processor),
        )
    )
    rgb_list = [
        torch.full((2, 3, 2, 2), 10 / 255.0),
        torch.full((2, 3, 1, 1), 20 / 255.0),
    ]
    rgb_list[0][1].fill_(11 / 255.0)
    rgb_list[1][1].fill_(21 / 255.0)

    results = module.UamVLAGR00T_DT._extract_gt_image_tokens(
        model,
        rgb_list,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )

    assert processor.calls == 1
    assert visual.calls == 1
    assert len(results) == 2
    assert [tuple(result.shape) for result in results] == [(2, 49, 1), (2, 49, 1)]
    torch.testing.assert_close(results[0][:, 0, 0], torch.tensor([10.0, 11.0]))
    torch.testing.assert_close(results[1][:, 0, 0], torch.tensor([20.0, 21.0]))


def test_calvin_temporal_embedding_uses_new_module_learning_rate():
    config = yaml.safe_load(Path("examples/calvin/train_files/run_uamgr00t_DT_calvin_finetune.yaml").read_text())

    assert config["framework"]["visual_resampler"]["enabled"] is True
    assert config["framework"]["visual_resampler"]["num_compressed_tokens"] == 16
    assert config["framework"]["history_frames"] == 10
    assert config["trainer"]["learning_rate"]["temporal_frame_embedding"] == 1.0e-4


def test_dt_routes_state_through_qwen_only(monkeypatch):
    module = _load_dt_module(monkeypatch)
    monkeypatch.setattr(module.torch, "autocast", lambda *args, **kwargs: contextlib.nullcontext())
    config = _AttrDict(
        framework=_AttrDict(
            history_frames=2,
            state_dim=2,
            future_branch={},
            visual_resampler=_AttrDict(enabled=True, num_orig_tokens=2, num_compressed_tokens=1),
            action_model=_AttrDict(state_dim=0, repeated_diffusion_steps=1),
        ),
        datasets=_AttrDict(vla_data=_AttrDict(obs=["video.primary", "video.wrist"])),
    )
    model = module.UamVLAGR00T_DT(config)
    model.action_horizon = 1

    qwen_states = []

    def _encode(examples, state):
        qwen_states.append(state)
        batch_size = len(examples)
        hidden = torch.ones(batch_size, 4, 8)
        future_hidden = torch.ones(batch_size, model._n_future, 8)
        return examples, {"attention_mask": torch.ones(batch_size, 4)}, hidden, future_hidden

    class _ActionModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.forward_state = object()
            self.predict_state = object()

        def forward(self, _hidden, _actions, state, encoder_attention_mask=None):
            self.forward_state = state
            return torch.tensor(1.0)

        def predict_action(self, hidden, state, encoder_attention_mask=None):
            self.predict_state = state
            return torch.zeros(hidden.shape[0], 1, 2)

    action_model = _ActionModel()
    model.action_model = action_model
    model._prepare_examples = lambda examples: examples
    model._encode_qwen_hidden = _encode
    model._encoder_attention_mask = lambda _inputs: None

    examples = [
        {
            "state": np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
            "action": np.zeros((1, 2), dtype=np.float32),
        }
    ]

    model.forward(examples)
    model.predict_action(examples)

    assert len(qwen_states) == 2
    torch.testing.assert_close(
        qwen_states[0],
        torch.tensor([[[1.0, 2.0], [3.0, 4.0]]]),
    )
    assert action_model.forward_state is None
    assert action_model.predict_state is None


def test_all_dt_configs_disable_direct_action_state_path():
    config_paths = [
        Path("examples/calvin/train_files/run_uamgr00t_DT_calvin_finetune.yaml"),
        Path("examples/Robotwin/train_files/run_uamgr00t_DT_robotwin_pretrain.yaml"),
        Path("examples/Robotwin/train_files/run_uamgr00t_DT_robotwin_finetune.yaml"),
    ]

    for path in config_paths:
        config = yaml.safe_load(path.read_text())
        assert config["framework"]["state_dim"] > 0, path
        assert config["framework"]["action_model"]["state_dim"] == 0, path
