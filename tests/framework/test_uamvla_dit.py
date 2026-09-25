"""Real Qwen3-VL (tiny backbone) + production DiT-B wiring tests."""
import importlib

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf
from PIL import Image
from transformers import AutoProcessor, Qwen3VLConfig, Qwen3VLForConditionalGeneration

from starVLA.model.modules.vlm.QWen3 import _QWen3_VL_Interface, _fix_processor_pixel_budget
from starVLA.training.trainer_utils.trainer_tools import build_param_lr_groups
from starVLA.training.trainer_utils.config_tracker import wrap_config
from starVLA.model.framework.base_framework import build_framework

RECIPE = 'examples/calvin/train_files/run_uamvla_DiT_calvin.yaml'


def tiny_interface(config):
    interface = _QWen3_VL_Interface.__new__(_QWen3_VL_Interface)
    torch.nn.Module.__init__(interface)
    interface.config = config
    interface.processor = AutoProcessor.from_pretrained('ckpt/RynnBrain-CoP-8B', local_files_only=True)
    interface.processor.tokenizer.padding_side = 'left'
    interface.fixed_image_pixels = _fix_processor_pixel_budget(interface.processor, int(config.framework.qwen_image_size))
    qconfig = Qwen3VLConfig(
        text_config=dict(vocab_size=151936, hidden_size=64, intermediate_size=128,
                         num_hidden_layers=3, num_attention_heads=4, num_key_value_heads=4,
                         head_dim=16, rope_scaling=dict(rope_type='default', mrope_section=[2, 3, 3],
                                                      mrope_interleaved=True)),
        vision_config=dict(depth=3, hidden_size=64, intermediate_size=128, num_heads=4,
                           out_hidden_size=64, patch_size=16, spatial_merge_size=2,
                           temporal_patch_size=2, deepstack_visual_indexes=[0, 1, 2]),
        image_token_id=151655, video_token_id=151656,
        vision_start_token_id=151652, vision_end_token_id=151653,
    )
    qconfig._attn_implementation = 'sdpa'
    interface.model = Qwen3VLForConditionalGeneration(qconfig)
    return interface


@pytest.fixture(scope='module')
def model():
    torch.set_num_threads(4)
    module = importlib.import_module('starVLA.model.framework.VLM4A.UamVLA_DiT')
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(module, 'get_vlm_model', tiny_interface)
        config = OmegaConf.load(RECIPE)
        config.framework.action_model.action_hidden_dim = 4096
        instance = build_framework(wrap_config(config))
        assert isinstance(instance, module.UamVLA_DiT)
        assert instance.config.framework.action_model.action_hidden_dim == 768
        assert instance.action_condition_projector.out_features == instance.action_model.token_size == 768
    yield instance.to('cuda' if torch.cuda.is_available() else 'cpu')


def examples():
    rng = np.random.default_rng(3)
    pair = [Image.fromarray(rng.integers(0, 256, (224, 224, 3), dtype=np.uint8)) for _ in range(2)]
    return [dict(image=pair, image_history=[pair] * 5, step=30,
                 lang=lang, action=rng.normal(size=(8, 7)).astype(np.float32))
            for lang in ('push the drawer', 'move the red block to the left side of the blue block')]


def test_condition_padding_and_two_step_backward(model):
    model.train()
    batch = examples()
    prepared, inputs, hidden = model._encode_qwen_hidden(model._prepare_examples(batch))
    assert inputs['image_grid_thw'].tolist() == [[1, 16, 16]] * 4
    assert ((inputs['input_ids'] == 151655).sum(dim=1) == 128).all()
    assert (inputs['attention_mask'] == 0).any()
    condition = model._action_condition(inputs, hidden)
    assert condition.shape == (2, 8, 768)
    for index in range(2):
        positions = (inputs['input_ids'][index] == model.action_token_id).nonzero().flatten()[-8:]
        expected = model.action_condition_projector(hidden[index, positions].to(model.action_condition_projector.weight.dtype))
        torch.testing.assert_close(condition[index], expected)
    del hidden, condition, inputs
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
    for _ in range(2):
        optimizer.zero_grad(set_to_none=True)
        loss = model(batch)['action_loss']
        assert torch.isfinite(loss)
        loss.backward()
        optimizer.step()
    for module in (model.qwen_vl_interface, model.history_fusion, model.action_condition_projector, model.action_model):
        grads = [p.grad for p in module.parameters() if p.grad is not None]
        assert grads and all(torch.isfinite(g).all() for g in grads)
        assert any(g.abs().max() > 0 for g in grads)
    assert model.action_model.net.num_cond_tokens == 8
    assert model.repeated_diffusion_steps == 4
    assert model.cfg_scale == 1.5
    assert model.action_model.diffusion_steps == 100
    assert model.action_model.noise_schedule == 'squaredcos_cap_v2'
    assert len(model.action_model.net.blocks) == 12
    assert model.action_model.net.positional_embedding.shape == (16, 768)
    assert not any('future' in name or 'state_encoder' in name for name, _ in model.named_parameters())
    model.zero_grad(set_to_none=True)


def test_sampling_cfg_and_ddim_cache(model):
    model.eval()
    device = next(model.parameters()).device
    with torch.inference_mode():
        condition = torch.randn(1, 8, 768, device=device)
        for steps, scale in ((10, 1.5), (20, 1.5), (10, 1.0)):
            output = model._sample_dit_actions(condition, cfg_scale=scale, num_ddim_steps=steps)
            assert output.shape == (1, 8, 7) and torch.isfinite(output).all()
            assert model.action_model.ddim_diffusion.num_timesteps == steps
        output = model.predict_action(examples()[:1])['normalized_actions']
        assert output.shape == (1, 8, 7) and np.isfinite(output).all()
        model.to(torch.bfloat16)
        for scale in (1.0, 1.5):
            output = model._sample_dit_actions(condition, cfg_scale=scale)
            assert torch.isfinite(output).all()
        model.float()


def test_lr_groups_and_strict_roundtrip(model, tmp_path):
    groups = build_param_lr_groups(model, model.config)
    rates = {g['name']: g['lr'] for g in groups}
    assert rates['action_model'] == 5e-5
    assert rates['qwen_vl_interface'] == rates['history_fusion'] == 5e-6
    assert rates['base'] == 1e-5
    assert 'action_condition_projector' not in rates
    base_params = {id(p) for g in groups if g['name'] == 'base' for p in g['params']}
    assert {id(p) for p in model.action_condition_projector.parameters()} <= base_params
    params = [id(p) for g in groups for p in g['params']]
    assert len(params) == len(set(params))
    assert set(params) == {id(p) for p in model.parameters() if p.requires_grad}
    model.config.save_accessed_config(tmp_path / 'config.yaml', use_original_values=False)
    saved_config = OmegaConf.load(tmp_path / 'config.yaml')
    assert saved_config.framework.name == 'UamVLA_DiT'
    assert saved_config.framework.action_model.n_condition_token == 8
    assert saved_config.framework.action_model.action_hidden_dim == 768
    checkpoint = tmp_path / 'model.pt'
    torch.save(model.state_dict(), checkpoint)
    state = torch.load(checkpoint, map_location='cpu', weights_only=True)
    module = importlib.import_module('starVLA.model.framework.VLM4A.UamVLA_DiT')
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(module, 'get_vlm_model', tiny_interface)
        restored = build_framework(saved_config)
    restored.load_state_dict(state, strict=True)
    assert restored.cfg_scale == 1.5 and restored.num_inference_timesteps == 10
    assert restored.action_horizon == 8
    assert restored.repeated_diffusion_steps == 4
    assert restored.action_model.net.num_cond_tokens == restored.action_horizon
    assert restored.action_model.diffusion_steps == 100
    assert restored.action_model.noise_schedule == 'squaredcos_cap_v2'
    assert restored.image_size == 224 and restored.qwen_image_size == 256
    assert restored.patches_per_view == 64


def test_dataset_offsets_and_no_labels():
    from starVLA.dataloader.gr00t_lerobot.registry import ROBOT_TYPE_CONFIG_MAP
    from starVLA.dataloader.gr00t_lerobot.datasets import LeRobotSingleDataset
    config = OmegaConf.load(RECIPE).datasets.vla_data
    modalities = ROBOT_TYPE_CONFIG_MAP['calvin_dit'].modality_config()
    assert modalities['video'].delta_indices == [-25, -20, -15, -10, -5, 0]
    reader = LeRobotSingleDataset.__new__(LeRobotSingleDataset)
    reader.data_cfg = config
    reader._modality_keys = {k: list(v.modality_keys) for k, v in modalities.items()}
    data = {key: np.zeros((6, 256, 256, 3), dtype=np.uint8) for key in reader._modality_keys['video']}
    data.update({key: np.zeros((8, 1), dtype=np.float32) for key in reader._modality_keys['action']})
    data[reader._modality_keys['language'][0]] = ['push the drawer']
    packed = reader._pack_sample(data)
    assert 'state' not in packed and 'future_rgb' not in packed
    assert len(packed['image_history']) == 5 and packed['action'].shape == (8, 7)


def test_shared_action_head_default_conditions_and_bf16_training():
    from starVLA.model.modules.action_model.DiTActionHeader import ActionModel
    head = ActionModel(384, 'DiT-S', 7, 7, 0).to(torch.bfloat16)
    assert head.net.num_cond_tokens == 64
    target = torch.randn(2, 8, 7, dtype=torch.bfloat16)
    condition = torch.randn(2, 64, 384, dtype=torch.bfloat16, requires_grad=True)
    prediction, noise, _ = head(target, condition)
    assert prediction.shape == target.shape
    loss = head.loss(prediction.float(), noise.float())
    loss.backward()
    assert torch.isfinite(loss)
    assert all(torch.isfinite(p.grad).all() for p in head.parameters() if p.grad is not None)


def test_processor_matches_original_for_current_and_history(model):
    prepared = model._prepare_examples(examples()[:1])
    images = prepared[0]['image'] + [im for pair in prepared[0]['image_history'] for im in pair]
    assert len(images) == 12 and all(im.size == (224, 224) for im in images)
    original = AutoProcessor.from_pretrained('ckpt/RynnBrain-CoP-8B', local_files_only=True)
    expected = original.image_processor(images=images, return_tensors='pt')
    actual = model.qwen_vl_interface.processor.image_processor(images=images, return_tensors='pt')
    assert actual['image_grid_thw'].tolist() == [[1, 16, 16]] * 12
    torch.testing.assert_close(actual['pixel_values'], expected['pixel_values'], rtol=0, atol=0)
    seen = []

    def record_shapes(module, args):
        history, current, _ = args
        seen.append((tuple(history.shape), tuple(current.shape)))

    handle = model.history_fusion.register_forward_pre_hook(record_shapes)
    try:
        with torch.inference_mode():
            model._encode_qwen_hidden(prepared)
    finally:
        handle.remove()
    assert seen == [((1, 3, 5, 2, 64, 64), (1, 3, 2, 64, 64))]
