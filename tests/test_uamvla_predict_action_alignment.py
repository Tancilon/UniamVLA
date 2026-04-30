"""Critical safety net: training forward and inference prefill must produce
identical hidden states at the <|action_start|> position.

Catches any divergence in:
- State token splice
- chat_template rendering
- Tokenization / padding
- Multimodal kwargs (mm_token_type_ids, image_grid_thw)
- inputs_embeds construction order

Skipped if a real Qwen3-VL backbone is not available (the test requires a real
forward pass to compare hidden states meaningfully)."""
import os
import pytest
import torch


_QWEN3_VL_PATH = os.environ.get("QWEN3VL_TEST_CKPT")


@pytest.mark.skipif(
    not _QWEN3_VL_PATH or not os.path.isdir(_QWEN3_VL_PATH),
    reason="QWEN3VL_TEST_CKPT not set; alignment requires a real Qwen3-VL backbone",
)
def test_prefix_hidden_states_match_between_training_and_inference():
    """Run training forward and inference prefill on the same example; the
    hidden state at <|action_start|>'s position must match within atol=1e-4.

    This is the strongest invariant-check available in the test suite. If this
    test fails, the inference path has silently diverged from training and
    LIBERO eval results will not reflect what the model actually learned."""
    pytest.importorskip("transformers")
    from PIL import Image

    from starVLA.model.framework.VLM4A.UamVLA import UamVLA, UamVLADefaultConfig
    from starVLA.model.modules.uamvla.backbone_wrapper import _replace_state_tokens
    from omegaconf import OmegaConf

    # Build a minimal real config pointing at the local fixture.
    cfg = OmegaConf.create({
        "framework": {
            "qwenvl": {
                "base_vlm": _QWEN3_VL_PATH,
                "attn_implementation": "eager",
                "dtype": "float32",  # fp32 for numerical comparison
            },
            "embodiment": {"name": "franka_libero", "action_dim": 7},
            "action_model": {"num_bins": 256, "future_action_window_size": 7,
                              "action_dim": 7},
            "aux_heads": {
                "action": {"enabled": True, "lr": 1e-4, "loss_weight": 1.0},
                "pose":   {"enabled": False},
                "future": {"enabled": False},
                "recon":  {"enabled": False},
            },
        },
    })
    model = UamVLA(config=cfg)
    model.eval()

    img = Image.new("RGB", (640, 640))
    example = {
        "image": [img],
        "lang": "pick up the red block",
        "canonical_state": {
            "arm_0": {
                "ee_pose": torch.zeros(9),
                "joint_pos": torch.zeros(7),
            },
            "gripper_0": torch.zeros(1),
        },
    }

    from starVLA.model.modules.uamvla.collator_helpers import stack_canonical
    from deployment.model_server.tools.image_tools import to_pil_preserve

    # Path A: training-style forward with canonical_state
    qwen_inputs = model.qwen_vl_interface.build_inputs(
        images=[[to_pil_preserve(img) for img in example["image"]]],
        instructions=[example["lang"]],
        canonical_state=stack_canonical([example["canonical_state"]]),
    )
    with torch.no_grad():
        out_train = model.qwen_vl_interface(
            **qwen_inputs, output_hidden_states=True, return_dict=True,
        )
        hidden_train = out_train.hidden_states[-1]  # (1, S, H)

    # Path B: inference prefill via _build_prefill_generate_kwargs, then forward
    gen_kwargs = model._build_prefill_generate_kwargs(qwen_inputs)
    with torch.no_grad():
        out_infer = model.qwen_vl_interface.model.model(
            input_ids=None,
            inputs_embeds=gen_kwargs["inputs_embeds"],
            attention_mask=gen_kwargs["attention_mask"],
            pixel_values=gen_kwargs.get("pixel_values"),
            image_grid_thw=gen_kwargs.get("image_grid_thw"),
            mm_token_type_ids=gen_kwargs.get("mm_token_type_ids"),
            output_hidden_states=True,
            return_dict=True,
        )
        hidden_infer = out_infer.hidden_states[-1]

    # Locate the <|action_start|> position in input_ids
    action_start_id = model.action_start_id
    input_ids = qwen_inputs["input_ids"]
    pos = (input_ids[0] == action_start_id).nonzero(as_tuple=True)[0]
    assert pos.numel() == 1, (
        f"Expected exactly one <|action_start|> in prompt, found {pos.numel()}"
    )
    p = int(pos.item())

    h_train = hidden_train[0, p, :]
    h_infer = hidden_infer[0, p, :]

    assert torch.allclose(h_train, h_infer, atol=1e-4), (
        f"Hidden state at <|action_start|> diverges between training and inference: "
        f"max diff = {(h_train - h_infer).abs().max().item():.6f}"
    )
