from __future__ import annotations

from pathlib import Path


def test_smoke_auxvla_reconvla_ar_recon_no_save_checks_ar_recon_contract():
    script = Path("tools/probes/smoke_auxvla_reconvla_ar_recon_no_save.py").read_text(
        encoding="utf-8",
    )

    assert "training_mode = \"reconvla_ar_recon\"" in script
    assert "disable_internal_recon_loss = False" in script
    assert "train_mm_inv_projector = True" in script
    assert "all-linear-except-frozen" in script
    assert "lm_head_lora_trainable_params" in script
    assert "vision_tower_lora_trainable_params" in script
    assert "pixel_decoder_lora_trainable_params" in script
    assert "action_token_count == 35" in script
    assert "internal ReconVLA vm_loss was not returned" in script
    assert "lm_head_lora_grad_norm.item() > 0" in script
    assert "action_grad_count == 0" in script
    assert "SMOKE_OK" in script
