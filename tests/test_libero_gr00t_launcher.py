from pathlib import Path
import subprocess
import json
import yaml


SCRIPT = Path("examples/LIBERO/train_files/run_uamvla_gr00t_libero_train.sh")
ACCEL_CONFIG = Path("starVLA/config/deepseeds/uamvla_gr00t_zero3.yaml")
DS_CONFIG = Path("starVLA/config/deepseeds/uamvla_gr00t_zero3_ds.json")


def test_uamvla_gr00t_libero_launcher_defaults_to_current_libero_stack():
    text = SCRIPT.read_text()

    assert "uamvla_gr00t_libero.yaml" in text
    assert 'DATA_ROOT_DIR="${DATA_ROOT_DIR:-datasets/libero2uam}"' in text
    assert 'DATA_MIX="${DATA_MIX:-uamvla_libero_all_h8}"' in text
    assert 'BASE_VLM="${BASE_VLM:-ckpt/Qwen3-VL-8B-Instruct}"' in text
    assert 'DEEPSPEED_CONFIG="${DEEPSPEED_CONFIG:-starVLA/config/deepseeds/uamvla_gr00t_zero3.yaml}"' in text
    assert "examples/Gemma4/_make_accelerate_config.py" not in text
    assert 'starVLA/training/train_starvla.py' in text
    assert '"$@"' in text


def test_uamvla_gr00t_libero_launcher_shell_syntax():
    subprocess.run(["bash", "-n", str(SCRIPT)], check=True)


def test_uamvla_gr00t_zero3_config_enables_zero3_init():
    accel_cfg = yaml.safe_load(ACCEL_CONFIG.read_text())
    ds_cfg = json.loads(DS_CONFIG.read_text())

    assert accel_cfg["distributed_type"] == "DEEPSPEED"
    assert accel_cfg["deepspeed_config"]["zero3_init_flag"] is True
    assert accel_cfg["deepspeed_config"]["deepspeed_config_file"] == f"./{DS_CONFIG}"
    assert ds_cfg["bf16"]["enabled"] is True
    assert ds_cfg["fp16"]["enabled"] is False
    assert ds_cfg["zero_optimization"]["stage"] == 3
    assert ds_cfg["zero_optimization"]["stage3_gather_16bit_weights_on_model_save"] is True
