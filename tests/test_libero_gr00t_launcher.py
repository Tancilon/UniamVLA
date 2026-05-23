from pathlib import Path
import subprocess


SCRIPT = Path("examples/LIBERO/train_files/run_uamvla_gr00t_libero_train.sh")


def test_uamvla_gr00t_libero_launcher_defaults_to_current_libero_stack():
    text = SCRIPT.read_text()

    assert "uamvla_gr00t_libero.yaml" in text
    assert 'DATA_ROOT_DIR="${DATA_ROOT_DIR:-datasets/libero2uam}"' in text
    assert 'DATA_MIX="${DATA_MIX:-uamvla_libero_all_h8}"' in text
    assert 'BASE_VLM="${BASE_VLM:-ckpt/Qwen3-VL-8B-Instruct}"' in text
    assert 'ZERO_STAGE="${ZERO_STAGE:-3}"' in text
    assert "examples/Gemma4/_make_accelerate_config.py" in text
    assert 'starVLA/training/train_starvla.py' in text
    assert '"$@"' in text


def test_uamvla_gr00t_libero_launcher_shell_syntax():
    subprocess.run(["bash", "-n", str(SCRIPT)], check=True)
