"""Unit tests for EnvironmentInfo probe and GPU attributes."""

from mlagent.helper.environment_probe import EnvironmentInfo, environment_to_text


def test_environment_info_has_gpu_and_gpu_memory_gb() -> None:
    # No GPU case
    env_none = EnvironmentInfo()
    assert not env_none.has_gpu
    assert env_none.gpu_memory_gb == 0.0
    assert not env_none.can_train_on_gpu

    # NVIDIA GPU case
    env_nvidia = EnvironmentInfo(
        gpu_name="NVIDIA GeForce RTX 4090",
        gpu_count=1,
        gpu_memory_total_gb=24.0,
        gpu_memory_free_gb=22.5,
        torch_cuda_available=True,
    )
    assert env_nvidia.has_gpu
    assert env_nvidia.gpu_memory_gb == 24.0
    assert env_nvidia.can_train_on_gpu

    # Apple Silicon MPS case
    env_mps = EnvironmentInfo(
        torch_mps_available=True,
    )
    assert env_mps.has_gpu
    assert env_mps.can_train_on_gpu


def test_environment_to_text() -> None:
    env = EnvironmentInfo(
        operating_system="Windows 11",
        shell_name="powershell",
        python_version="3.12.0",
        cpu_count=8,
        ram_total_gb=32.0,
        ram_free_gb=16.0,
        gpu_name="RTX 4090",
        gpu_count=1,
        gpu_memory_total_gb=24.0,
        gpu_memory_free_gb=20.0,
    )
    text = environment_to_text(env)
    assert "RTX 4090" in text
    assert "24.0 GB total" in text
