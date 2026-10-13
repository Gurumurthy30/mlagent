import platform

from mlagent.environment_probe import (
    EnvironmentInfo,
    can_connect,
    environment_to_text,
    get_shell_name,
    load_environment,
    parse_nvidia_smi_output,
    parse_torch_probe_output,
    probe_environment,
    save_environment,
)


def test_real_probe_on_this_machine_returns_sane_values():
    environment = probe_environment()

    assert environment.python_version == platform.python_version()
    assert environment.cpu_count > 0
    assert environment.ram_total_gb > 0
    assert "numpy" in environment.installed_packages
    assert environment.shell_name == get_shell_name()
    assert set(environment.installed_packages).isdisjoint(environment.missing_packages)


def test_parse_nvidia_smi_output_for_one_gpu():
    gpus = parse_nvidia_smi_output("NVIDIA GeForce RTX 4090, 24564, 18432\n")

    assert gpus == [("NVIDIA GeForce RTX 4090", 24.0, 18.0)]


def test_parse_nvidia_smi_output_for_two_gpus_and_a_comma_in_the_name():
    output_text = "Tesla T4, 15360, 15000\nWeird, Name GPU, 8192, 4096\n"

    gpus = parse_nvidia_smi_output(output_text)

    assert [gpu[0] for gpu in gpus] == ["Tesla T4", "Weird, Name GPU"]
    assert gpus[1][1:] == (8.0, 4.0)


def test_parse_nvidia_smi_output_skips_lines_it_cannot_read():
    assert parse_nvidia_smi_output("GPU, [N/A], [N/A]\n") == []
    assert parse_nvidia_smi_output("") == []


def test_parse_torch_probe_output_uses_the_last_line_and_survives_noise():
    output_text = 'UserWarning: something\n{"version": "2.5.0", "cuda_available": true}\n'

    assert parse_torch_probe_output(output_text)["cuda_available"] is True
    assert parse_torch_probe_output("") == {}
    assert parse_torch_probe_output("not json at all") == {}
    assert parse_torch_probe_output("{}") == {}


def test_can_connect_is_false_for_a_closed_port():
    assert can_connect("localhost", port=1, timeout_seconds=1.0) is False


def build_gpu_environment(**changes) -> EnvironmentInfo:
    values = {
        "operating_system": "Windows 11",
        "shell_name": "powershell",
        "python_version": "3.12.3",
        "cpu_count": 16,
        "ram_total_gb": 32.0,
        "ram_free_gb": 20.0,
        "gpu_name": "NVIDIA GeForce RTX 4090",
        "gpu_count": 1,
        "gpu_memory_total_gb": 24.0,
        "gpu_memory_free_gb": 12.0,
        "torch_version": "2.5.0+cu124",
        "torch_cuda_available": True,
        "torch_cuda_version": "12.4",
        "can_reach_pypi": True,
        "installed_packages": {"pandas": "2.3.3"},
        "missing_packages": ["librosa"],
    }
    values.update(changes)
    return EnvironmentInfo(**values)


def test_text_for_a_machine_that_can_train_on_gpu():
    text = environment_to_text(build_gpu_environment())

    assert "OS: Windows 11 (shell: powershell)" in text
    assert "NVIDIA GeForce RTX 4090" in text
    assert "can train on GPU (CUDA 12.4)" in text
    assert "pandas 2.3.3" in text
    assert "Not installed: librosa" in text
    assert "NOTE" not in text


def test_text_warns_when_a_gpu_exists_but_torch_is_cpu_only():
    environment = build_gpu_environment(torch_cuda_available=False, torch_version="2.5.0+cpu")

    text = environment_to_text(environment)

    assert "CPU only" in text
    assert "NOTE: a GPU exists but this PyTorch build cannot use it" in text
    assert environment.can_train_on_gpu is False


def test_text_for_a_machine_without_gpu_or_torch():
    environment = build_gpu_environment(gpu_name="", gpu_count=0, torch_version="")

    text = environment_to_text(environment)

    assert "GPU: none found" in text
    assert "PyTorch: not installed" in text


def test_apple_silicon_counts_as_gpu_training():
    environment = build_gpu_environment(
        gpu_name="", torch_cuda_available=False, torch_mps_available=True
    )

    assert environment.can_train_on_gpu is True
    assert "Apple Silicon (MPS)" in environment_to_text(environment)


def test_environment_file_round_trip(tmp_path):
    environment = build_gpu_environment()
    file_path = tmp_path / "memory" / "environment.json"

    save_environment(environment, file_path)

    assert load_environment(file_path) == environment
