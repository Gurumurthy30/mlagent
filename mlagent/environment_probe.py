"""Looks at the machine at start: OS, memory, GPU, installed libraries and internet access.

The result is saved as memory/environment.json and written into every prompt as short text.
The agent never imports torch itself. The torch check runs in a separate Python process.
"""

from __future__ import annotations

import json
import platform
import socket
import subprocess
import sys
from importlib import metadata
from pathlib import Path

import psutil
from pydantic import BaseModel

PROBED_PACKAGE_NAMES = [
    "numpy",
    "pandas",
    "scipy",
    "scikit-learn",
    "lightgbm",
    "xgboost",
    "catboost",
    "optuna",
    "statsmodels",
    "matplotlib",
    "pillow",
    "pyarrow",
    "torch",
    "torchvision",
    "torchaudio",
    "transformers",
    "sentence-transformers",
    "timm",
    "librosa",
    "soundfile",
    "opencv-python-headless",
]

TORCH_PROBE_CODE = """
import json
try:
    import torch
    print(json.dumps({
        "version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda or "",
        "mps_available": torch.backends.mps.is_available(),
    }))
except ImportError:
    print(json.dumps({}))
"""

BYTES_PER_GB = 1024**3
MEGABYTES_PER_GB = 1024


class EnvironmentInfo(BaseModel):
    operating_system: str = ""
    shell_name: str = ""
    python_version: str = ""
    cpu_count: int = 0
    ram_total_gb: float = 0.0
    ram_free_gb: float = 0.0
    gpu_name: str = ""  # empty means no NVIDIA GPU was found
    gpu_count: int = 0
    gpu_memory_total_gb: float = 0.0
    gpu_memory_free_gb: float = 0.0
    torch_version: str = ""  # empty means torch is not installed
    torch_cuda_available: bool = False
    torch_cuda_version: str = ""
    torch_mps_available: bool = False
    can_reach_pypi: bool = False
    can_reach_huggingface: bool = False
    installed_packages: dict[str, str] = {}
    missing_packages: list[str] = []

    @property
    def can_train_on_gpu(self) -> bool:
        return self.torch_cuda_available or self.torch_mps_available


def get_shell_name() -> str:
    return "powershell" if platform.system() == "Windows" else "bash"


def can_connect(host: str, port: int = 443, timeout_seconds: float = 3.0) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout_seconds):
            return True
    except OSError:
        return False


def read_nvidia_gpus() -> list[tuple[str, float, float]]:
    command = [
        "nvidia-smi",
        "--query-gpu=name,memory.total,memory.free",
        "--format=csv,noheader,nounits",
    ]
    try:
        completed = subprocess.run(command, capture_output=True, text=True, timeout=15)
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return []
    if completed.returncode != 0:
        return []

    gpus = []
    for line in completed.stdout.strip().splitlines():
        try:
            name, total_mb, free_mb = [part.strip() for part in line.rsplit(",", 2)]
            gpus.append(
                (
                    name,
                    round(int(total_mb) / MEGABYTES_PER_GB, 1),
                    round(int(free_mb) / MEGABYTES_PER_GB, 1),
                )
            )
        except ValueError:
            continue
    return gpus


def read_torch_status() -> dict:
    command = [sys.executable, "-c", TORCH_PROBE_CODE]
    try:
        completed = subprocess.run(command, capture_output=True, text=True, timeout=120)
    except subprocess.TimeoutExpired:
        return {}
    lines = completed.stdout.strip().splitlines()
    if not lines:
        return {}
    try:
        return json.loads(lines[-1])
    except json.JSONDecodeError:
        return {}


def read_package_versions() -> tuple[dict[str, str], list[str]]:
    installed_packages = {}
    missing_packages = []
    for package_name in PROBED_PACKAGE_NAMES:
        try:
            installed_packages[package_name] = metadata.version(package_name)
        except metadata.PackageNotFoundError:
            missing_packages.append(package_name)
    return installed_packages, missing_packages


def probe_environment() -> EnvironmentInfo:
    memory = psutil.virtual_memory()
    gpus = read_nvidia_gpus()
    torch_status = read_torch_status()
    installed_packages, missing_packages = read_package_versions()

    first_gpu_name, first_gpu_total_gb, first_gpu_free_gb = gpus[0] if gpus else ("", 0.0, 0.0)
    return EnvironmentInfo(
        operating_system=f"{platform.system()} {platform.release()}",
        shell_name=get_shell_name(),
        python_version=platform.python_version(),
        cpu_count=psutil.cpu_count(logical=True) or 0,
        ram_total_gb=round(memory.total / BYTES_PER_GB, 1),
        ram_free_gb=round(memory.available / BYTES_PER_GB, 1),
        gpu_name=first_gpu_name,
        gpu_count=len(gpus),
        gpu_memory_total_gb=first_gpu_total_gb,
        gpu_memory_free_gb=first_gpu_free_gb,
        torch_version=torch_status.get("version", ""),
        torch_cuda_available=torch_status.get("cuda_available", False),
        torch_cuda_version=torch_status.get("cuda_version", ""),
        torch_mps_available=torch_status.get("mps_available", False),
        can_reach_pypi=can_connect("pypi.org"),
        can_reach_huggingface=can_connect("huggingface.co"),
        installed_packages=installed_packages,
        missing_packages=missing_packages,
    )


def environment_to_text(environment: EnvironmentInfo) -> str:
    """The short description of this machine that goes into every prompt."""
    lines = [
        f"OS: {environment.operating_system} (shell: {environment.shell_name})",
        f"Python: {environment.python_version}",
        f"CPU cores: {environment.cpu_count}, RAM: {environment.ram_total_gb} GB total, "
        f"{environment.ram_free_gb} GB free",
    ]

    # GPU description inline
    if environment.gpu_name:
        lines.append(
            f"GPU: {environment.gpu_name} x{environment.gpu_count}, "
            f"{environment.gpu_memory_total_gb} GB total, {environment.gpu_memory_free_gb} GB free "
            "(the Ollama model also uses this GPU)"
        )
    elif environment.torch_mps_available:
        lines.append("GPU: Apple Silicon (MPS)")
    else:
        lines.append("GPU: none found")

    if not environment.torch_version:
        lines.append("PyTorch: not installed")
    elif environment.can_train_on_gpu:
        gpu_kind = (
            f"CUDA {environment.torch_cuda_version}" if environment.torch_cuda_available else "MPS"
        )
        lines.append(f"PyTorch: {environment.torch_version}, can train on GPU ({gpu_kind})")
    else:
        lines.append(f"PyTorch: {environment.torch_version}, CPU only")
        if environment.gpu_name:
            lines.append("NOTE: a GPU exists but this PyTorch build cannot use it. Plan for CPU.")

    lines.append(
        f"Internet: PyPI {'yes' if environment.can_reach_pypi else 'no'}, "
        f"Hugging Face {'yes' if environment.can_reach_huggingface else 'no'}"
    )
    installed_texts = [
        f"{name} {version}" for name, version in environment.installed_packages.items()
    ]
    lines.append("Installed libraries: " + (", ".join(installed_texts) or "none"))
    lines.append("Not installed: " + (", ".join(environment.missing_packages) or "none"))
    return "\n".join(lines)


def save_environment(environment: EnvironmentInfo, file_path: Path) -> None:
    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_path.write_text(environment.model_dump_json(indent=2), encoding="utf-8")
