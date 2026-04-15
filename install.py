"""
ComfyUI-BDC_DeepWhisper · install.py
=====================================
Executed automatically by ComfyUI Manager when this custom node is installed
or updated. Installs the deep-whisper package into ComfyUI's Python
environment using sys.executable — which is always ComfyUI's own Python,
regardless of what other Python installations exist on the system.

This script is intentionally minimal. All pipeline dependency management
(PyTorch CUDA version, cuDNN, etc.) is handled by deep-whisper's own
installer. For a fresh standalone install outside of ComfyUI Manager, see
the deep-whisper README.
"""

import subprocess
import sys
from pathlib import Path


def install() -> None:
    pip = [sys.executable, "-m", "pip", "install"]

    # Install from requirements.txt so the pinned version is respected.
    req_file = Path(__file__).parent / "requirements.txt"
    if req_file.exists():
        print("[ComfyUI-BDC_DeepWhisper] Installing requirements...")
        subprocess.run([*pip, "-r", str(req_file)], check=True)
    else:
        # Fallback: install the package directly if requirements.txt is missing
        print("[ComfyUI-BDC_DeepWhisper] Installing deep-whisper...")
        subprocess.run([*pip, "deep-whisper>=0.1.0"], check=True)

    print("[ComfyUI-BDC_DeepWhisper] Installation complete.")


install()
