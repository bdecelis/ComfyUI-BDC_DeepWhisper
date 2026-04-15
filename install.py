"""
ComfyUI-BDC_DeepWhisper · install.py
=====================================
Executed automatically by ComfyUI Manager when this node is installed
or updated. Uses sys.executable — always ComfyUI's own Python.

Flow
----
1. Fix known broken packages (pytorch-lightning < 2.0.0)
2. Capture torch state before any pip installs
3. pip install deep-whisper  ← may break torch (silero-vad declares a torch dep)
4. Repair torch immediately if broken (we know the right version from step 2)
5. Delegate the full GPU stack to deep_whisper.pipeline.setup_gpu
   (handles faster-whisper, whisperx, cuDNN, torch protection, verification)

Why two-phase torch protection?
--------------------------------
pip install deep-whisper can break torch independently of whisperx because
silero-vad (a required deep-whisper dependency) declares its own torch
requirement. This can cause pip to resolve and install a CPU-only torch even
before whisperx enters the picture. Capturing the version in step 2 means we
always know the correct CUDA tag to restore, even if torch is already broken
by the time setup_gpu runs.
"""

import os
import subprocess
import sys
from pathlib import Path
from typing import Optional


TAG = "[ComfyUI-BDC_DeepWhisper]"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def clean_env() -> dict:
    """os.environ with PYTHONNOUSERSITE=1 to isolate pip from user site-packages."""
    env = os.environ.copy()
    env["PYTHONNOUSERSITE"] = "1"
    return env


def pip_run(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-m", "pip", *args, "--no-user"],
        env=clean_env(),
        check=True,
    )


def run_py(code: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True, text=True, env=clean_env(),
    )


def get_torch_state() -> Optional[dict]:
    """Return torch version, CUDA version, and cuda_available. None if not installed."""
    r = run_py(
        "import torch; "
        "print(torch.__version__); "
        "print(torch.version.cuda or 'None'); "
        "print(torch.cuda.is_available())"
    )
    if r.returncode != 0:
        return None
    lines = r.stdout.strip().splitlines()
    if len(lines) < 3:
        return None
    return {
        "version":        lines[0].strip(),
        "cuda_version":   lines[1].strip(),
        "cuda_available": lines[2].strip() == "True",
    }


def torch_index_url(cuda_version: str) -> Optional[str]:
    if not cuda_version or cuda_version == "None":
        return None
    try:
        major = int(cuda_version.split(".")[0])
        minor = int(cuda_version.split(".")[1])
    except (ValueError, IndexError):
        return None
    if   major >= 12 and minor >= 8: tag = "cu128"
    elif major >= 12 and minor >= 6: tag = "cu126"
    elif major >= 12 and minor >= 4: tag = "cu124"
    elif major >= 12 and minor >= 1: tag = "cu121"
    elif major >= 11 and minor >= 8: tag = "cu118"
    else: return None
    return f"https://download.pytorch.org/whl/{tag}"


def restore_torch(state: dict) -> bool:
    """
    Force-reinstall the exact original torch + torchaudio + torchvision CUDA build.
    Returns True on success.
    """
    index_url = torch_index_url(state["cuda_version"])
    if not index_url:
        print(f"{TAG}   Cannot determine index URL for CUDA {state['cuda_version']}")
        return False
    base = state["version"].split("+")[0]
    print(f"{TAG}   Restoring torch=={base} ({index_url}) ...")
    result = subprocess.run(
        [sys.executable, "-m", "pip", "install",
         "--no-user", "--force-reinstall",
         "--index-url", index_url,
         f"torch=={base}", f"torchaudio=={base}", f"torchvision=={base}"],
        env=clean_env(),
    )
    return result.returncode == 0


def torch_is_broken(before: dict) -> bool:
    """Return True if torch CUDA has been overwritten since we captured 'before'."""
    after = get_torch_state()
    return (
        after is None
        or not after["cuda_available"]
        or after["cuda_version"] != before["cuda_version"]
    )


# ---------------------------------------------------------------------------
# Pre-flight: pytorch-lightning
# ---------------------------------------------------------------------------

def fix_pytorch_lightning() -> None:
    """
    Upgrade/remove pytorch-lightning < 2.0.0.
    These versions have a malformed version specifier (torch>=1.9.*) that
    pip 24.1+ rejects during dependency resolution even with --no-user, when
    the package is present in a sys.path entry added by ComfyUI at startup.
    """
    r = run_py("-m pip show pytorch-lightning".split()[0] +
               " show pytorch-lightning")
    # Use subprocess directly to avoid the arg-splitting issue
    r = subprocess.run(
        [sys.executable, "-m", "pip", "show", "pytorch-lightning"],
        capture_output=True, text=True, env=clean_env(),
    )
    if r.returncode != 0:
        print(f"{TAG}   pytorch-lightning not present — OK")
        return

    version = ""
    for line in r.stdout.splitlines():
        if line.startswith("Version:"):
            version = line.split(":", 1)[1].strip()
    if not version:
        return
    try:
        major = int(version.split(".")[0])
    except (ValueError, IndexError):
        major = 0
    if major >= 2:
        print(f"{TAG}   pytorch-lightning {version} — OK")
        return

    print(f"{TAG}   pytorch-lightning {version} has invalid metadata — upgrading ...")
    up = subprocess.run(
        [sys.executable, "-m", "pip", "install", "--no-user", "pytorch-lightning>=2.0.0"],
        env=clean_env(), capture_output=True, text=True,
    )
    if up.returncode == 0:
        print(f"{TAG}   pytorch-lightning upgraded.")
    else:
        subprocess.run(
            [sys.executable, "-m", "pip", "uninstall", "pytorch-lightning", "-y"],
            env=clean_env(), check=False,
        )
        print(f"{TAG}   pytorch-lightning removed.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def install() -> None:

    # ── 0. Pre-flight ────────────────────────────────────────────────────
    print(f"\n{TAG} Checking for known dependency conflicts...")
    fix_pytorch_lightning()

    # ── 1. Capture torch state before ANY pip installs ───────────────────
    # pip install deep-whisper can break torch independently of whisperx
    # because silero-vad (a deep-whisper dependency) declares its own torch
    # requirement. Capturing NOW means we always have the right CUDA version
    # to restore, even if torch is already broken when setup_gpu runs.
    print(f"\n{TAG} Capturing torch state before install...")
    torch_before = get_torch_state()
    if torch_before:
        print(
            f"{TAG}   torch {torch_before['version']}  "
            f"CUDA {torch_before['cuda_version']}  "
            f"available={torch_before['cuda_available']}"
        )
        if not torch_before["cuda_available"]:
            print(f"{TAG}   WARNING: torch CUDA not available before install.")
            print(f"{TAG}   This may mean torch is already broken or no GPU is present.")
    else:
        print(f"{TAG}   torch not installed — setup_gpu will handle installation.")

    # ── 2. Install deep-whisper ───────────────────────────────────────────
    print(f"\n{TAG} Installing deep-whisper...")
    try:
        pip_run("install", "deep-whisper>=0.1.0")
        print(f"{TAG}   deep-whisper installed.")
    except subprocess.CalledProcessError as e:
        print(f"{TAG}   ERROR: deep-whisper installation failed: {e}")
        print(f"{TAG}   Try installing manually:")
        print(f"{TAG}     python -m pip install --no-user deep-whisper>=0.1.0")
        return

    # ── 3. Repair torch if pip install broke it ───────────────────────────
    # Do this BEFORE running setup_gpu so it has a working torch to detect.
    if torch_before and torch_before["cuda_available"]:
        print(f"\n{TAG} Verifying torch CUDA after deep-whisper install...")
        if torch_is_broken(torch_before):
            print(f"{TAG}   torch CUDA overwritten — repairing before GPU stack install...")
            ok = restore_torch(torch_before)
            after = get_torch_state()
            if ok and after and after["cuda_available"]:
                print(f"{TAG}   torch restored: {after['version']} / CUDA {after['cuda_version']}")
            else:
                print(
                    f"\n{TAG}   WARNING: Could not restore torch automatically.\n"
                    f"{TAG}   setup_gpu will attempt to detect and repair on its own,\n"
                    f"{TAG}   but you may need to reinstall torch manually:\n"
                    f"{TAG}     python -m pip install --force-reinstall --no-user \\\n"
                    f"{TAG}       --index-url https://download.pytorch.org/whl/"
                    f"{torch_index_url(torch_before['cuda_version']) or 'cu128'} \\\n"
                    f"{TAG}       torch torchaudio torchvision"
                )
        else:
            after = get_torch_state()
            print(f"{TAG}   torch CUDA intact: {after['version']} / CUDA {after['cuda_version']}")

    # ── 4. Run setup_gpu for the full GPU stack ───────────────────────────
    # Handles: faster-whisper, whisperx (with torch protection), cuDNN,
    # and final verification. All GPU logic lives in setup_gpu — no duplication.
    print(f"\n{TAG} Running GPU stack installer (setup_gpu)...")
    result = subprocess.run(
        [sys.executable, "-m", "deep_whisper.pipeline.setup_gpu"],
        env=clean_env(),
    )
    if result.returncode != 0:
        print(
            f"\n{TAG} GPU stack setup completed with errors.\n"
            f"{TAG} Check the output above, or run manually:\n"
            f"{TAG}   python -m deep_whisper.pipeline.setup_gpu"
        )
    else:
        print(f"\n{TAG} Installation complete.")


install()
