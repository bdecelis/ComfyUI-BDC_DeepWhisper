"""
ComfyUI-BDC_DeepWhisper · install.py
=====================================
Executed automatically by ComfyUI Manager when this custom node is installed
or updated. Installs the deep-whisper package into ComfyUI's Python
environment using sys.executable.

Known issues handled here:

1. pytorch-lightning invalid requirement (pip 24.1+)
   pytorch-lightning < 2.0.0 has a malformed version specifier (torch>=1.9.*)
   that pip 24.1+ rejects. Upgraded or removed before any pip resolution runs.

2. User site-packages polluting pip's resolver
   ComfyUI and some custom nodes add user site-packages to sys.path at
   runtime. PYTHONNOUSERSITE=1 is set in all subprocess environments to
   prevent pip from seeing these packages during resolution.

3. Locked .pyd files (Windows)
   When ComfyUI is running, imported packages have their .pyd extension
   modules locked by Windows. pip cannot replace them. We detect this,
   skip the locked package if it already satisfies the requirement, and
   warn the user to restart ComfyUI if a restart is needed.

4. whisperx overwrites CUDA torch
   whisperx pulls in a CPU-only torch as a transitive dependency. We capture
   the torch state before, then detect and restore automatically after.
"""

import importlib.metadata
import os
import subprocess
import sys
from pathlib import Path
from typing import Optional


TAG = "[ComfyUI-BDC_DeepWhisper]"


# ---------------------------------------------------------------------------
# Packages deep-whisper and its deps need — in install order.
# Pinned to minimum acceptable versions. We check each one individually so
# that packages already present at a satisfactory version are never touched.
# ---------------------------------------------------------------------------
REQUIRED_PACKAGES = [
    # Core pipeline
    ("faster-whisper",   "faster_whisper",   "1.0.0"),
    ("whisperx",         "whisperx",          "3.0.0"),
    ("silero-vad",       "silero_vad",        "5.0.0"),
    ("librosa",          "librosa",           "0.10.0"),
    ("soundfile",        "soundfile",         "0.12.0"),
    ("num2words",        "num2words",         "0.5.13"),
    ("diff-match-patch", "diff_match_patch",  "20230430"),
    ("numpy",            "numpy",             "1.24.0"),
]


def clean_env() -> dict:
    """
    Return os.environ with PYTHONNOUSERSITE=1.
    Prevents Python from adding user site-packages to sys.path in subprocesses,
    which stops pip from seeing packages installed there during resolution.
    """
    env = os.environ.copy()
    env["PYTHONNOUSERSITE"] = "1"
    return env


def is_satisfied(import_name: str, min_version: str) -> bool:
    """
    Return True if the package is importable and meets the minimum version.
    Uses importlib.metadata — checks what's actually installed in this
    Python environment, not what's on sys.path.
    """
    try:
        installed = importlib.metadata.version(import_name)
        from packaging.version import Version
        return Version(installed) >= Version(min_version)
    except Exception:
        return False


def pip_install(pip_name: str, min_version: str) -> bool:
    """
    Install a single package if not already satisfied.
    Returns True on success, False on failure.
    Catches WinError 5 (Access Denied / locked file) specifically and
    reports it as a restart-required condition rather than a hard failure.
    """
    result = subprocess.run(
        [
            sys.executable, "-m", "pip", "install",
            "--no-user",
            f"{pip_name}>={min_version}",
        ],
        env=clean_env(),
        capture_output=True,
        text=True,
    )
    if result.returncode == 0:
        return True

    stderr = result.stderr + result.stdout
    if "WinError 5" in stderr or "Access is denied" in stderr:
        # Extract the locked filename for a more helpful message
        locked_file = ""
        for line in stderr.splitlines():
            if "WinError 5" in line or "Access is denied" in line:
                locked_file = line.strip()
                break
        print(
            f"{TAG}   LOCKED: {pip_name} — a .pyd file is in use by the running "
            f"ComfyUI process.\n"
            f"{TAG}   {locked_file}\n"
            f"{TAG}   This package is already loaded — restart ComfyUI to complete "
            f"the update."
        )
        return False

    # Other failure — print output and return False
    print(f"{TAG}   FAILED: {pip_name}")
    print(stderr[-800:] if len(stderr) > 800 else stderr)
    return False


def fix_pytorch_lightning() -> None:
    """
    Detect and fix pytorch-lightning < 2.0.0 before any pip resolution runs.
    Versions below 2.0.0 have a malformed version specifier that pip 24.1+
    rejects regardless of --no-user or PYTHONNOUSERSITE.
    """
    result = subprocess.run(
        [sys.executable, "-m", "pip", "show", "pytorch-lightning"],
        capture_output=True, text=True, env=clean_env(),
    )
    if result.returncode != 0:
        return

    version = ""
    for line in result.stdout.splitlines():
        if line.startswith("Version:"):
            version = line.split(":", 1)[1].strip()
            break
    if not version:
        return

    try:
        major = int(version.split(".")[0])
    except (ValueError, IndexError):
        major = 0

    if major >= 2:
        return

    print(
        f"{TAG} Found pytorch-lightning {version} with invalid pip metadata.\n"
        f"{TAG} Upgrading before dependency resolution..."
    )
    upgrade = subprocess.run(
        [sys.executable, "-m", "pip", "install", "--no-user",
         "pytorch-lightning>=2.0.0"],
        env=clean_env(),
        capture_output=True, text=True,
    )
    if upgrade.returncode == 0:
        print(f"{TAG} pytorch-lightning upgraded successfully.")
    else:
        print(f"{TAG} Upgrade failed — removing pytorch-lightning...")
        subprocess.run(
            [sys.executable, "-m", "pip", "uninstall", "pytorch-lightning", "-y"],
            env=clean_env(), check=False,
        )
        print(f"{TAG} pytorch-lightning removed.")


def get_torch_state() -> Optional[dict]:
    """Return torch version, CUDA version, and CUDA availability. None if not installed."""
    result = subprocess.run(
        [sys.executable, "-c",
         "import torch; "
         "print(torch.__version__); "
         "print(torch.version.cuda or 'None'); "
         "print(torch.cuda.is_available())"],
        capture_output=True, text=True, env=clean_env(),
    )
    if result.returncode != 0:
        return None
    lines = result.stdout.strip().splitlines()
    if len(lines) < 3:
        return None
    return {
        "version":        lines[0].strip(),
        "cuda_version":   lines[1].strip(),
        "cuda_available": lines[2].strip() == "True",
    }


def torch_index_url(cuda_version: str) -> Optional[str]:
    """Map torch.version.cuda to the matching PyTorch index URL."""
    if not cuda_version or cuda_version == "None":
        return None
    try:
        major, minor = int(cuda_version.split(".")[0]), int(cuda_version.split(".")[1])
    except (ValueError, IndexError):
        return None
    if   major >= 12 and minor >= 8: tag = "cu128"
    elif major >= 12 and minor >= 6: tag = "cu126"
    elif major >= 12 and minor >= 4: tag = "cu124"
    elif major >= 12 and minor >= 1: tag = "cu121"
    elif major >= 11 and minor >= 8: tag = "cu118"
    else: return None
    return f"https://download.pytorch.org/whl/{tag}"


def restore_torch(version: str, index_url: str) -> None:
    """Force-reinstall the exact original torch + torchaudio CUDA build."""
    base = version.split("+")[0]
    print(f"{TAG} Restoring torch=={base} from {index_url} ...")
    subprocess.run(
        [sys.executable, "-m", "pip", "install",
         "--no-user", "--force-reinstall",
         "--index-url", index_url,
         f"torch=={base}", f"torchaudio=={base}"],
        env=clean_env(), check=True,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def install() -> None:

    # ── 0. Fix known broken packages before any pip resolution ───────────
    print(f"{TAG} Checking for known dependency conflicts...")
    fix_pytorch_lightning()

    # ── 1. Capture torch state before whisperx can touch it ──────────────
    print(f"{TAG} Checking torch state...")
    before = get_torch_state()
    if before:
        print(
            f"{TAG}   torch {before['version']}  "
            f"CUDA {before['cuda_version']}  "
            f"available={before['cuda_available']}"
        )
    else:
        print(f"{TAG}   torch not found — skipping torch protection.")

    # ── 2. Install each package individually ─────────────────────────────
    # Check satisfaction first — only install what's missing or outdated.
    # This avoids pip trying to touch locked .pyd files that are already
    # loaded by the running ComfyUI process.
    print(f"\n{TAG} Installing packages...")
    needs_restart = []
    failed        = []

    for pip_name, import_name, min_ver in REQUIRED_PACKAGES:
        if is_satisfied(import_name, min_ver):
            print(f"{TAG}   OK (already installed):  {pip_name}>={min_ver}")
            continue
        print(f"{TAG}   Installing: {pip_name}>={min_ver} ...")
        ok = pip_install(pip_name, min_ver)
        if not ok:
            # Check whether the package is at least loadable at some version.
            # If it is, the failure was likely a lock — add to restart list.
            try:
                importlib.metadata.version(import_name)
                needs_restart.append(pip_name)
            except Exception:
                failed.append(pip_name)

    # ── 3. Detect and repair torch breakage caused by whisperx ───────────
    if before and before["cuda_available"]:
        print(f"\n{TAG} Verifying torch CUDA after install...")
        after = get_torch_state()
        cuda_broken = (
            after is None
            or not after["cuda_available"]
            or after["cuda_version"] != before["cuda_version"]
        )
        if cuda_broken:
            print(f"{TAG} WARNING: torch CUDA was modified.")
            index_url = torch_index_url(before["cuda_version"])
            if index_url:
                restore_torch(before["version"], index_url)
                final = get_torch_state()
                if final and final["cuda_available"]:
                    print(f"{TAG} torch CUDA restored. ({final['version']} / CUDA {final['cuda_version']})")
                else:
                    base = before["version"].split("+")[0]
                    failed.append(f"torch (manual fix: pip install --force-reinstall "
                                  f"--index-url {index_url} torch=={base} torchaudio=={base})")
        else:
            print(f"{TAG} torch CUDA intact. ({after['version']} / CUDA {after['cuda_version']})")

    # ── 4. Report outcome ─────────────────────────────────────────────────
    print(f"\n{TAG} {'='*50}")
    if not needs_restart and not failed:
        print(f"{TAG} Installation complete — all packages ready.")
    else:
        if needs_restart:
            print(f"\n{TAG} RESTART REQUIRED to complete updates for:")
            for pkg in needs_restart:
                print(f"{TAG}   - {pkg}")
            print(f"{TAG} These packages are already loaded by the running ComfyUI")
            print(f"{TAG} process. Restart ComfyUI — no further action needed.")
        if failed:
            print(f"\n{TAG} The following packages failed to install:")
            for pkg in failed:
                print(f"{TAG}   - {pkg}")
            print(f"{TAG} Run install.py manually or check the log above.")
    print(f"{TAG} {'='*50}\n")


install()
