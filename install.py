"""
ComfyUI-BDC_DeepWhisper · install.py
=====================================
Executed automatically by ComfyUI Manager when this custom node is installed
or updated. Installs the deep-whisper package into ComfyUI's Python
environment using sys.executable.

Known issues handled automatically:

1. pytorch-lightning invalid requirement (pip 24.1+)
   pytorch-lightning < 2.0.0 ships with a malformed version specifier
   (torch>=1.9.*) that pip 24.1+ rejects during dependency resolution.

   This persists even with --no-user because ComfyUI (and some custom nodes)
   explicitly add the user site-packages directory to sys.path via
   sys.path.append(). pip inherits this sys.path and sees the broken package
   regardless of --no-user, which only controls install targets.

   Fix: set PYTHONNOUSERSITE=1 in the subprocess environment. This prevents
   Python from adding the user site-packages to sys.path in the pip subprocess
   before pip starts, so the broken package is never visible to the resolver.

2. whisperx overwrites CUDA torch with a CPU-only build
   whisperx pulls in a CPU-only torch as a transitive dependency, silently
   replacing ComfyUI's CUDA build. We capture the torch state before, then
   detect and restore automatically after if broken.
"""

import os
import subprocess
import sys
from pathlib import Path


TAG = "[ComfyUI-BDC_DeepWhisper]"


def clean_env() -> dict:
    """
    Return a copy of os.environ with PYTHONNOUSERSITE=1 set.

    This prevents Python from adding the user site-packages directory
    (e.g. AppData/Roaming/Python/Python312/site-packages) to sys.path
    in any subprocess we spawn. Without this, pip inherits the parent
    process's sys.path — which ComfyUI has already populated with the
    user site — and sees any broken packages installed there, even when
    --no-user is passed.
    """
    env = os.environ.copy()
    env["PYTHONNOUSERSITE"] = "1"
    return env


def pip_run(*args: str) -> None:
    """
    Run a pip command using ComfyUI's Python.
    Always sets PYTHONNOUSERSITE=1 and --no-user for a clean resolution
    environment regardless of what the parent process has on sys.path.
    """
    subprocess.run(
        [sys.executable, "-m", "pip", *args, "--no-user"],
        env=clean_env(),
        check=True,
    )


def run_python(*args: str) -> subprocess.CompletedProcess:
    """Run a Python one-liner in a clean environment (no user site-packages)."""
    return subprocess.run(
        [sys.executable, *args],
        capture_output=True,
        text=True,
        env=clean_env(),
    )


def get_torch_state() -> dict | None:
    """
    Return the current torch version, CUDA version, and CUDA availability.
    Returns None if torch is not installed.
    """
    result = run_python(
        "-c",
        "import torch; "
        "print(torch.__version__); "
        "print(torch.version.cuda or 'None'); "
        "print(torch.cuda.is_available())"
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


def torch_index_url(cuda_version: str) -> str | None:
    """
    Map a torch.version.cuda string to the matching PyTorch index URL.
    Returns None if no matching tag is found.
    """
    if not cuda_version or cuda_version == "None":
        return None
    try:
        parts = cuda_version.split(".")
        major, minor = int(parts[0]), int(parts[1])
    except (ValueError, IndexError):
        return None

    if   major >= 12 and minor >= 8: tag = "cu128"
    elif major >= 12 and minor >= 6: tag = "cu126"
    elif major >= 12 and minor >= 4: tag = "cu124"
    elif major >= 12 and minor >= 1: tag = "cu121"
    elif major >= 11 and minor >= 8: tag = "cu118"
    else:
        return None

    return f"https://download.pytorch.org/whl/{tag}"


def restore_torch(version: str, index_url: str) -> None:
    """
    Force-reinstall an exact torch + torchaudio build from the given index.
    Strips the local version suffix (e.g. +cu128) so pip can resolve it
    from the remote index.
    """
    base_version = version.split("+")[0]
    print(f"{TAG} Restoring torch=={base_version} from {index_url} ...")
    subprocess.run(
        [
            sys.executable, "-m", "pip", "install",
            "--no-user",
            "--force-reinstall",
            "--index-url", index_url,
            f"torch=={base_version}",
            f"torchaudio=={base_version}",
        ],
        env=clean_env(),
        check=True,
    )


def fix_broken_packages() -> None:
    """
    Detect and fix packages with known invalid metadata that cause pip 24.1+
    to abort dependency resolution.

    pytorch-lightning < 2.0.0 ships with torch>=1.9.* which is invalid.
    We upgrade it rather than removing it — safer if other nodes depend on it.
    Falls back to removal if the upgrade itself fails.
    """
    result = run_python("-m", "pip", "show", "pytorch-lightning")
    if result.returncode != 0:
        return  # not installed

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
        return  # already a valid version

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
        print(
            f"{TAG} Upgrade failed — removing pytorch-lightning instead.\n"
            f"{TAG} (It will be reinstalled at a valid version if required.)"
        )
        subprocess.run(
            [sys.executable, "-m", "pip", "uninstall",
             "pytorch-lightning", "-y"],
            env=clean_env(),
            check=True,
        )
        print(f"{TAG} pytorch-lightning removed.")


def install() -> None:

    # ── 0. Fix known broken packages before resolving dependencies ───────
    print(f"{TAG} Checking for known dependency conflicts...")
    fix_broken_packages()

    # ── 1. Capture torch state before any installs ────────────────────────
    print(f"{TAG} Checking torch before install...")
    before = get_torch_state()
    if before:
        print(
            f"{TAG}   torch {before['version']}  "
            f"CUDA {before['cuda_version']}  "
            f"cuda_available={before['cuda_available']}"
        )
    else:
        print(f"{TAG}   torch not installed — skipping torch protection.")

    # ── 2. Install deep-whisper and its dependencies ──────────────────────
    req_file = Path(__file__).parent / "requirements.txt"
    if req_file.exists():
        print(f"{TAG} Installing from requirements.txt ...")
        pip_run("install", "-r", str(req_file))
    else:
        print(f"{TAG} Installing deep-whisper ...")
        pip_run("install", "deep-whisper>=0.1.0")

    # ── 3. Detect and repair torch breakage caused by whisperx ───────────
    if before and before["cuda_available"]:
        print(f"{TAG} Verifying torch CUDA after install...")
        after = get_torch_state()

        cuda_broken = (
            after is None
            or not after["cuda_available"]
            or after["cuda_version"] != before["cuda_version"]
        )

        if cuda_broken:
            print(f"{TAG} WARNING: torch CUDA was modified by a dependency.")
            if after:
                print(f"{TAG}   Before: {before['version']} / CUDA {before['cuda_version']}")
                print(f"{TAG}   After:  {after['version']} / CUDA {after['cuda_version']}")

            index_url = torch_index_url(before["cuda_version"])
            if index_url:
                restore_torch(before["version"], index_url)

                final = get_torch_state()
                if final and final["cuda_available"]:
                    print(
                        f"{TAG} torch CUDA restored successfully. "
                        f"({final['version']} / CUDA {final['cuda_version']})"
                    )
                else:
                    base = before["version"].split("+")[0]
                    print(
                        f"\n{TAG} ERROR: Could not restore torch CUDA automatically.\n"
                        f"{TAG} Please run this command manually:\n"
                        f"{TAG}\n"
                        f"{TAG}   python_embeded\\python.exe -m pip install "
                        f"--force-reinstall --no-user "
                        f"--index-url {index_url} "
                        f"torch=={base} torchaudio=={base}\n"
                    )
            else:
                print(
                    f"\n{TAG} Could not determine PyTorch index URL for "
                    f"CUDA {before['cuda_version']}.\n"
                    f"{TAG} See: https://github.com/bdecelis/deep-whisper#installation\n"
                )
        else:
            print(
                f"{TAG} torch CUDA intact. "
                f"({after['version']} / CUDA {after['cuda_version']})"
            )

    print(f"\n{TAG} Installation complete.")


install()
