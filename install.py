"""
ComfyUI-BDC_DeepWhisper · install.py
=====================================
Executed automatically by ComfyUI Manager when this custom node is installed
or updated. Installs the deep-whisper package into ComfyUI's Python
environment using sys.executable.

Two known issues are handled automatically here:

1. pytorch-lightning invalid requirement (pip 24.1+)
   Old versions of pytorch-lightning in the user site-packages have a
   malformed version specifier that pip 24.1+ rejects during dependency
   resolution. --no-user prevents pip from scanning user site-packages,
   avoiding the collision entirely.

2. whisperx overwrites CUDA torch with a CPU-only build
   whisperx pulls in a CPU-only torch as a transitive dependency, which
   silently replaces ComfyUI's CUDA build. We capture the exact torch
   version and CUDA tag before installing, then detect and repair the
   breakage automatically after.
"""

import subprocess
import sys
from pathlib import Path


TAG = "[ComfyUI-BDC_DeepWhisper]"


def pip_run(*args: str) -> None:
    """Run a pip command using ComfyUI's Python. Always uses --no-user."""
    subprocess.run(
        [sys.executable, "-m", "pip", *args, "--no-user"],
        check=True,
    )


def get_torch_state() -> dict | None:
    """
    Return the current torch version, CUDA version, and CUDA availability.
    Returns None if torch is not installed.
    """
    result = subprocess.run(
        [
            sys.executable, "-c",
            "import torch; "
            "print(torch.__version__); "
            "print(torch.version.cuda or 'None'); "
            "print(torch.cuda.is_available())"
        ],
        capture_output=True, text=True,
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
    Returns None if no matching tag is found (e.g. CPU-only or unknown).
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
    Strips the local version suffix (e.g. +cu128) before reinstalling so
    pip can resolve the package from the remote index.
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
        check=True,
    )


def install() -> None:

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

    # ── 3. Detect and repair torch breakage ──────────────────────────────
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

                # Final verification
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
                    f"{TAG} Please reinstall torch manually with the correct "
                    f"--index-url for your CUDA version.\n"
                    f"{TAG} See: https://github.com/bdecelis/deep-whisper#installation\n"
                )
        else:
            print(
                f"{TAG} torch CUDA intact. "
                f"({after['version']} / CUDA {after['cuda_version']})"
            )

    print(f"\n{TAG} Installation complete.")


install()
