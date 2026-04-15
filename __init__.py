"""
ComfyUI-BDC_DeepWhisper · __init__.py
======================================
ComfyUI entry point. This file is loaded by ComfyUI at startup when it
scans the custom_nodes directory.

Responsibilities:
    1. Confirm deep-whisper is installed and provide a clear message if not.
    2. Re-export NODE_CLASS_MAPPINGS and NODE_DISPLAY_NAME_MAPPINGS so
       ComfyUI can discover and register the nodes.

deep-whisper (the backend package) must be installed separately — either
automatically via ComfyUI Manager (which runs install.py), or manually:

    pip install deep-whisper>=0.1.0

See the README for full installation instructions.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Confirm deep-whisper is available before attempting to import the nodes.
# A clear error here is far more helpful than a cryptic ImportError later.
# ---------------------------------------------------------------------------
try:
    import pipeline  # noqa: F401 — confirms deep-whisper is installed
except ImportError:
    print(
        "\n[ComfyUI-BDC_DeepWhisper] ERROR: deep-whisper is not installed.\n"
        "  Run install.py or install manually:\n"
        "    pip install deep-whisper>=0.1.0\n"
        "  Then restart ComfyUI.\n"
    )
    # Export empty mappings so ComfyUI loads without crashing.
    # The nodes will simply not appear in the node list.
    NODE_CLASS_MAPPINGS        = {}
    NODE_DISPLAY_NAME_MAPPINGS = {}
    raise   # re-raise so ComfyUI logs the full traceback

# ---------------------------------------------------------------------------
# Import and re-export node mappings.
# ---------------------------------------------------------------------------
from comfyui_nodes import (  # noqa: E402
    NODE_CLASS_MAPPINGS,
    NODE_DISPLAY_NAME_MAPPINGS,
)

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]