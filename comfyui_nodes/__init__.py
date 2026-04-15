"""
ComfyUI-BDC_DeepWhisper · comfyui_nodes/__init__.py
====================================================
Registers all deep-whisper nodes with ComfyUI.

NODE_CLASS_MAPPINGS     maps internal node IDs to their Python classes.
NODE_DISPLAY_NAME_MAPPINGS  maps internal IDs to the names shown in the UI.

Adding a new node:
    1. Create comfyui_nodes/my_node.py  with a class  MyNode
    2. Import it here and add entries to both dicts below.
"""

from .transcribe_node import TranscribeNode
from .align_node      import AlignTimestampsNode

NODE_CLASS_MAPPINGS: dict = {
    "DeepWhisperTranscribe":     TranscribeNode,
    "DeepWhisperAlignTimestamps": AlignTimestampsNode,
}

NODE_DISPLAY_NAME_MAPPINGS: dict = {
    "DeepWhisperTranscribe":      "Transcribe Audio (deep-whisper)",
    "DeepWhisperAlignTimestamps": "Align Timestamps (deep-whisper)",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
