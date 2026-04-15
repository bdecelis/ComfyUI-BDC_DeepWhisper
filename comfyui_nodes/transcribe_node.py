"""
ComfyUI-BDC_DeepWhisper · comfyui_nodes/transcribe_node.py
===========================================================
TranscribeNode: full pipeline — audio file in, JSON transcript out.

Orchestrates the complete deep-whisper pipeline:
    load → VAD → transcribe → normalise → align → build output

For the user-transcript path (adding timestamps to existing text),
use AlignTimestampsNode instead.
"""

from __future__ import annotations


class TranscribeNode:
    """
    Transcribe an audio file and return a JSON string with word-level
    and sentence-level timestamps.
    """

    CATEGORY    = "deep-whisper"
    FUNCTION    = "execute"
    RETURN_TYPES  = ("STRING",)
    RETURN_NAMES  = ("transcription_json",)

    @classmethod
    def INPUT_TYPES(cls) -> dict:
        return {
            "required": {
                "audio_path": ("STRING", {
                    "default":   "",
                    "multiline": False,
                    "tooltip":   "Absolute path to the audio file (WAV, FLAC, MP3, etc.)",
                }),
                "prompt": ("STRING", {
                    "default":   "",
                    "multiline": True,
                    "tooltip":   "Context prompt to anchor Whisper's vocabulary. "
                                 "Use domain-specific terms or a rough transcript.",
                }),
                "whisper_model": (
                    ["large-v3-turbo", "large-v2", "large-v3"],
                    {
                        "default": "large-v3-turbo",
                        "tooltip": "large-v3-turbo: fastest, best balance. "
                                   "large-v3: highest accuracy, more VRAM.",
                    },
                ),
                "compute_type": (
                    ["int8_float16", "float16", "int8"],
                    {
                        "default": "int8_float16",
                        "tooltip": "int8_float16: best speed/accuracy on 8 GB VRAM (recommended).",
                    },
                ),
                "quality": (
                    ["balanced", "fast", "accurate"],
                    {
                        "default": "balanced",
                        "tooltip": "fast: greedy decode, ~same accuracy on clean speech. "
                                   "accurate: beam search, slower.",
                    },
                ),
                "alignment_model": (
                    ["wav2vec2-base-960h", "wav2vec2-large-960h-lv60"],
                    {
                        "default": "wav2vec2-base-960h",
                        "tooltip": "base: fast, sufficient for clean speech. "
                                   "large: more accurate on fast or complex speech.",
                    },
                ),
                "language": ("STRING", {
                    "default": "en",
                    "tooltip": "ISO 639-1 language code (e.g. en, fr, de, ja).",
                }),
                "timestamp_level": (
                    ["both", "segment", "word", "none"],
                    {
                        "default": "both",
                        "tooltip": "both: word + sentence timestamps. "
                                   "segment: sentence only. "
                                   "none: plain transcript only.",
                    },
                ),
            }
        }

    # ------------------------------------------------------------------

    def execute(
        self,
        audio_path:      str,
        prompt:          str,
        whisper_model:   str,
        compute_type:    str,
        quality:         str,
        alignment_model: str,
        language:        str,
        timestamp_level: str,
    ) -> tuple[str]:

        from deep_whisper.pipeline.audio       import load_audio, normalize_audio
        from deep_whisper.pipeline.vad         import get_speech_chunks
        from deep_whisper.pipeline.transcribe  import transcribe_chunks
        from deep_whisper.pipeline.normalise   import normalise_segments
        from deep_whisper.pipeline.align       import align_segments
        from deep_whisper.pipeline.postprocess import build_output, serialise

        audio    = normalize_audio(load_audio(audio_path))
        chunks   = get_speech_chunks(audio)
        segments = transcribe_chunks(
            chunks,
            initial_prompt = prompt,
            model_name     = whisper_model,
            compute_type   = compute_type,
            quality        = quality,
            language       = language,
        )
        segments = normalise_segments(segments, language=language)
        segments = align_segments(
            segments,
            audio,
            model_label = alignment_model,
            language    = language,
        )
        output = build_output(
            segments,
            audio,
            language                 = language,
            whisper_model            = whisper_model,
            alignment_model          = alignment_model,
            prompt                   = prompt,
            user_transcript_provided = False,
            timestamp_level          = timestamp_level,
        )
        return (serialise(output),)
