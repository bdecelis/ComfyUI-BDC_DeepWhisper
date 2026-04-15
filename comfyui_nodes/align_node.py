"""
ComfyUI-BDC_DeepWhisper · comfyui_nodes/align_node.py
======================================================
AlignTimestampsNode: add timestamps to an existing transcript.

Use this node when you already have the correct text and only need
precise word- and sentence-level timestamps added to it.

Orchestrates the user-transcript pipeline path:
    load → VAD → transcribe (acoustic grounding) → reconcile
    → normalise → align → build output

Your transcript wins on vocabulary, capitalisation, and phrasing.
Whisper runs on the audio to provide acoustic grounding for alignment.
"""

from __future__ import annotations


class AlignTimestampsNode:
    """
    Add precise word-level and sentence-level timestamps to a
    user-provided transcript by aligning it to an audio file.
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
                "transcript_text": ("STRING", {
                    "default":   "",
                    "multiline": True,
                    "tooltip":   "Your transcript of the audio. This text wins on vocabulary, "
                                 "capitalisation, and phrasing. Whisper provides acoustic "
                                 "grounding for timestamp alignment.",
                }),
                "whisper_model": (
                    ["large-v3-turbo", "large-v2", "large-v3"],
                    {
                        "default": "large-v3-turbo",
                        "tooltip": "Model used for acoustic grounding. "
                                   "Does not affect the output transcript text.",
                    },
                ),
                "compute_type": (
                    ["int8_float16", "float16", "int8"],
                    {
                        "default": "int8_float16",
                        "tooltip": "int8_float16: best speed/accuracy on 8 GB VRAM (recommended).",
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
        transcript_text: str,
        whisper_model:   str,
        compute_type:    str,
        alignment_model: str,
        language:        str,
        timestamp_level: str,
    ) -> tuple[str]:

        from deep_whisper.pipeline.audio       import load_audio, normalize_audio
        from deep_whisper.pipeline.vad         import get_speech_chunks
        from deep_whisper.pipeline.transcribe  import transcribe_chunks
        from deep_whisper.pipeline.reconcile   import reconcile_segments
        from deep_whisper.pipeline.normalise   import normalise_segments
        from deep_whisper.pipeline.align       import align_segments
        from deep_whisper.pipeline.postprocess import build_output, serialise

        audio    = normalize_audio(load_audio(audio_path))
        chunks   = get_speech_chunks(audio)

        # Whisper runs for acoustic grounding — seeded with the user's
        # transcript so its vocabulary is anchored from the first chunk.
        segments = transcribe_chunks(
            chunks,
            initial_prompt = transcript_text,
            model_name     = whisper_model,
            compute_type   = compute_type,
            quality        = "balanced",
            language       = language,
        )

        # Reconcile: user's vocabulary wins; Whisper provides acoustic grounding
        segments = reconcile_segments(transcript_text, segments)
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
            prompt                   = transcript_text,
            user_transcript_provided = True,
            timestamp_level          = timestamp_level,
        )
        return (serialise(output),)
