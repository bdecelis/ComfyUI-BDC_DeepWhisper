# ComfyUI-BDC_DeepWhisper

> GPU-accelerated audio transcription with word-level timestamps, as ComfyUI nodes.

This package exposes the [deep-whisper](https://github.com/bdecelis/deep-whisper)
transcription pipeline as two composable nodes for ComfyUI graphs. Feed it an
audio file and get back a structured JSON transcript with precise word- and
sentence-level timestamps — running entirely on your local GPU.

---

## Nodes

### Transcribe Audio

Transcribes an audio file from scratch. Takes an audio path and optional context
prompt, runs the full pipeline, and returns a JSON transcript.

**Use this when:** you have audio and want to produce a transcript with timestamps.

### Align Timestamps

Adds timestamps to text you already have. Takes an audio path and your transcript,
runs acoustic alignment, and returns the same text with precise word- and
sentence-level timestamps attached.

**Use this when:** you already know what was said and only need timing information.
Your transcript wins on vocabulary, capitalisation, and phrasing — Whisper runs
on the audio purely to provide acoustic grounding for the aligner.

---

## Requirements

| Requirement | Minimum | Notes |
|---|---|---|
| ComfyUI | Any recent version | |
| Python | 3.10+ | ComfyUI's embedded Python |
| CUDA GPU | 8 GB VRAM | NVIDIA only |
| CUDA Driver | 11.8+ | |
| Disk space | ~6 GB | Model weights, downloaded on first run |

---

## Installation

### Via ComfyUI Manager (recommended)

1. Open ComfyUI Manager → Install Custom Nodes
2. Search for **deep-whisper**
3. Click Install — `install.py` runs automatically and installs
   [deep-whisper](https://github.com/bdecelis/deep-whisper) into ComfyUI's Python
4. Restart ComfyUI

### Manually

```powershell
cd ComfyUI\custom_nodes
git clone https://github.com/bdecelis/ComfyUI-BDC_DeepWhisper.git
cd ComfyUI-BDC_DeepWhisper
python install.py
```

Then restart ComfyUI.

> **Note:** `install.py` uses `sys.executable` — it always installs into
> ComfyUI's own Python, regardless of what other Python versions are on your
> system. You do not need to activate any environment first.

---

## Node parameters

### Transcribe Audio

| Parameter | Options | Default | Notes |
|---|---|---|---|
| `audio_path` | string | — | Absolute path to WAV, FLAC, or MP3 file |
| `prompt` | multiline string | — | Context prompt; use domain terms or a rough transcript |
| `whisper_model` | `large-v3-turbo` / `large-v2` / `large-v3` | `large-v3-turbo` | Turbo is fastest; large-v3 is highest accuracy |
| `compute_type` | `int8_float16` / `float16` / `int8` | `int8_float16` | Best speed/accuracy on 8 GB VRAM |
| `quality` | `balanced` / `fast` / `accurate` | `balanced` | Controls beam search width |
| `alignment_model` | `wav2vec2-base-960h` / `wav2vec2-large-960h-lv60` | `wav2vec2-base-960h` | Large gives better timestamps on fast speech |
| `language` | ISO 639-1 string | `en` | e.g. `fr`, `de`, `ja` |
| `timestamp_level` | `both` / `segment` / `word` / `none` | `both` | Controls output granularity |

### Align Timestamps

| Parameter | Options | Default | Notes |
|---|---|---|---|
| `audio_path` | string | — | Absolute path to WAV, FLAC, or MP3 file |
| `transcript_text` | multiline string | — | Your transcript of the audio |
| `whisper_model` | `large-v3-turbo` / `large-v2` / `large-v3` | `large-v3-turbo` | Used for acoustic grounding only |
| `compute_type` | `int8_float16` / `float16` / `int8` | `int8_float16` | |
| `alignment_model` | `wav2vec2-base-960h` / `wav2vec2-large-960h-lv60` | `wav2vec2-base-960h` | |
| `language` | ISO 639-1 string | `en` | |
| `timestamp_level` | `both` / `segment` / `word` / `none` | `both` | |

---

## Output format

Both nodes return a `STRING` containing a JSON object. Connect the output
to a **Convert JSON to Python Object** node (built into ComfyUI) for
downstream use, or wire it directly to any node that accepts a string.

```json
{
  "schema_version": "1.0",
  "metadata": {
    "duration_seconds": 125.4,
    "language": "en",
    "whisper_model": "large-v3-turbo",
    "alignment_model": "wav2vec2-base-960h",
    "timestamp_level": "both"
  },
  "transcript": "Hello, this is a test of the pipeline.",
  "segments": [
    {
      "id": 0,
      "start": 0.0,
      "end": 4.85,
      "text": "Hello, this is a test of the pipeline.",
      "confidence": 0.96,
      "flagged": false,
      "words": [
        { "word": "Hello",    "start": 0.00, "end": 0.42, "confidence": 0.99, "low_confidence": false },
        { "word": "pipeline", "start": 1.46, "end": 2.10, "confidence": 0.51, "low_confidence": true }
      ]
    }
  ],
  "words": [ ... ]
}
```

### `timestamp_level` field matrix

| Level | Sentence timing | Word timing | Top-level `words` |
|---|---|---|---|
| `both` | ✅ | ✅ | ✅ |
| `segment` | ✅ | ❌ | ❌ |
| `word` | ✅ | ✅ | ✅ |
| `none` | ❌ | ❌ | ❌ |

`transcript` (plain text) is always present regardless of level.

---

## VRAM usage

Both nodes load models once on first execution and keep them resident for
the ComfyUI session. Subsequent runs are fast — no reload cost.

| Component | VRAM |
|---|---|
| Whisper large-v3-turbo (int8_float16) | ~3.5 GB |
| wav2vec2-base alignment | ~0.5 GB |
| PyTorch runtime | ~0.5 GB |
| **Total** | **~4.5 GB** |

Switching to `wav2vec2-large` or `large-v3` increases usage but stays
within 8 GB.

---

## Troubleshooting

**Nodes do not appear after installation**
Run `install.py` manually from ComfyUI's Python, then restart:
```powershell
cd ComfyUI\custom_nodes\ComfyUI-BDC_DeepWhisper
python install.py
```

**`deep-whisper is not installed` in the ComfyUI console**
Same as above — `install.py` did not run successfully. Check the console
for pip errors.

**`Could not load cudnn_ops_infer64_8.dll`**
cuDNN 8 DLLs are missing from the CTranslate2 package directory. See the
[deep-whisper installation guide](https://github.com/bdecelis/deep-whisper#step-2--install-cudnn-8-windows-only)
for the fix.

**First run is slow**
Model weights (~4 GB) are downloaded on first use and cached locally.
Subsequent runs are fast.

**All word timestamps are `0.000`**
Alignment failed silently on a segment. This usually means the normalised
text and audio diverged significantly. Check that the correct language is
set and that the audio is clean spoken speech.

---

## Links

- [deep-whisper](https://github.com/bdecelis/deep-whisper) — the backend package, standalone pipeline, and full documentation
