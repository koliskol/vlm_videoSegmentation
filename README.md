# VLM Action Segmentation

Segment any video of a person doing something into labeled action clips —
using a local vision-language model (VideoLLaMA3-2B), with no pose
estimation, no object detection, and no assumptions about what kind of
task is being performed.

Upload a video, click segment, get back a table of action segments (with
start/end time, a short description, and a MOST-style TMU-equivalent
timing) plus a gallery of the actual cut clips so you can inspect each
segment for yourself.

## How it works (short version)

A window slides across the video. For each window, the model is asked
*"is this the same action as the previous segment?"* — visually, on that
window's own footage, not by comparing text descriptions. A "no" closes
the current segment and starts a new one. Once boundaries are settled,
each final segment gets one richer caption describing what's actually
happening in it.

Full detail, including exact prompts and the reasoning behind several
design choices that came from hitting real failure modes (OOM crashes,
degenerate model outputs, overlapping segment boundaries), is in the
**Methodology & limitations** tab inside the app.

## What this is *not*

- **Not a certified MOST (predetermined time system) analysis.** The
  Index/TMU/Time columns exist for unit comparability, but a real MOST
  element's index comes from a distance/weight lookup table, not from the
  clip's own observed duration (which is what this tool does, for lack of
  any distance parameter to look up). It's closer to a video-based time
  study expressed in TMU units. See the in-app Methodology tab for the
  full explanation.
- **Not frame-accurate.** Segment boundaries are only as precise as the
  slide step you choose.
- **Not ground truth.** A 2B-parameter model's captions and same/different
  judgments are its best guess on short, sometimes low-resolution clips —
  spot-check the gallery.

## Requirements

- **NVIDIA GPU with ~6GB+ free VRAM.** Tested and tuned on a 12GB laptop
  GPU (RTX 4080 Laptop) with no flash-attention installed — the resolution
  and frame-count constants in `vlm_label.py` are deliberately conservative
  for that hardware. If you have more VRAM or flash-attention available,
  you likely can raise them, but re-verify empirically: this model OOMs
  suddenly mid-call, not gracefully, and the failure mode scales with
  `(frames per window x patches per frame)^2`.
- **`ffmpeg`/`ffprobe`** available on `PATH` (system package, not a pip
  package — e.g. `apt install ffmpeg` on Debian/Ubuntu).
- **Python 3.10–3.12.** `transformers==4.49.0` is pinned deliberately (see
  Setup below); newer/older combinations are untested.
- **~4GB disk** for the model weights (downloaded automatically on first
  run, cached under `~/.cache/huggingface`) plus a little scratch space for
  temporary resized video copies and cut clips (under your OS temp dir).

## Setup

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

**Why `transformers==4.49.0` specifically:** VideoLLaMA3's model code is
loaded via `trust_remote_code=True` and imports `VideoInput` from
`transformers.image_utils` — an API that transformers removed in its 5.x
line. Installing transformers 4.49.0 pulls in `huggingface_hub<1.0`, which
technically conflicts with newer Gradio's stated `huggingface_hub>=1.16`
requirement — pip will print a dependency-conflict warning during install.
This is expected and harmless: Gradio's actual runtime code doesn't hit the
removed/changed APIs, and the app has been tested end-to-end against this
exact combination.

The first time you click "Segment," the app downloads
`DAMO-NLP-SG/VideoLLaMA3-2B` (~4GB) from Hugging Face — this needs network
access and will take a few minutes on a typical connection.

## Run

```bash
python app.py
```

Open the printed local URL (typically `http://127.0.0.1:7860`), upload a
video, and click **Segment by VLM action recognition**. Adjust "Frames per
window" and "Slide step" to trade off boundary precision against caption
quality/speed — smaller windows give more precise cuts but blander,
more-repetitive captions (a real capability limit of a 2B model on tiny
clips, not a setting you can prompt-engineer around; see the Methodology
tab for what we tried).

## Project structure

| File | Purpose |
|---|---|
| `app.py` | Gradio UI and wiring |
| `vlm_label.py` | VideoLLaMA3-2B session management, clip captioning/classification |
| `vlm_segment.py` | Sliding-window segmentation algorithm (the core logic) |
| `viz_clips.py` | Cuts one playable clip per discovered segment for the gallery |
| `most_index.py` | MOST index/TMU/seconds unit conversions (see the "not a MOST analysis" note above) |
