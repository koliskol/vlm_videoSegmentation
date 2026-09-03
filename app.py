"""
Gradio app: upload any video and segment it into human action segments
using VideoLLaMA3-2B (local vision-language model), with no pose estimation
and no domain assumptions about the task.

Run with:
    conda activate <your env>   # see README.md for setup
    python app.py
"""

from __future__ import annotations

import json
import os
import subprocess
import tempfile

import gradio as gr
import pandas as pd

import most_index as mi
import viz_clips
import vlm_label
import vlm_segment


def _video_duration_s(video_path: str) -> float:
    """Read duration via ffprobe -- avoids an OpenCV/MediaPipe dependency
    this project otherwise has no use for."""
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "format=duration", "-of", "json", video_path],
        capture_output=True, text=True, check=True,
    )
    data = json.loads(result.stdout)
    return float(data["format"]["duration"])


def segment_fn(video_path, vlm_session, frames_per_window, step_frames, progress=gr.Progress()):
    if not video_path:
        raise gr.Error("Upload a video first.")

    duration_s = _video_duration_s(video_path)
    if duration_s <= 0:
        raise gr.Error("Could not read this video's duration.")

    if vlm_session is None or vlm_session.source_video_path != video_path:
        progress(0, desc="Loading VideoLLaMA3-2B (first time only, ~5GB VRAM)...")
        vlm_session = vlm_label.load_session(video_path, duration_s, existing=vlm_session)

    def cb(frac, i, n):
        progress(frac, desc=f"Classifying window {i}/{n}")

    segments = vlm_segment.segment_by_action(
        vlm_session, duration_s, int(frames_per_window), int(step_frames), progress_cb=cb,
    )

    rows = []
    total_index = 0
    for i, s in enumerate(segments):
        index = mi.duration_to_index(s.duration_s)
        tmu = mi.index_to_tmu(index)
        total_index += index
        rows.append({
            "Seq": i + 1, "Start (s)": round(s.start_t, 2), "End (s)": round(s.end_t, 2),
            "Duration (s)": round(s.duration_s, 2), "Action": s.label,
            "Index": index, "TMU": round(tmu, 1), "Time (s)": round(mi.tmu_to_seconds(tmu), 3),
        })
    df = pd.DataFrame(rows)

    total_tmu = mi.index_to_tmu(total_index)
    summary = (
        f"**{len(segments)} action segments** over {duration_s:.2f}s of video — "
        f"total index **{total_index}** → **{total_tmu:.0f} TMU** → "
        f"**{mi.tmu_to_seconds(total_tmu):.2f}s** of TMU-equivalent time.\n\n"
        f"⚠️ This is **not** a MOST analysis in the methodological sense — each "
        f"segment's index comes directly from its *observed* duration "
        f"(`duration ÷ 0.036s ÷ 10`, snapped to the standard index scale), not "
        f"from a distance/weight lookup table the way real MOST elements are "
        f"built. It's a video-based time study expressed in TMU units, not a "
        f"certified predetermined-time standard. See README.md."
    )

    out_dir = os.path.join(tempfile.gettempdir(), "vlm_action_clips")
    gallery_items = viz_clips.export_action_clips(video_path, segments, out_dir)

    return vlm_session, summary, df, gallery_items


METHODOLOGY_MD = """
## How this works

1. **Slide a window** of `frames_per_window` frames (sampled at 5/s) across
   the whole video, moving `step_frames` frames each step. Frames are drawn
   from a 160px-wide downscaled copy of the video, so more frames fit in
   VRAM budget (see `vlm_label.py`'s ceiling notes for why).
2. **Caption the first window** — that becomes the working label ("anchor")
   for the segment currently being built.
3. **For every next window**, show the model that window's own footage and
   ask, visually: *"Is this the SAME action as `<anchor>`?"* — a
   constrained YES/NO question, not a text-similarity comparison (a small
   model phrases the same action differently call to call, so comparing
   captions as text would over-segment).
4. **YES** extends the current segment. **NO** closes it — cutting the
   boundary exactly at the disagreeing window's start, since windows
   overlap and the segment may already extend past that point — and starts
   a new segment anchored on a fresh caption of that window.
5. **Recaption pass**: once boundaries are final, each segment gets one
   more caption call over its *full* span, using sharper settings (224px,
   up to 6 frames) than the tiny window used for step 3's boundary
   decisions. Short, low-res clips are enough for the model to judge
   "same or different" reliably, but not enough for it to describe *what*
   the action specifically is — it tends to fall back to a generic
   scene-level description. The richer recaption pass helps, but only
   partially: on fast, visually repetitive footage with short segments, a
   2B model will still land on generic phrasing fairly often. That's a real
   capability ceiling, not a prompt-wording problem — we tried several
   more elaborate prompts and they made it worse (the model either went
   silent or collapsed into repeating one fixed word for every window).

## What this is not

- **Not frame-accurate.** Boundary precision is bounded by the slide step,
  not by individual frames. A shorter step gives more precise boundaries
  but proportionally more (slower) VLM calls.
- **Not a MOST analysis.** The Index/TMU/Time columns exist for
  comparability with the [MOST work-measurement
  technique](https://en.wikipedia.org/wiki/Maynard_Operation_Sequence_Technique)'s
  units, but a real MOST element's index comes from a distance/weight
  lookup table — the entire point of a *predetermined* time system, as
  opposed to time study (stopwatching the actual footage). Since a
  free-form action segment has no distance parameter, its index here is
  derived directly from observed duration instead. Treat the timing columns
  as a video-based time study, not a certified labor standard.
- **Not ground truth.** A 2B model's captions and same/different judgments
  are its best guess, not verified fact — always spot-check a few clips in
  the gallery against the label shown.

## Hardware notes

Tuned and tested on a single 12GB-VRAM GPU (RTX 4080 Laptop) with no
flash-attention installed. If you have flash-attention available, or more
VRAM, `LABEL_WIDTH`/`LABEL_MAX_FRAMES`/`LOWRES_WIDTH`/`LOWRES_MAX_FRAMES` in
`vlm_label.py` are conservative — you can likely raise them for better
detail, but re-verify empirically rather than assuming; OOM here happens
suddenly and mid-call, not gracefully.
"""


with gr.Blocks(title="VLM Action Segmentation") as demo:
    gr.Markdown(
        "# VLM Action Segmentation\n"
        "Upload a video of a person doing something, and segment it into "
        "action clips using VideoLLaMA3-2B — a local vision-language model, "
        "no pose estimation or task-specific assumptions involved. Works on "
        "any human action video."
    )

    vlm_state = gr.State(None)

    with gr.Tab("Segment"):
        with gr.Row():
            with gr.Column(scale=1):
                video_in = gr.Video(label="Video", sources=["upload"])
                frames_per_window_slider = gr.Slider(2, 12, value=vlm_segment.FRAMES_PER_WINDOW, step=1, label="Frames per window")
                step_frames_slider = gr.Slider(1, 12, value=vlm_segment.STEP_FRAMES, step=1, label="Slide step (frames)")
                segment_btn = gr.Button("Segment by VLM action recognition", variant="primary")
                gr.Markdown(
                    "_Frames are sampled at 5/s (so 10 frames ≈ 2s of "
                    "footage). **12 is the tested-safe ceiling on a 12GB "
                    "GPU** — verified experimentally (16 frames OOMs); the "
                    "slider is capped there deliberately. Step ≤ window = "
                    "overlapping windows (recommended, catches short "
                    "actions better). First click loads the model "
                    "(~5GB VRAM, a few seconds); later clicks with a "
                    "different video reuse the loaded model and only "
                    "regenerate the small resized copies._"
                )
            with gr.Column(scale=2):
                summary_md = gr.Markdown()
                segments_df = gr.Dataframe(label="Discovered action segments (with TMU-equivalent timing)")
                clips_gallery = gr.Gallery(
                    label="Action segment clips (chronological)",
                    columns=3, height=420, object_fit="contain", preview=True,
                )

    with gr.Tab("Methodology & limitations"):
        gr.Markdown(METHODOLOGY_MD)

    segment_btn.click(
        segment_fn,
        inputs=[video_in, vlm_state, frames_per_window_slider, step_frames_slider],
        outputs=[vlm_state, summary_md, segments_df, clips_gallery],
    )


if __name__ == "__main__":
    demo.launch()
