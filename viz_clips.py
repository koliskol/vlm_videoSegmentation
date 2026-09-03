"""Cuts one short, browser-playable H.264 clip per discovered action segment
so they can be inspected in a gr.Gallery."""

from __future__ import annotations

import os
import subprocess

from vlm_segment import ActionSegment


def _clear_dir(out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    for fn in os.listdir(out_dir):
        try:
            os.remove(os.path.join(out_dir, fn))
        except OSError:
            pass


def export_action_clips(video_path: str, segments: list[ActionSegment], out_dir: str, min_clip_s: float = 0.3) -> list[tuple[str, str]]:
    _clear_dir(out_dir)

    items = []
    for i, seg in enumerate(segments):
        start_t, end_t = seg.start_t, seg.end_t
        if end_t - start_t < min_clip_s:
            mid = (start_t + end_t) / 2
            start_t = max(0.0, mid - min_clip_s / 2)
            end_t = mid + min_clip_s / 2

        out_path = os.path.join(out_dir, f"action_{i:02d}.mp4")
        subprocess.run(
            ["ffmpeg", "-y", "-i", video_path, "-ss", f"{start_t:.3f}", "-to", f"{end_t:.3f}",
             "-an", "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
             "-movflags", "+faststart", out_path],
            check=True, capture_output=True,
        )
        caption = f"#{i + 1}  {seg.start_t:.2f}-{seg.end_t:.2f}s ({seg.duration_s:.2f}s): {seg.label}"
        items.append((out_path, caption))
    return items
