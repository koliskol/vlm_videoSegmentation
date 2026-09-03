"""
VideoLLaMA3-2B session management and short-clip captioning/classification.

Loaded lazily (only when requested from the UI) since it needs ~5GB VRAM and
a few seconds to load.
"""

from __future__ import annotations

import os
import subprocess
import tempfile
from dataclasses import dataclass

import torch
from transformers import AutoModelForCausalLM, AutoProcessor

MODEL_NAME = "DAMO-NLP-SG/VideoLLaMA3-2B"

# Resolution/frame budget tuned empirically on a 12GB GPU with no
# flash-attention available: this model's vision encoder computes dense
# (non-block-sparse) attention jointly across every sampled frame's patches,
# so attention memory scales with (frames x patches/frame)^2 -- resolution
# and frame count trade directly against each other. Measured safe ceilings
# before OOM: 224px width -> 6 frames (~5GB), 160px -> 12 frames (~5.3GB),
# 128px -> 16 frames (~5GB). LABEL_WIDTH/LABEL_MAX_FRAMES below is the
# few-frames/high-detail point (used for the final per-segment caption);
# LOWRES_WIDTH is for callers that need more frames per window at reduced
# detail (used for the sliding-window boundary detection in
# vlm_segment.py). Going past these ceilings reliably OOMs -- verified
# experimentally on that hardware; re-test before raising them on different
# hardware.
LABEL_WIDTH = 224
LABEL_FPS = 4
LABEL_MAX_FRAMES = 6
LOWRES_WIDTH = 160
LOWRES_MAX_FRAMES = 12
MIN_CLIP_S = 0.15  # pad very short clips so at least one frame is sampled

DEFAULT_QUESTION = (
    "In one short phrase, name the specific action the person is performing in this clip."
)
# NOTE: a phrasing that appended a list of quoted example labels (e.g.
# "'picking up a component', 'aligning parts', ...") caused this 2B model to
# degenerate into echoing one fixed example verbatim regardless of the actual
# clip content. The plain, example-free phrasing above produces answers that
# actually vary with what's on screen -- keep it that way.


@dataclass
class VlmSession:
    model: object
    processor: object
    source_video_path: str  # original video this session's *_video_path fields were built from
    small_video_path: str
    lowres_video_path: str
    video_duration_s: float


def _prepare_small_video(video_path: str, width: int) -> str:
    out_path = os.path.join(tempfile.gettempdir(), f"vlm_seg_{width}.mp4")
    subprocess.run(
        ["ffmpeg", "-y", "-i", video_path, "-vf", f"scale={width}:-2",
         "-c:v", "libx264", "-crf", "18", out_path],
        check=True, capture_output=True,
    )
    return out_path


def load_session(video_path: str, video_duration_s: float, existing: "VlmSession | None" = None) -> VlmSession:
    """Build a session for `video_path`. If `existing` is given, its (slow to
    load, ~5s + VRAM) model/processor are reused -- only the (cheap) resized
    video copies are regenerated. Callers should pass their current cached
    session here whenever the video might have changed, rather than
    unconditionally reusing it: a session's *_video_path fields are
    downscaled copies of ONE specific source video, so reusing a session
    against a different video would silently analyze the wrong footage."""
    if existing is not None:
        model, processor = existing.model, existing.processor
    else:
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME, trust_remote_code=True, device_map="cuda:0",
            torch_dtype=torch.bfloat16, attn_implementation="sdpa",
        )
        model.eval()
        processor = AutoProcessor.from_pretrained(MODEL_NAME, trust_remote_code=True)

    small_video_path = _prepare_small_video(video_path, width=LABEL_WIDTH)
    lowres_video_path = _prepare_small_video(video_path, width=LOWRES_WIDTH)
    return VlmSession(
        model=model, processor=processor, source_video_path=video_path,
        small_video_path=small_video_path, lowres_video_path=lowres_video_path,
        video_duration_s=video_duration_s,
    )


SIMPLIFY_QUESTION_TEMPLATE = (
    'In 1-3 words, name the core physical action in this description, '
    'stripped of any object or location detail. Description: "{caption}". '
    'Answer with ONLY the short action phrase, nothing else.'
)
# NOTE: an earlier version of this prompt appended a list of quoted examples
# (e.g. "'walking', 'squatting', 'standing', ..."), matching the pattern
# that caused this 2B model to degenerate/echo in two earlier, unrelated
# prompts (DEFAULT_QUESTION, the DWELL/MOVE constrained-vocab checks). This
# text-to-text task didn't degenerate either way when tested, but the
# example-free version also gave a MORE CORRECT answer on one case -- for a
# caption about fastening a screw with a screwdriver, the examples version
# extracted "screwdriver" (the tool, not the action); this version correctly
# said "fastening". Keep it example-free.


def simplify_caption(session: VlmSession, caption: str) -> str:
    """Text-only (no video) reduction of an already-generated caption to a
    short action phrase, e.g. "The person is squatting down to pick up a
    box." -> "squatting". Meant to run after a segment's full caption is
    already trusted (post-recaption), to produce a normalized label that's
    both easier to read and more reliable to compare during the merge
    pass (vlm_segment.merge_similar_neighbors) than the full sentence."""
    conversation = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": [
            {"type": "text", "text": SIMPLIFY_QUESTION_TEMPLATE.format(caption=caption)},
        ]},
    ]
    inputs = session.processor(conversation=conversation, return_tensors="pt")
    inputs = {k: (v.cuda() if isinstance(v, torch.Tensor) else v) for k, v in inputs.items()}
    with torch.no_grad():
        output_ids = session.model.generate(**inputs, max_new_tokens=10, do_sample=False)
    response = session.processor.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
    return response.strip(' .').lower()


SAME_CAPTION_QUESTION_TEMPLATE = (
    'Do these two descriptions refer to the SAME physical action? '
    'A: "{caption_a}"  B: "{caption_b}" '
    'Answer with exactly one word: YES or NO.'
)


def compare_captions(session: VlmSession, caption_a: str, caption_b: str) -> "str | None":
    """Text-only comparison (no video attached) of two already-generated
    captions. Cheaper than, and for low-motion/visually-ambiguous content
    (e.g. two clips that both just show someone standing still) more
    reliable than, re-grounding the judgment in footage a second time --
    we already trust these captions, they came from a full-span,
    higher-resolution pass (see vlm_segment._recaption_full_span)."""
    conversation = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": [
            {"type": "text", "text": SAME_CAPTION_QUESTION_TEMPLATE.format(caption_a=caption_a, caption_b=caption_b)},
        ]},
    ]
    inputs = session.processor(conversation=conversation, return_tensors="pt")
    inputs = {k: (v.cuda() if isinstance(v, torch.Tensor) else v) for k, v in inputs.items()}
    with torch.no_grad():
        output_ids = session.model.generate(**inputs, max_new_tokens=20, do_sample=False)
    response = session.processor.batch_decode(output_ids, skip_special_tokens=True)[0].strip().upper()
    for choice in ("YES", "NO"):
        if choice in response:
            return choice
    return None


def classify_clip(
    session: VlmSession, start_t: float, end_t: float, question: str, choices: list[str],
    fps: float = LABEL_FPS, max_frames: int = LABEL_MAX_FRAMES, video_path: "str | None" = None,
) -> "str | None":
    """Ask a constrained question and return whichever of `choices` appears
    first in the response, or None if the model didn't produce any of them."""
    response = label_clip(session, start_t, end_t, question=question, fps=fps, max_frames=max_frames, video_path=video_path).upper()
    best_pos, best_choice = None, None
    for choice in choices:
        pos = response.find(choice)
        if pos != -1 and (best_pos is None or pos < best_pos):
            best_pos, best_choice = pos, choice
    return best_choice


def label_clip(
    session: VlmSession, start_t: float, end_t: float, question: str = DEFAULT_QUESTION,
    fps: float = LABEL_FPS, max_frames: int = LABEL_MAX_FRAMES, video_path: "str | None" = None,
) -> str:
    if end_t - start_t < MIN_CLIP_S:
        mid = (start_t + end_t) / 2
        start_t = max(0.0, mid - MIN_CLIP_S / 2)
        end_t = min(session.video_duration_s, mid + MIN_CLIP_S / 2)

    conversation = [
        {"role": "system", "content": "You are a helpful assistant analyzing a video of a person performing physical actions."},
        {"role": "user", "content": [
            {"type": "video", "video": {
                "video_path": video_path or session.small_video_path, "fps": fps,
                "max_frames": max_frames, "start_time": start_t, "end_time": end_t,
            }},
            {"type": "text", "text": question},
        ]},
    ]
    inputs = session.processor(conversation=conversation, return_tensors="pt")
    inputs = {k: (v.cuda() if isinstance(v, torch.Tensor) else v) for k, v in inputs.items()}
    if "pixel_values" in inputs:
        inputs["pixel_values"] = inputs["pixel_values"].to(torch.bfloat16)
    with torch.no_grad():
        output_ids = session.model.generate(**inputs, max_new_tokens=40, do_sample=False)
    return session.processor.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
