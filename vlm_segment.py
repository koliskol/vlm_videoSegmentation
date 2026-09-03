"""
General-purpose action segmentation using VideoLLaMA3-2B: slide a window
across the whole video and let the model discover + segment actions itself.

No pose estimation, no domain assumptions about the task -- only the video
file and a loaded VLM session -- so this works on any human-action video.

Algorithm:
  1. Slide a window of FRAMES_PER_WINDOW frames (at SAMPLE_FPS) across the
     video, moving STEP_FRAMES frames each step.
  2. Caption the first window -> that becomes the "anchor" for a new segment.
  3. For each next window, ask the VLM (visually, on that window's own clip)
     whether it shows the SAME action as the anchor caption. If yes, extend
     the current segment. If no, close it and start a new one anchored on
     this window's own caption.
  4. Once boundaries are final, recaption each segment over its FULL span at
     higher resolution (see _recaption_full_span) for a better label than
     the tiny boundary-detection window can produce.

Boundary precision is bounded by the slide step, not frame-accurate.

Window size is expressed in FRAMES, not seconds, because that's what
actually bounds this model's VRAM use (attention memory scales with
(frames x patches/frame)^2 -- see vlm_label.py's ceiling notes). Frames are
sampled from a 160px-wide copy of the video (VlmSession.lowres_video_path,
lower-res than the final captioning pass's 224px copy) specifically so more
frames per window fit in budget: tested safe up to 12 frames at 160px
before OOM on a 12GB GPU with no flash-attention.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import vlm_label

# Tested safe ceiling at vlm_label.LOWRES_WIDTH (160px) is 12 frames before
# OOM (see vlm_label.py's ceiling notes) -- default sits with headroom below
# that. 50% overlap (STEP_FRAMES = FRAMES_PER_WINDOW / 2) is the standard
# default for sliding-window action detection.
FRAMES_PER_WINDOW = 10
STEP_FRAMES = 5
SAMPLE_FPS = 5.0  # frame spacing within a window; window_s = frames / fps

CAPTION_QUESTION = (
    "In one short phrase, name the specific action the person is performing in this clip."
)
SAME_ACTION_QUESTION_TEMPLATE = (
    'Watch this clip. Is the person doing the SAME action as this description: '
    '"{anchor_caption}"? Answer with exactly one word: YES or NO.'
)
SAME_ACTION_CHOICES = ["YES", "NO"]


@dataclass
class ActionSegment:
    label: str
    start_t: float
    end_t: float
    short_label: str = ""  # set by simplify_captions(); "" until then

    @property
    def duration_s(self) -> float:
        return self.end_t - self.start_t

    @property
    def merge_key(self) -> str:
        """What merge_similar_neighbors compares: the simplified label if
        available (shorter, more normalized -> more reliable text
        comparison), else the full caption."""
        return self.short_label or self.label


def slide_windows(duration_s: float, window_s: float, step_s: float) -> list[tuple[float, float]]:
    if duration_s <= 0:
        return []
    windows = []
    t = 0.0
    while True:
        end = min(t + window_s, duration_s)
        windows.append((t, end))
        if end >= duration_s:
            break
        t += step_s
    return windows


def _recaption_full_span(session: vlm_label.VlmSession, segments: list[ActionSegment]) -> None:
    """Replace each segment's label (originally just the tiny anchor
    window's caption) with a fresh caption over the segment's FULL time
    span, using the sharper 224px/6-frame settings instead of the small
    window used for boundary detection.

    Why: a 2-4 frame / 160px sliver is enough for the model to judge "same
    action as X?" reliably (that's what drives the actual segmentation),
    but it's too little context for it to describe WHAT the action
    specifically is -- it falls back to a generic scene-level gist almost
    every time. Captioning the full merged span at higher resolution gives
    it more to work with. This helps, but only partially: on fast, low-res,
    visually repetitive footage with short segments, a 2B model will still
    land on generic phrasing fairly often -- that's a real capability
    ceiling, not something a better prompt reliably fixes."""
    for seg in segments:
        seg.label = vlm_label.label_clip(
            session, seg.start_t, seg.end_t, question=CAPTION_QUESTION,
            fps=vlm_label.LABEL_FPS, max_frames=vlm_label.LABEL_MAX_FRAMES,
            video_path=session.small_video_path,
        )


def simplify_captions(
    session: vlm_label.VlmSession,
    segments: list[ActionSegment],
    progress_cb: Optional[Callable[[float, int, int], None]] = None,
) -> None:
    """Reduce each segment's full caption to a short action phrase (sets
    `.short_label`), e.g. "The person is squatting down to pick up a box."
    -> "squatting". Text-only, no video re-examined -- mainly a readability
    aid; meant to run after recaptioning (segment_by_action already does
    this) and before merge_similar_neighbors, which uses the short label
    ONLY for a free exact-string-match short-circuit (see
    ActionSegment.merge_key and the note in merge_similar_neighbors on why
    it doesn't trust the model's judgment on short-label comparisons
    beyond that)."""
    n = len(segments)
    for i, seg in enumerate(segments):
        if progress_cb:
            progress_cb((i + 1) / max(n, 1), i + 1, n)
        seg.short_label = vlm_label.simplify_caption(session, seg.label)


def merge_similar_neighbors(
    session: vlm_label.VlmSession,
    segments: list[ActionSegment],
    progress_cb: Optional[Callable[[float, int, int], None]] = None,
) -> list[ActionSegment]:
    """Second consolidation pass over already-divided, already-recaptioned
    segments: walk neighbors in order and decide if each pair is the SAME
    action. Two tiers, cheapest/most-trustworthy first:
      1. Exact match on `merge_key` (short label if simplify_captions() ran,
         else the full caption) -> merge immediately, no VLM call.
      2. Otherwise, ask the VLM via a TEXT-only comparison of the two
         segments' FULL captions (never the short labels beyond tier 1 --
         see the note below on why).
    A "yes" merges them; a "no" keeps them separate.

    Why tier 2 always uses the full caption, never the short label: testing
    found the model's judgment on bare short-word comparisons is
    unreliable even for genuine synonyms ("squatting" vs "crouching" ->
    NO) and, worse, for literally identical short words ("assembling" vs
    "assembling" -> NO, which is why tier 1's exact-match short-circuit
    exists at all -- without it, identical short labels would get wrongly
    split apart). The same model answers correctly on full sentences.

    Why compare captions instead of re-grounding in footage a second time
    (an earlier version of this function did that, the same way the divide
    step's YES/NO check works): on low-motion/visually-ambiguous content --
    e.g. two clips that both just show someone standing still -- re-showing
    the model footage produced inconsistent, unreliable "different" verdicts
    even when the two segments' full-span captions were IDENTICAL text.
    We already trust those captions (they came from a full-span, higher-
    resolution pass), so comparing them directly is both cheaper (no video
    processing at all) and more reliable than asking the model to
    re-discriminate between two clips that may carry very little visual
    signal to begin with.

    NOTE: this only ever merges ADJACENT segments, by design -- if the same
    action recurs later with something different in between (A, B, A), the
    two A's stay separate rather than merging, since merging means
    extending one contiguous time range and a non-adjacent "same caption"
    would wrongly swallow whatever sits between them.

    Why this is a separate pass rather than just using a smaller step size
    in segment_by_action: the divide step's tiny 2-4 frame / 160px windows
    are tuned to be sensitive to change, which means they can also be
    over-sensitive -- splitting what's really one continuous action into
    several pieces because two adjacent slivers looked different enough by
    themselves.

    Only genuinely-merged segments are recaptioned afterward; segments that
    never got merged with anything keep their already-good label instead of
    paying for a redundant caption call."""
    if len(segments) <= 1:
        return segments

    merged = [segments[0]]
    was_merged = [False]
    n = len(segments)

    for i in range(1, n):
        if progress_cb:
            progress_cb(i / max(n - 1, 1), i + 1, n)

        seg = segments[i]
        last = merged[-1]

        if last.merge_key.strip().lower() == seg.merge_key.strip().lower():
            # Free win, no VLM call needed. Also sidesteps a real failure
            # mode found by testing: compare_captions on bare short words
            # (post-simplify_captions) is unreliable even for genuine
            # synonyms ("squatting" vs "crouching" -> NO) and, worse, for
            # LITERALLY IDENTICAL short words ("assembling" vs
            # "assembling" -> NO). The model answers fine on full
            # sentences; something about terse single-word "descriptions"
            # specifically breaks the comparison. So: exact match on the
            # short label short-circuits here; anything else falls back to
            # comparing the full captions below, never the short labels.
            answer = "YES"
        else:
            answer = vlm_label.compare_captions(session, last.label, seg.label)

        if answer == "YES":
            merged[-1] = ActionSegment(label=last.label, start_t=last.start_t, end_t=seg.end_t, short_label=last.short_label)
            was_merged[-1] = True
        else:
            merged.append(seg)
            was_merged.append(False)

    to_recaption = [s for s, m in zip(merged, was_merged) if m]
    if to_recaption:
        _recaption_full_span(session, to_recaption)
        # short_label is now stale for these (it reflects the pre-merge
        # span's caption) -- refresh it against the new full-span caption.
        simplify_captions(session, to_recaption)

    return merged


def segment_by_action(
    session: vlm_label.VlmSession,
    duration_s: float,
    frames_per_window: int = FRAMES_PER_WINDOW,
    step_frames: int = STEP_FRAMES,
    fps: float = SAMPLE_FPS,
    progress_cb: Optional[Callable[[float, int, int], None]] = None,
    recaption_full_span: bool = True,
) -> list[ActionSegment]:
    window_s = frames_per_window / fps
    step_s = step_frames / fps
    video_path = session.lowres_video_path

    def _caption(w_start, w_end):
        return vlm_label.label_clip(session, w_start, w_end, question=CAPTION_QUESTION,
                                     fps=fps, max_frames=frames_per_window, video_path=video_path)

    def _same_action(w_start, w_end, anchor_caption):
        question = SAME_ACTION_QUESTION_TEMPLATE.format(anchor_caption=anchor_caption)
        return vlm_label.classify_clip(session, w_start, w_end, question, SAME_ACTION_CHOICES,
                                        fps=fps, max_frames=frames_per_window, video_path=video_path)

    windows = slide_windows(duration_s, window_s, step_s)
    if not windows:
        return []

    n = len(windows)
    anchor_caption = _caption(*windows[0])
    seg_start, seg_end = windows[0]
    segments: list[ActionSegment] = []

    for i in range(1, n):
        w_start, w_end = windows[i]
        if progress_cb:
            progress_cb(i / max(n - 1, 1), i + 1, n)

        answer = _same_action(w_start, w_end, anchor_caption)

        if answer == "YES":
            seg_end = w_end
        else:
            # Windows overlap (step < window length), so seg_end may already
            # extend past this window's start. Cut the boundary exactly at
            # w_start so consecutive segments are contiguous, never
            # overlapping -- the transition is attributed to the start of
            # the first window that showed a different action.
            segments.append(ActionSegment(label=anchor_caption, start_t=seg_start, end_t=w_start))
            anchor_caption = _caption(w_start, w_end)
            seg_start, seg_end = w_start, w_end

    segments.append(ActionSegment(label=anchor_caption, start_t=seg_start, end_t=seg_end))

    if recaption_full_span:
        if progress_cb:
            progress_cb(1.0, n, n)
        _recaption_full_span(session, segments)

    return segments
