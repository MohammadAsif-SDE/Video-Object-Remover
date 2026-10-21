"""Core pipeline: lasso polygon -> (track the object through the video) -> fill every frame ->
H.264 with the original audio.

Frames are always read sequentially. OpenCV's CAP_PROP_POS_FRAMES seek is unreliable on many
H.264 files (it can land seconds away from the requested frame), so single frames are fetched
with ffmpeg's accurate seek and tracking runs on an in-memory, low-resolution copy of the video
(`VideoIndex`) that can be walked in either direction.
"""

import queue
import subprocess
import tempfile
import threading
from collections import deque
from typing import Callable, List, Optional, Sequence, Tuple

import cv2
import imageio_ffmpeg
import numpy as np

Point = Tuple[float, float]
Box = Tuple[float, float, float, float]  # x, y, w, h
ProgressFn = Optional[Callable[[float], None]]
CancelFn = Optional[Callable[[], bool]]

FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()


def video_meta(video_path: str) -> dict:
    """fps, width, height and frame count. Width/height come from a decoded frame, so they
    always match what `cv2.VideoCapture.read()` returns (rotation applied)."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    try:
        ok, frame = cap.read()
        if not ok:
            raise RuntimeError("Could not read a frame from the video")
        return {
            "fps": cap.get(cv2.CAP_PROP_FPS) or 30.0,
            "width": int(frame.shape[1]),
            "height": int(frame.shape[0]),
            "frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        }
    finally:
        cap.release()


def read_frame(video_path: str, index: int = 0) -> Tuple[np.ndarray, dict]:
    """Return frame `index` (BGR) plus basic metadata, using ffmpeg's frame-accurate seek."""
    meta = video_meta(video_path)
    w, h = meta["width"], meta["height"]
    index = max(0, min(int(index), max(meta["frames"] - 1, 0)))
    # Ask for a time a quarter frame early: the first frame at/after it is exactly `index`.
    t = max(index - 0.25, 0.0) / meta["fps"]
    cmd = [FFMPEG, "-v", "error", "-ss", f"{t:.6f}", "-i", video_path, "-frames:v", "1",
           "-f", "rawvideo", "-pix_fmt", "bgr24", "-"]
    res = subprocess.run(cmd, capture_output=True)
    if res.returncode != 0 or len(res.stdout) < w * h * 3:
        raise RuntimeError(f"Could not read frame {index}: {res.stderr.decode(errors='replace').strip()}")
    frame = np.frombuffer(res.stdout[: w * h * 3], np.uint8).reshape(h, w, 3).copy()
    return frame, meta


def polygon_to_mask(points: Sequence[Point], h: int, w: int, feather_px: int = 10) -> np.ndarray:
    """Rasterise a closed lasso polygon into a 0/255 mask, dilated by `feather_px`."""
    if len(points) < 3:
        raise ValueError("A lasso needs at least 3 points")
    mask = np.zeros((h, w), dtype=np.uint8)
    pts = np.round(np.asarray(points, dtype=np.float32)).astype(np.int32).reshape(-1, 1, 2)
    cv2.fillPoly(mask, [pts], 255)
    if feather_px > 0:
        k = 2 * feather_px + 1
        mask = cv2.dilate(mask, np.ones((k, k), np.uint8))
    return mask


def _odd(n: int) -> int:
    n = int(n)
    return n if n % 2 == 1 else n + 1


def _stretch_horizontal(sub: np.ndarray, m: np.ndarray) -> np.ndarray:
    """For every masked pixel, linearly interpolate between the nearest unmasked pixel to its
    left and to its right on the same row. Produces the 'pixel stretch' look: the fill
    starts from exactly the neighbouring colours, so it joins the scene without a seam.
    `sub` is float32 HxWx3, `m` is bool HxW. Fully vectorised."""
    H, W = m.shape
    idx = np.broadcast_to(np.arange(W), (H, W))
    left = np.maximum.accumulate(np.where(m, -1, idx), axis=1)
    right = np.minimum.accumulate(np.where(m, W, idx)[:, ::-1], axis=1)[:, ::-1]
    has_l, has_r = left >= 0, right < W
    l_idx = np.where(has_l, left, right).clip(0, W - 1)
    r_idx = np.where(has_r, right, left).clip(0, W - 1)

    src = cv2.GaussianBlur(sub, (3, 3), 0)  # sample slightly denoised edge colours
    rows = np.arange(H)[:, None]
    cl, cr = src[rows, l_idx], src[rows, r_idx]
    span = (r_idx - l_idx).astype(np.float32)
    span[span == 0] = 1.0
    t = np.clip((idx - l_idx) / span, 0.0, 1.0)[..., None]
    fill = cl * (1.0 - t) + cr * t

    out = sub.copy()
    valid = m & (has_l | has_r)
    out[valid] = fill[valid]
    return out


def stretch_fill(frame: np.ndarray, mask: np.ndarray, direction: str = "horizontal") -> np.ndarray:
    """Fill the mask by stretching neighbouring pixels across it.

    direction: "horizontal" (left<->right streaks), "vertical" (top<->bottom), or "both"
    (average of the two, smoother and less streaky).
    """
    h, w = mask.shape
    x, y, bw, bh = cv2.boundingRect(mask)
    if bw == 0 or bh == 0:
        return frame
    pad = 6  # make sure the neighbouring (unmasked) pixels are inside the working window
    x0, y0 = max(x - pad, 0), max(y - pad, 0)
    x1, y1 = min(x + bw + pad, w), min(y + bh + pad, h)
    sub = frame[y0:y1, x0:x1].astype(np.float32)
    m = mask[y0:y1, x0:x1] > 0

    if direction == "horizontal":
        filled = _stretch_horizontal(sub, m)
    elif direction == "vertical":
        filled = _stretch_horizontal(sub.transpose(1, 0, 2), m.T).transpose(1, 0, 2)
    else:
        fh = _stretch_horizontal(sub, m)
        fv = _stretch_horizontal(sub.transpose(1, 0, 2), m.T).transpose(1, 0, 2)
        filled = (fh + fv) * 0.5

    out = frame.copy()
    out[y0:y1, x0:x1] = np.clip(filled, 0, 255).astype(np.uint8)
    return out


def fill_region(
    frame: np.ndarray,
    mask: np.ndarray,
    mode: str = "stretch",
    direction: str = "horizontal",
    inpaint_radius: int = 7,
    blur_ksize: int = 41,
) -> np.ndarray:
    """Fill the masked region so the object disappears.

    mode="stretch": stretch the surrounding pixels across the region (seamless smear).
    mode="inpaint": reconstruct the background from surrounding pixels (Telea),
                    then soften the patch so it sinks into the scene.
    mode="blur":    only blur the region (classic censor look).
    `blur_ksize` ("softness") controls how much smoothing is applied inside the patch.
    """
    blur_ksize = _odd(blur_ksize)
    mask_f = mask.astype(np.float32) / 255.0

    if mode == "stretch":
        base = stretch_fill(frame, mask, direction)
        # Smooth across the streaks (perpendicular to the stretch) to hide row-to-row jitter.
        k = _odd(max(blur_ksize // 4, 1))
        if direction == "horizontal":
            blurred = cv2.GaussianBlur(base, (1, k), 0)
        elif direction == "vertical":
            blurred = cv2.GaussianBlur(base, (k, 1), 0)
        else:
            blurred = cv2.GaussianBlur(base, (k, k), 0)
        alpha = cv2.GaussianBlur(mask_f, (9, 9), 0)[..., None]
    elif mode == "blur":
        base = frame
        blurred = cv2.GaussianBlur(base, (blur_ksize, blur_ksize), 0)
        alpha = cv2.GaussianBlur(mask_f, (blur_ksize, blur_ksize), 0)[..., None]
    else:  # inpaint
        base = cv2.inpaint(frame, mask, inpaint_radius, cv2.INPAINT_TELEA)
        blurred = cv2.GaussianBlur(base, (blur_ksize, blur_ksize), 0)
        # Keep most of the inpainted texture; only blur a little to hide Telea's streaks.
        alpha = (cv2.GaussianBlur(mask_f, (blur_ksize, blur_ksize), 0) * 0.4)[..., None]

    out = base.astype(np.float32) * (1.0 - alpha) + blurred.astype(np.float32) * alpha
    return np.clip(out, 0, 255).astype(np.uint8)



def selection_box(points: Sequence[Point]) -> Box:
    """Bounding box of the (undilated) selection polygon."""
    pts = np.asarray(points, np.float32)
    x0, y0 = pts.min(0)
    x1, y1 = pts.max(0)
    return float(x0), float(y0), float(max(x1 - x0, 1.0)), float(max(y1 - y0, 1.0))


def place_points(points: Sequence[Point], box0: Box, box: Box) -> np.ndarray:
    """Move/scale the selection polygon from where it was drawn (box0) to `box`."""
    pts = np.asarray(points, np.float32)
    sx, sy = box[2] / box0[2], box[3] / box0[3]
    return np.stack([box[0] + (pts[:, 0] - box0[0]) * sx, box[1] + (pts[:, 1] - box0[1]) * sy], axis=1)


def roi_mask(shape, points: np.ndarray, feather_px: int, mode: str, blur_ksize: int):
    """Rasterise the selection into a mask covering only the region around it.

    Returns ((x0, y0, x1, y1), mask) or None if the selection is outside the frame. The region
    is padded enough that every filter in `fill_region` sees real pixels around the mask."""
    h, w = shape[:2]
    k = _odd(blur_ksize)
    pad = feather_px + 16 + (k // 4 if mode == "stretch" else k // 2)
    x0 = max(int(np.floor(points[:, 0].min() - pad)), 0)
    y0 = max(int(np.floor(points[:, 1].min() - pad)), 0)
    x1 = min(int(np.ceil(points[:, 0].max() + pad)), w)
    y1 = min(int(np.ceil(points[:, 1].max() + pad)), h)
    if x1 - x0 < 2 or y1 - y0 < 2:
        return None
    mask = polygon_to_mask(points - np.float32([x0, y0]), y1 - y0, x1 - x0, feather_px)
    if not mask.any():
        return None
    return (x0, y0, x1, y1), mask


def fill_frame(frame: np.ndarray, points: np.ndarray, feather_px: int = 20, mode: str = "stretch",
               direction: str = "horizontal", blur_ksize: int = 41) -> np.ndarray:
    """Fill the selection in a copy of `frame`, processing only the region around it."""
    out = frame.copy()
    r = roi_mask(frame.shape, points, feather_px, mode, blur_ksize)
    if r is not None:
        (x0, y0, x1, y1), mask = r
        out[y0:y1, x0:x1] = fill_region(frame[y0:y1, x0:x1], mask, mode=mode, direction=direction,
                                        blur_ksize=blur_ksize)
    return out


def _ring(mask: np.ndarray, width: int = 20) -> np.ndarray:
    """Boolean band just outside the mask, used to measure how much the surroundings move."""
    k = np.ones((2 * width + 1, 2 * width + 1), np.uint8)
    ring = (cv2.dilate(mask, k) > 0) & ~(mask > 0)
    if ring.sum() < 100:  # mask fills its whole region: fall back to everything
        ring = np.ones(mask.shape, dtype=bool)
    return ring


def estimate_object_mask(frame: np.ndarray, selection: np.ndarray, ring_px: int = 14) -> np.ndarray:
    """Guess which pixels inside the user's selection are the object itself.

    Colours that also occur in a ring just outside the selection are assumed to be
    background. Returns a 0/255 mask; falls back to the whole selection when the guess is
    implausible (almost nothing or almost everything classified as object).
    """
    sel = selection > 0
    k = np.ones((2 * ring_px + 1, 2 * ring_px + 1), np.uint8)
    ring = (cv2.dilate(selection, k) > 0) & ~sel
    if ring.sum() < 50 or sel.sum() < 50:
        return selection
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB).astype(np.float32)
    bg = lab[ring]
    if len(bg) > 4000:
        bg = bg[np.random.default_rng(0).choice(len(bg), 4000, replace=False)]
    K = min(8, len(bg))
    _, _, centers = cv2.kmeans(bg, K, None, (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0),
                               2, cv2.KMEANS_PP_CENTERS)
    inside = lab[sel]  # N x 3
    d = np.sqrt(((inside[:, None, :] - centers[None, :, :]) ** 2).sum(-1)).min(1)
    obj = np.zeros(selection.shape, np.uint8)
    obj[sel] = (d > 22.0).astype(np.uint8) * 255
    obj = cv2.morphologyEx(obj, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    obj = cv2.morphologyEx(obj, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    frac = (obj > 0).sum() / max(sel.sum(), 1)
    if frac < 0.03 or frac > 0.95:
        return selection
    return obj


COLOR_TOL = 60.0  # mean abs colour difference (0-255) at which colour similarity reaches 0


def _masked_ncc(a: np.ndarray, b: np.ndarray, m: np.ndarray) -> float:
    """Similarity in [~-1, 1] between two equally sized colour patches, using only the pixels
    where `m` is True (the object itself, not the background around it).

    Combines normalised cross-correlation (structure) with a plain colour-difference term,
    because NCC is meaningless for flat-coloured objects such as solid arrows or logos."""
    if a.shape != b.shape or m.sum() < 16:
        return 0.0
    av = a[m].astype(np.float32).reshape(-1, a.shape[-1] if a.ndim == 3 else 1)
    bv = b[m].astype(np.float32).reshape(av.shape)
    color_sim = float(np.clip(1.0 - np.abs(av - bv).mean() / COLOR_TOL, 0.0, 1.0))
    # Centre each channel separately so a flat-coloured patch has zero variance (ncc -> 0)
    # instead of degenerating into a "which channel is brightest" match.
    av -= av.mean(axis=0)
    bv -= bv.mean(axis=0)
    d = float(np.sqrt((av * av).sum() * (bv * bv).sum()))
    ncc = float((av * bv).sum() / d) if d > 1e-6 else 0.0
    return max(ncc, color_sim)


# ---------------------------------------------------------------------------------------------
# Low-resolution index of the whole video
# ---------------------------------------------------------------------------------------------

class VideoIndex:
    """Every frame of the video, downscaled and JPEG-compressed in memory, plus shot boundaries.

    Built with one sequential decode (~600 fps for 1080p). Tracking runs on this copy: it is
    several times faster than full resolution, needs no seeking, and can go backwards.
    """

    CUT_HIST = 0.35       # colour-histogram distance between adjacent frames that means a hard cut
    CUT_CORR = 0.60       # ... or structural correlation (edges/layout) below this
    DISSOLVE_HIST = 0.50  # histogram distance across a quarter second that means a dissolve/flash

    def __init__(self, video_path: str, progress: ProgressFn = None, should_cancel: CancelFn = None):
        meta = video_meta(video_path)
        self.fps = meta["fps"]
        self.width, self.height = meta["width"], meta["height"]
        n_est = max(meta["frames"], 1)
        long_side = 640 if n_est <= 6000 else 480 if n_est <= 15000 else 360  # bound memory
        self.scale = min(1.0, long_side / max(self.width, self.height))
        size = (max(int(round(self.width * self.scale)), 1), max(int(round(self.height * self.scale)), 1))

        self._frames: List[np.ndarray] = []
        cuts = [0]
        gap = max(int(round(self.fps / 4)), 2)
        hists: deque = deque(maxlen=gap + 1)
        prev_tiny = None
        cap = cv2.VideoCapture(video_path)
        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                if should_cancel and should_cancel():
                    raise InterruptedError("Cancelled")
                small = cv2.resize(frame, size, interpolation=cv2.INTER_AREA)
                self._frames.append(cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, 85])[1])
                hist = cv2.calcHist([cv2.cvtColor(small, cv2.COLOR_BGR2HSV)], [0, 1], None, [16, 8],
                                    [0, 180, 0, 256])
                cv2.normalize(hist, hist)
                gray = cv2.resize(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY), (36, 64), interpolation=cv2.INTER_AREA)
                gray = gray.astype(np.float32)
                tiny = (gray - gray.mean()) / gray.std() if gray.std() > 2 else None  # None: flat frame
                i = len(self._frames) - 1
                if hists:
                    hard = cv2.compareHist(hists[-1], hist, cv2.HISTCMP_BHATTACHARYYA) > self.CUT_HIST
                    if tiny is not None and prev_tiny is not None:
                        hard = hard or float((tiny * prev_tiny).mean()) < self.CUT_CORR
                    # Dissolves and flashes change the picture gradually, each step below the
                    # hard-cut threshold, so also compare against a quarter second ago.
                    soft = len(hists) > gap and i - cuts[-1] > gap and \
                        cv2.compareHist(hists[0], hist, cv2.HISTCMP_BHATTACHARYYA) > self.DISSOLVE_HIST
                    if (hard or soft) and cuts[-1] != i:
                        cuts.append(i)
                hists.append(hist)
                prev_tiny = tiny
                if progress and len(self._frames) % 30 == 0:
                    progress(min(len(self._frames) / n_est, 0.99))
        finally:
            cap.release()
        self.n = len(self._frames)
        if not self.n:
            raise RuntimeError("Could not decode any frames")
        bounds = cuts + [self.n]
        self.shots = [(a, b) for a, b in zip(bounds, bounds[1:]) if b > a]
        if progress:
            progress(1.0)

    def frame(self, i: int) -> np.ndarray:
        return cv2.imdecode(self._frames[i], cv2.IMREAD_COLOR)

    def shot_of(self, i: int) -> Tuple[int, int]:
        for a, b in self.shots:
            if a <= i < b:
                return a, b
        return 0, self.n


# ---------------------------------------------------------------------------------------------
# Tracking
# ---------------------------------------------------------------------------------------------

PIN_SIM = 0.85       # object still matches at the spot it started from -> it hasn't moved; stay put
CONTINUE_SIM = 0.75  # at a shot cut, keep following only if the object is still clearly there
LOST_SIM = 0.35      # below this (vs. the drawn object) for a quarter second, the tracker has lost it
SEARCH_SIM = 0.72    # coarse whole-frame match needed to consider re-finding the object in another shot
CONFIRM_SIM = 0.80   # ... and the match it must reach at tracking resolution to be accepted
SEARCH_SCALE = 0.5


def _int_box(box: Box, w: int, h: int) -> Tuple[int, int, int, int]:
    """Round a box and clip it to the frame (trackers reject boxes that leave the image)."""
    x, y, bw, bh = box
    x1, y1 = min(max(int(round(x)), 0), w - 4), min(max(int(round(y)), 0), h - 4)
    x2, y2 = min(max(int(round(x + bw)), x1 + 4), w), min(max(int(round(y + bh)), y1 + 4), h)
    return x1, y1, x2 - x1, y2 - y1


def _patch(frame: np.ndarray, box: Box, shape) -> Optional[np.ndarray]:
    """The part of `frame` under `box`, resized to `shape`; None if the box is mostly off-frame."""
    h, w = frame.shape[:2]
    x, y, bw, bh = [int(round(v)) for v in box]
    x1, y1, x2, y2 = max(x, 0), max(y, 0), min(x + bw, w), min(y + bh, h)
    if bw < 4 or bh < 4 or x2 - x1 < bw * 0.5 or y2 - y1 < bh * 0.5:
        return None
    p = frame[y1:y2, x1:x2]
    return p if p.shape[:2] == shape[:2] else cv2.resize(p, (shape[1], shape[0]))


def _sim(patch: Optional[np.ndarray], ref: Optional[np.ndarray], m: np.ndarray) -> float:
    return _masked_ncc(patch, ref, m) if patch is not None and ref is not None else 0.0


class _Target:
    """What the user selected, as seen on the frame they drew it on (tracking resolution)."""

    def __init__(self, frame: np.ndarray, box: Tuple[int, int, int, int], selection: np.ndarray):
        x, y, bw, bh = box
        self.box = box
        self.template = frame[y:y + bh, x:x + bw].copy()
        obj = estimate_object_mask(frame, selection)
        self.tmask = obj[y:y + bh, x:x + bw] > 0
        if self.tmask.sum() < 16:
            self.tmask = np.ones((bh, bw), dtype=bool)
        s = SEARCH_SCALE
        self.small = cv2.resize(self.template, (max(int(bw * s), 2), max(int(bh * s), 2)))
        self.small_mask = cv2.resize(self.tmask.astype(np.uint8), self.small.shape[1::-1],
                                     interpolation=cv2.INTER_NEAREST)

    def search(self, frame: np.ndarray) -> Tuple[float, Box]:
        """Best match for the object anywhere in `frame`: (score, box)."""
        s = SEARCH_SCALE
        small = cv2.resize(frame, (max(int(frame.shape[1] * s), 1), max(int(frame.shape[0] * s), 1)))
        th, tw = self.small.shape[:2]
        if small.shape[0] <= th or small.shape[1] <= tw or self.small_mask.sum() < 4:
            return 0.0, tuple(map(float, self.box))
        ncc = cv2.matchTemplate(small, self.small, cv2.TM_CCOEFF_NORMED, mask=self.small_mask)
        ncc = np.nan_to_num(ncc, nan=-1.0, posinf=-1.0, neginf=-1.0)
        sq = cv2.matchTemplate(small, self.small, cv2.TM_SQDIFF, mask=self.small_mask)
        n_px = max(int(self.small_mask.sum()) * 3, 1)
        color_sim = np.clip(1.0 - np.sqrt(np.maximum(sq, 0) / n_px) / COLOR_TOL, 0.0, 1.0)
        _, best, _, loc = cv2.minMaxLoc(np.maximum(ncc, color_sim))
        return float(best), (loc[0] / s, loc[1] / s, float(self.box[2]), float(self.box[3]))


class _TemplateTracker:
    """Follows an object by matching its colour template (object pixels only) in a window
    around where it is expected next. Keeps the selection's size, adapts slowly to changes in
    appearance, and copes with flat-coloured graphics where correlation trackers drift."""

    ADAPT = 0.08  # how fast the template follows changes in appearance

    def __init__(self, frame: np.ndarray, box: Tuple[int, int, int, int], tmask: np.ndarray):
        x, y, self.w, self.h = box
        self.template = frame[y:y + self.h, x:x + self.w].astype(np.float32)
        if tmask.shape != (self.h, self.w):
            tmask = cv2.resize(tmask.astype(np.uint8), (self.w, self.h), interpolation=cv2.INTER_NEAREST) > 0
        self.mask = tmask.astype(np.uint8) if tmask.sum() >= 16 else np.ones((self.h, self.w), np.uint8)
        self.n_px = max(int(self.mask.sum()) * 3, 1)
        self.pos = np.array([x, y], np.float32)
        self.vel = np.zeros(2, np.float32)

    def update(self, frame: np.ndarray) -> Tuple[bool, Box]:
        fh, fw = frame.shape[:2]
        pred = self.pos + self.vel
        r = int(max(0.6 * max(self.w, self.h), 24))
        x0, y0 = int(max(pred[0] - r, 0)), int(max(pred[1] - r, 0))
        x1, y1 = int(min(pred[0] + self.w + r, fw)), int(min(pred[1] + self.h + r, fh))
        if x1 - x0 <= self.w or y1 - y0 <= self.h:
            return False, (float(self.pos[0]), float(self.pos[1]), float(self.w), float(self.h))
        win = frame[y0:y1, x0:x1].astype(np.float32)
        sq = cv2.matchTemplate(win, self.template, cv2.TM_SQDIFF, mask=self.mask)
        color_sim = np.clip(1.0 - np.sqrt(np.maximum(sq, 0) / self.n_px) / COLOR_TOL, 0.0, 1.0)
        ncc = cv2.matchTemplate(win, self.template, cv2.TM_CCOEFF_NORMED, mask=self.mask)
        score = np.maximum(color_sim, np.nan_to_num(ncc, nan=-1.0, posinf=-1.0, neginf=-1.0))
        # Slight preference for the predicted position breaks ties on repetitive backgrounds.
        yy, xx = np.mgrid[0:score.shape[0], 0:score.shape[1]]
        score -= 0.05 * np.hypot(xx + x0 - pred[0], yy + y0 - pred[1]) / (r + 1)
        _, best, _, loc = cv2.minMaxLoc(score)
        new = np.array([loc[0] + x0, loc[1] + y0], np.float32)
        self.vel = 0.5 * self.vel + 0.5 * (new - self.pos)
        self.pos = new
        nx, ny = int(new[0]), int(new[1])
        patch = frame[ny:ny + self.h, nx:nx + self.w]
        if best > 0.5 and patch.shape[:2] == self.template.shape[:2]:
            self.template += self.ADAPT * (patch.astype(np.float32) - self.template)
        return True, (float(new[0]), float(new[1]), float(self.w), float(self.h))


def _follow(index: VideoIndex, target: _Target, start: int, box: Box, step: int, stop: int,
            boxes: List[Optional[Box]], tick: Callable[[], None], should_cancel: CancelFn,
            cross_cuts: bool) -> None:
    """Track from `start` (object at `box`) one frame at a time towards `stop` (exclusive),
    writing boxes[i]. Ends at a shot cut unless `cross_cuts` and the object is still clearly
    there on the other side (so a false cut, e.g. a flash, doesn't end tracking). Also ends
    when the object has been lost: the box suddenly jumps (the tracker switched to something
    else, e.g. another person), or it has stopped looking like the object for a quarter second
    (those frames are un-written)."""
    frame = index.frame(start)
    h, w = frame.shape[:2]
    pin = _int_box(box, w, h)
    tracker = _TemplateTracker(frame, pin, target.tmask)
    last: Box = tuple(map(float, pin))
    pin_ref = _patch(frame, last, target.template.shape)  # the object where this run started
    # Recent appearances. At a cut we compare with how the object looked a quarter second
    # earlier, because one frame ago may already be halfway through a dissolve.
    quarter = max(int(round(index.fps / 4)), 2)
    history: deque = deque([pin_ref], maxlen=quarter)
    shot = index.shot_of(start)
    misses = 0
    max_jump = max(0.3 * min(pin[2], pin[3]), 12.0)  # tracking-resolution pixels per frame
    tracked_prev = False

    i = start + step
    while i != stop:
        if should_cancel and should_cancel():
            raise InterruptedError("Cancelled")
        frame = index.frame(i)
        # Guard against drift: if the object still sits where this run started, it hasn't moved
        # (typical for overlays on top of moving footage, which pull plain trackers away).
        if _sim(_patch(frame, pin, target.template.shape), pin_ref, target.tmask) >= PIN_SIM:
            cur: Box = tuple(map(float, pin))
            if cur != last:
                tracker = _TemplateTracker(frame, pin, target.tmask)
            tracked_prev = False
        else:
            ok, b = tracker.update(frame)
            cur = b if ok else last
            if tracked_prev and np.hypot(cur[0] - last[0], cur[1] - last[1]) > max_jump:
                return
            tracked_prev = True
        patch = _patch(frame, cur, target.template.shape)

        if not (shot[0] <= i < shot[1]):
            looks_same = max(_sim(patch, target.template, target.tmask), _sim(patch, pin_ref, target.tmask))
            still_there = min(_sim(patch, history[0], target.tmask), looks_same)
            if not cross_cuts or still_there < CONTINUE_SIM:
                return
            shot = index.shot_of(i)

        like = max(_sim(patch, target.template, target.tmask), _sim(patch, pin_ref, target.tmask))
        misses = misses + 1 if like < LOST_SIM else 0
        if misses > quarter:
            for j in range(i - step * (misses - 1), i, step):
                boxes[j] = None
            return

        boxes[i] = cur
        last = cur
        if patch is not None:
            history.append(patch)
        tick()
        i += step


def _smooth(boxes: List[Optional[Box]], radius: int = 2) -> List[Optional[Box]]:
    """Moving average over each continuous run of boxes, to take out tracker jitter."""
    out = list(boxes)
    i, n = 0, len(boxes)
    while i < n:
        if boxes[i] is None:
            i += 1
            continue
        j = i
        while j < n and boxes[j] is not None:
            j += 1
        run = np.asarray(boxes[i:j], np.float64)
        if len(run) > 2:
            pad = np.pad(run, ((radius, radius), (0, 0)), mode="edge")
            kernel = np.ones(2 * radius + 1) / (2 * radius + 1)
            run = np.stack([np.convolve(pad[:, c], kernel, mode="valid") for c in range(4)], axis=1)
        out[i:j] = [tuple(r) for r in run]
        i = j
    return out


def track_object(index: VideoIndex, points: Sequence[Point], start_frame: int,
                 progress: ProgressFn = None, should_cancel: CancelFn = None):
    """Follow the selected object through the whole video.

    The object is tracked forwards and backwards from `start_frame` through the shot it was
    drawn in. Every other shot is searched for it, and where it is clearly found it is tracked
    through that shot too. Returns (box0, boxes): box0 is the selection's bounding box and
    boxes[i] is where that box is in frame i (full-resolution pixels), or None where the object
    is not in the frame.
    """
    s = index.scale
    pts = np.asarray(points, np.float32)
    box0 = selection_box(points)
    start = max(0, min(int(start_frame), index.n - 1))
    frame0 = index.frame(start)
    h, w = frame0.shape[:2]
    sbox = _int_box((box0[0] * s, box0[1] * s, box0[2] * s, box0[3] * s), w, h)
    selection = polygon_to_mask(pts * s, h, w, 0)
    target = _Target(frame0, sbox, selection)

    boxes: List[Optional[Box]] = [None] * index.n
    boxes[start] = tuple(map(float, sbox))
    done = [0]

    def tick(k: int = 1):
        done[0] += k
        if progress and done[0] % 10 < k:
            progress(min(done[0] / index.n, 0.99))

    _follow(index, target, start, sbox, 1, index.n, boxes, tick, should_cancel, cross_cuts=True)
    _follow(index, target, start, sbox, -1, -1, boxes, tick, should_cancel, cross_cuts=True)

    for a, b in index.shots:
        if any(bx is not None for bx in boxes[a:b]):
            continue
        for k in sorted({a, a + (b - a) // 3, a + 2 * (b - a) // 3}):
            if should_cancel and should_cancel():
                raise InterruptedError("Cancelled")
            frame = index.frame(k)
            score, cand = target.search(frame)
            if score < SEARCH_SIM:
                continue
            cand = tuple(map(float, _int_box(cand, w, h)))
            if _sim(_patch(frame, cand, target.template.shape), target.template, target.tmask) < CONFIRM_SIM:
                continue
            boxes[k] = cand
            _follow(index, target, k, cand, 1, b, boxes, tick, should_cancel, cross_cuts=False)
            _follow(index, target, k, cand, -1, a - 1, boxes, tick, should_cancel, cross_cuts=False)
            break
        tick(sum(bx is None for bx in boxes[a:b]))

    # Back to full resolution, relative to the drawn box so a frame where the object did not
    # move gets exactly the selection the user drew.
    full: List[Optional[Box]] = []
    for bx in _smooth(boxes):
        if bx is None:
            full.append(None)
            continue
        full.append((box0[0] + (bx[0] - sbox[0]) / s, box0[1] + (bx[1] - sbox[1]) / s,
                     box0[2] * bx[2] / sbox[2], box0[3] * bx[3] / sbox[3]))
    if progress:
        progress(1.0)
    return box0, full


# ---------------------------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------------------------

class _VideoWriter:
    """Pipe raw frames into ffmpeg, which encodes H.264 and copies in the source's audio in the
    same pass (no intermediate file, no second encode)."""

    def __init__(self, out_path: str, w: int, h: int, fps: float, audio_from: str):
        self._log = tempfile.TemporaryFile()
        cmd = [
            FFMPEG, "-y", "-v", "error",
            "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{w}x{h}", "-r", f"{fps:.6f}", "-i", "-",
            "-i", audio_from,
            "-map", "0:v:0", "-map", "1:a:0?",
            "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2",
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "18", "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-b:a", "192k", "-shortest", "-movflags", "+faststart",
            out_path,
        ]
        self._proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=self._log)

    def _error(self) -> str:
        self._log.seek(0)
        return self._log.read().decode(errors="replace").strip()

    def write(self, frame: np.ndarray) -> None:
        try:
            self._proc.stdin.write(np.ascontiguousarray(frame).data)
        except BrokenPipeError:
            self._proc.wait()
            raise RuntimeError(f"ffmpeg failed: {self._error()}")

    def close(self) -> None:
        self._proc.stdin.close()
        if self._proc.wait() != 0:
            raise RuntimeError(f"ffmpeg failed: {self._error()}")

    def abort(self) -> None:
        self._proc.kill()
        self._proc.wait()


def _read_ahead(cap: cv2.VideoCapture, out: "queue.Queue", stop: threading.Event) -> None:
    """Decode frames on a background thread so decoding overlaps filling and encoding."""
    try:
        while not stop.is_set():
            ok, frame = cap.read()
            out.put(frame if ok else None)
            if not ok:
                return
    except BaseException as e:  # noqa: BLE001 - re-raised on the consuming thread
        out.put(e)


def remove_object(
    video_in: str,
    video_out: str,
    lasso_points: Sequence[Point],
    boxes: Optional[List[Optional[Box]]] = None,
    frame_range: Optional[Tuple[int, int]] = None,
    mode: str = "stretch",
    direction: str = "horizontal",
    feather_px: int = 20,
    blur_ksize: int = 41,
    temporal_smooth: float = 0.5,
    progress: ProgressFn = None,
    should_cancel: CancelFn = None,
) -> str:
    """Fill the selection in every frame of `video_in` and write `video_out` (H.264 + audio).

    `boxes` comes from `track_object`: the selection is moved to boxes[i] in frame i and frames
    with None are left untouched. Without `boxes` the selection is filled in place in every
    frame (fixed watermark / caption). `frame_range` (first, last), inclusive, limits the
    removal to those frames. Only the region around the selection is processed.
    `temporal_smooth` (0..1) blends the patch with the previous frame's when the selection did
    not move and its surroundings are static, which suppresses per-frame fill flicker.
    """
    meta = video_meta(video_in)
    w, h = meta["width"], meta["height"]
    n = max(meta["frames"], 1)
    first, last = frame_range if frame_range else (0, n + 1_000_000)
    box0 = selection_box(lasso_points)
    writer = _VideoWriter(video_out, w, h, meta["fps"], video_in)
    cap = cv2.VideoCapture(video_in)
    frames: "queue.Queue" = queue.Queue(maxsize=8)
    stop = threading.Event()
    reader = threading.Thread(target=_read_ahead, args=(cap, frames, stop), daemon=True)
    reader.start()
    prev = None  # (box, rect, ring, gray, filled) of the previous frame's patch
    try:
        i = 0
        while True:
            frame = frames.get()
            if isinstance(frame, BaseException):
                raise frame
            if frame is None:
                break
            if should_cancel and should_cancel():
                raise InterruptedError("Cancelled")
            box: Optional[Box] = None
            if first <= i <= last:
                if boxes is None:
                    box = box0
                elif i < len(boxes):
                    box = boxes[i]

            r = None
            if box is not None:
                r = roi_mask(frame.shape, place_points(lasso_points, box0, box), feather_px, mode, blur_ksize)
            if r is None:
                prev = None
            else:
                (x0, y0, x1, y1), mask = r
                src = frame[y0:y1, x0:x1]
                gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
                same = prev is not None and prev[0] == box and prev[1] == (x0, y0, x1, y1)
                ring = prev[2] if same else _ring(mask)
                filled = fill_region(src, mask, mode=mode, direction=direction, blur_ksize=blur_ksize)
                # Flicker suppression: blend with the previous patch, but only as much as the
                # surroundings are static. Over moving footage the blend would average frames
                # together and show up as a translucent box, so it fades to zero there.
                if same and temporal_smooth > 0:
                    motion = float(np.abs(gray[ring].astype(np.int16) - prev[3][ring].astype(np.int16)).mean())
                    factor = max(0.0, 1.0 - motion / 6.0)
                    if factor > 0.05:
                        a = mask.astype(np.float32)[..., None] / 255.0 * temporal_smooth * factor
                        filled = (filled.astype(np.float32) * (1 - a) + prev[4].astype(np.float32) * a).astype(np.uint8)
                frame[y0:y1, x0:x1] = filled
                prev = (box, (x0, y0, x1, y1), ring, gray, filled)
            writer.write(frame)
            i += 1
            if progress and i % 10 == 0:
                progress(min(i / n, 0.99))
    except BaseException:
        writer.abort()
        raise
    finally:
        stop.set()
        while reader.is_alive():  # unblock the reader if it is waiting on a full queue
            try:
                frames.get_nowait()
            except queue.Empty:
                reader.join(0.05)
        cap.release()
    writer.close()
    if progress:
        progress(1.0)
    return video_out
