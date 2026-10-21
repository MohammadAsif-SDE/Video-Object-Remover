"""FastAPI server: upload a video, draw a lasso on a frame, remove the object, download result."""

import base64
import logging
import shutil
import threading
import uuid
from collections import OrderedDict
from pathlib import Path
from typing import Dict, List, Literal, Optional, Tuple

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from remover import VideoIndex, fill_frame, read_frame, remove_object, track_object

ROOT = Path(__file__).parent
UPLOADS = ROOT / "uploads"
OUTPUTS = ROOT / "outputs"
STATIC = ROOT / "static"
UPLOADS.mkdir(exist_ok=True)
OUTPUTS.mkdir(exist_ok=True)

log = logging.getLogger("uvicorn.error")

app = FastAPI(title="Video Object Remover")
app.mount("/static", StaticFiles(directory=STATIC), name="static")
app.mount("/outputs", StaticFiles(directory=OUTPUTS), name="outputs")

jobs: Dict[str, dict] = {}
jobs_lock = threading.Lock()


def _frame_to_data_url(frame, max_w: int = 1280) -> str:
    h, w = frame.shape[:2]
    if w > max_w:
        frame = cv2.resize(frame, (max_w, int(h * max_w / w)), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 88])
    if not ok:
        raise RuntimeError("Could not encode frame")
    return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode()


# ---------- low-res video index (built in the background right after upload) ----------

class _IndexEntry:
    def __init__(self):
        self.index: Optional[VideoIndex] = None
        self.progress = 0.0
        self.error: Optional[str] = None
        self.ready = threading.Event()


_indexes: "OrderedDict[str, _IndexEntry]" = OrderedDict()
_indexes_lock = threading.Lock()
MAX_INDEXES = 2  # each holds a compressed low-res copy of a whole video (~100-200 MB)


def _start_index(video_id: str, path: Path) -> _IndexEntry:
    with _indexes_lock:
        entry = _indexes.get(video_id)
        if entry is not None:
            _indexes.move_to_end(video_id)
            return entry
        entry = _indexes[video_id] = _IndexEntry()
        while len(_indexes) > MAX_INDEXES:
            _indexes.popitem(last=False)

    def build():
        try:
            entry.index = VideoIndex(str(path), progress=lambda p: setattr(entry, "progress", p))
        except Exception as e:  # noqa: BLE001
            entry.error = str(e)
            log.exception("indexing %s failed", video_id)
        finally:
            entry.ready.set()

    threading.Thread(target=build, daemon=True).start()
    return entry


def _wait_index(video_id: str, job: dict) -> VideoIndex:
    """Block until the video's index is built, reporting it as the job's 'analyzing' stage."""
    entry = _start_index(video_id, _find_upload(video_id))
    job["stage"] = "analyzing"
    while not entry.ready.wait(0.25):
        if job["cancel"]:
            raise InterruptedError("Cancelled")
        job["progress"] = entry.progress
    if entry.error or entry.index is None:
        raise RuntimeError(f"Could not analyse the video: {entry.error}")
    return entry.index


# ---------- jobs ----------

def _new_job(kind: str) -> Tuple[str, dict]:
    job_id = uuid.uuid4().hex
    job = {"kind": kind, "status": "running", "stage": "starting", "progress": 0.0,
           "output": None, "error": None, "cancel": False, "result": None, "done": threading.Event()}
    with jobs_lock:
        jobs[job_id] = job
    return job_id, job


def _run_job(job: dict, fn) -> None:
    def run():
        try:
            fn()
            job["status"] = "done"
        except InterruptedError:
            job["status"] = "cancelled"
        except Exception as e:  # noqa: BLE001
            log.exception("job failed")
            job["status"] = "error"
            job["error"] = str(e)
        finally:
            job["done"].set()

    threading.Thread(target=run, daemon=True).start()


def _get_job(job_id: str) -> dict:
    job = jobs.get(job_id)
    if not job:
        raise HTTPException(404, "Unknown job")
    return job


def _public(job: dict) -> dict:
    return {k: job[k] for k in ("kind", "status", "stage", "progress", "output", "error")}


# ---------- routes ----------

@app.get("/", response_class=HTMLResponse)
def index():
    return (STATIC / "index.html").read_text(encoding="utf-8")


@app.post("/api/upload")
async def upload(file: UploadFile = File(...)):
    ext = Path(file.filename or "video.mp4").suffix.lower() or ".mp4"
    video_id = uuid.uuid4().hex
    dest = UPLOADS / f"{video_id}{ext}"
    with dest.open("wb") as f:
        shutil.copyfileobj(file.file, f)
    try:
        frame, meta = read_frame(str(dest), 0)
    except Exception as e:  # noqa: BLE001
        dest.unlink(missing_ok=True)
        raise HTTPException(400, f"Could not read video: {e}")
    _start_index(video_id, dest)  # analyse in the background while the user draws
    return {"video_id": video_id, "ext": ext, **meta, "frame": _frame_to_data_url(frame)}


@app.get("/api/frame/{video_id}")
def frame(video_id: str, index: int = 0):
    path = _find_upload(video_id)
    img, _ = read_frame(str(path), index)
    return {"index": index, "frame": _frame_to_data_url(img)}


class FillOptions(BaseModel):
    video_id: str
    points: List[Tuple[float, float]] = Field(..., min_length=3)
    start_frame: int = 0
    mode: Literal["stretch", "inpaint", "blur"] = "stretch"
    direction: Literal["horizontal", "vertical", "both"] = "horizontal"
    feather_px: int = Field(20, ge=0, le=100)
    blur_ksize: int = Field(41, ge=3, le=201)


@app.post("/api/preview")
def preview(req: FillOptions):
    """Apply the fill to one frame (points already placed for that frame), for exact feedback."""
    src = _find_upload(req.video_id)
    img, _ = read_frame(str(src), req.start_frame)
    out = fill_frame(img, np.asarray(req.points, np.float32), req.feather_px, req.mode, req.direction, req.blur_ksize)
    return {"frame": _frame_to_data_url(out)}


class TrackRequest(BaseModel):
    video_id: str
    points: List[Tuple[float, float]] = Field(..., min_length=3)
    start_frame: int = 0


def _track(req: TrackRequest, job: dict):
    idx = _wait_index(req.video_id, job)
    job["stage"], job["progress"] = "tracking", 0.0
    box0, boxes = track_object(idx, req.points, req.start_frame,
                               progress=lambda p: job.__setitem__("progress", p),
                               should_cancel=lambda: job["cancel"])
    return box0, boxes


@app.post("/api/track")
def track(req: TrackRequest):
    """Follow the selection through the video. Poll /api/status; the boxes are in /api/track/{id}."""
    _find_upload(req.video_id)
    job_id, job = _new_job("track")
    job["request"] = req

    def work():
        box0, boxes = _track(req, job)
        job["result"] = (box0, boxes)

    _run_job(job, work)
    return {"job_id": job_id}


@app.get("/api/track/{job_id}")
def track_result(job_id: str):
    job = _get_job(job_id)
    if job["kind"] != "track":
        raise HTTPException(400, "Not a tracking job")
    out = _public(job)
    if job["status"] == "done":
        box0, boxes = job["result"]
        out["box0"] = [round(v, 1) for v in box0]
        out["boxes"] = [None if b is None else [round(v, 1) for v in b] for b in boxes]
    return out


class ProcessRequest(FillOptions):
    track: bool = True
    track_id: Optional[str] = None  # reuse a finished (or running) /api/track job for this selection
    frame_range: Optional[Tuple[int, int]] = None  # first, last frame (inclusive) to apply to


@app.post("/api/process")
def process(req: ProcessRequest):
    src = _find_upload(req.video_id)
    job_id, job = _new_job("process")
    xs = [p[0] for p in req.points]
    ys = [p[1] for p in req.points]
    log.info(
        "job %s: %d pts bbox=(%.0f,%.0f)-(%.0f,%.0f) start_frame=%d track=%s range=%s mode=%s dir=%s feather=%d blur=%d",
        job_id, len(req.points), min(xs), min(ys), max(xs), max(ys), req.start_frame, req.track,
        req.frame_range, req.mode, req.direction, req.feather_px, req.blur_ksize,
    )
    out_path = OUTPUTS / f"{job_id}.mp4"

    def work():
        boxes = None
        if req.track:
            tj = jobs.get(req.track_id or "")
            treq = tj.get("request") if tj else None
            if treq is not None and treq.video_id == req.video_id and treq.points == req.points \
                    and treq.start_frame == req.start_frame:
                job["stage"] = "tracking"
                while not tj["done"].wait(0.25):
                    if job["cancel"]:
                        raise InterruptedError("Cancelled")
                    job["progress"] = tj["progress"]
                if tj["status"] != "done":
                    raise RuntimeError(tj["error"] or "Tracking did not finish")
                boxes = tj["result"][1]
            else:
                boxes = _track(TrackRequest(video_id=req.video_id, points=req.points,
                                            start_frame=req.start_frame), job)[1]
        job["stage"], job["progress"] = "rendering", 0.0
        remove_object(
            str(src), str(out_path), req.points,
            boxes=boxes, frame_range=req.frame_range,
            mode=req.mode, direction=req.direction,
            feather_px=req.feather_px, blur_ksize=req.blur_ksize,
            progress=lambda p: job.__setitem__("progress", p),
            should_cancel=lambda: job["cancel"],
        )
        job["output"] = f"/outputs/{out_path.name}"

    _run_job(job, work)
    return {"job_id": job_id}


@app.get("/api/status/{job_id}")
def status(job_id: str):
    return _public(_get_job(job_id))


@app.post("/api/cancel/{job_id}")
def cancel(job_id: str):
    _get_job(job_id)["cancel"] = True
    return {"ok": True}


@app.get("/api/download/{job_id}")
def download(job_id: str):
    if not job_id.isalnum():
        raise HTTPException(400, "Bad id")
    path = OUTPUTS / f"{job_id}.mp4"
    if not path.exists():
        raise HTTPException(404, "Not ready")
    return FileResponse(path, media_type="video/mp4", filename="object_removed.mp4")


def _find_upload(video_id: str) -> Path:
    if not video_id.isalnum():
        raise HTTPException(400, "Bad id")
    matches = list(UPLOADS.glob(f"{video_id}.*"))
    if not matches:
        raise HTTPException(404, "Unknown video")
    return matches[0]


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=False)
