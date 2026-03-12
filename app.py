#!/usr/bin/env python3
from __future__ import annotations

from contextlib import asynccontextmanager
import tempfile
import threading
import time
import uuid
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from qwen_pipeline import (
    DEFAULT_SYSTEM,
    DEFAULT_TASK,
    DEFAULT_VL_PROMPT,
    PipelineConfig,
    QwenPipeline,
)


BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
pipeline = QwenPipeline()
jobs: dict[str, dict] = {}
jobs_lock = threading.Lock()


@asynccontextmanager
async def lifespan(_: FastAPI):
    try:
        pipeline.load_models()
    except Exception:
        # Keep the API reachable so the page can show the load error via /api/status.
        pass
    yield


app = FastAPI(title="Local Qwen MLX Demo", version="0.1.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


def build_config(
    vl_prompt: str,
    task: str,
    system: str,
    vl_max_tokens: int,
    llm_max_tokens: int,
    temperature: float,
    top_p: float,
    seed: int,
    resize_max_edge: int,
    strip_thinking: bool,
) -> PipelineConfig:
    return PipelineConfig(
        vl_prompt=vl_prompt,
        task=task,
        system=system,
        vl_max_tokens=vl_max_tokens,
        llm_max_tokens=llm_max_tokens,
        temperature=temperature,
        top_p=top_p,
        seed=seed,
        resize_max_edge=resize_max_edge,
        strip_thinking=strip_thinking,
    )


def create_job() -> dict:
    job_id = uuid.uuid4().hex
    job = {
        "id": job_id,
        "state": "queued",
        "message": "任务已创建，等待执行。",
        "created_at": time.time(),
        "updated_at": time.time(),
        "partial": {},
        "result": None,
        "error": None,
    }
    with jobs_lock:
        jobs[job_id] = job
    return job


def update_job(job_id: str, **changes) -> None:
    with jobs_lock:
        job = jobs[job_id]
        partial = changes.pop("partial", None)
        if partial:
            job["partial"].update(partial)
        job.update(changes)
        job["updated_at"] = time.time()


def get_job(job_id: str) -> dict:
    with jobs_lock:
        job = jobs.get(job_id)
        if job is None:
            raise KeyError(job_id)
        return {
            **job,
            "partial": dict(job["partial"]),
        }


def run_job(job_id: str, upload_path: Path, temp_dir: str, config: PipelineConfig) -> None:
    def on_progress(event: dict) -> None:
        state = event.get("state", "running")
        message = event.get("message", "推理中。")
        partial = {
            key: value
            for key, value in event.items()
            if key not in {"state", "message"}
        }
        update_job(job_id, state=state, message=message, partial=partial)

    try:
        update_job(job_id, state="running", message="任务开始执行。")
        result = pipeline.infer_with_progress(upload_path, config, progress_callback=on_progress)
        update_job(
            job_id,
            state="completed",
            message="推理完成。",
            result=result,
            partial=result,
        )
    except Exception as exc:  # noqa: BLE001
        update_job(job_id, state="error", message="推理失败。", error=str(exc))
    finally:
        try:
            Path(temp_dir).joinpath(upload_path.name).unlink(missing_ok=True)
            Path(temp_dir).rmdir()
        except OSError:
            pass


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/status")
def status() -> dict:
    payload = pipeline.get_status()
    payload["defaults"] = {
            "vl_prompt": DEFAULT_VL_PROMPT,
            "task": DEFAULT_TASK,
            "system": DEFAULT_SYSTEM,
            "resize_max_edge": 768,
            "vl_max_tokens": 131072,
            "llm_max_tokens": 131072,
            "temperature": 0.2,
            "top_p": 0.9,
            "seed": 7,
        }
    return payload


@app.post("/api/infer")
async def infer(
    image: UploadFile = File(...),
    vl_prompt: str = Form(DEFAULT_VL_PROMPT),
    task: str = Form(DEFAULT_TASK),
    system: str = Form(DEFAULT_SYSTEM),
    vl_max_tokens: int = Form(131072),
    llm_max_tokens: int = Form(131072),
    temperature: float = Form(0.2),
    top_p: float = Form(0.9),
    seed: int = Form(7),
    resize_max_edge: int = Form(768),
    strip_thinking: bool = Form(True),
) -> dict:
    suffix = Path(image.filename or "upload.jpg").suffix or ".jpg"
    contents = await image.read()
    if not contents:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    config = build_config(
        vl_prompt,
        task,
        system,
        vl_max_tokens,
        llm_max_tokens,
        temperature,
        top_p,
        seed,
        resize_max_edge,
        strip_thinking,
    )

    with tempfile.TemporaryDirectory(prefix="qwen-upload-") as tmp:
        upload_path = Path(tmp) / f"input{suffix}"
        upload_path.write_bytes(contents)
        try:
            return pipeline.infer(upload_path, config)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/api/jobs")
async def create_infer_job(
    image: UploadFile = File(...),
    vl_prompt: str = Form(DEFAULT_VL_PROMPT),
    task: str = Form(DEFAULT_TASK),
    system: str = Form(DEFAULT_SYSTEM),
    vl_max_tokens: int = Form(131072),
    llm_max_tokens: int = Form(131072),
    temperature: float = Form(0.2),
    top_p: float = Form(0.9),
    seed: int = Form(7),
    resize_max_edge: int = Form(768),
    strip_thinking: bool = Form(True),
) -> dict:
    suffix = Path(image.filename or "upload.jpg").suffix or ".jpg"
    contents = await image.read()
    if not contents:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    config = build_config(
        vl_prompt,
        task,
        system,
        vl_max_tokens,
        llm_max_tokens,
        temperature,
        top_p,
        seed,
        resize_max_edge,
        strip_thinking,
    )

    temp_dir = tempfile.mkdtemp(prefix="qwen-job-")
    upload_path = Path(temp_dir) / f"input{suffix}"
    upload_path.write_bytes(contents)

    job = create_job()
    thread = threading.Thread(
        target=run_job,
        args=(job["id"], upload_path, temp_dir, config),
        daemon=True,
    )
    thread.start()
    return get_job(job["id"])


@app.get("/api/jobs/{job_id}")
def job_status(job_id: str) -> dict:
    try:
        return get_job(job_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Job not found.") from exc


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=False)
