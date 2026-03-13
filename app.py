#!/usr/bin/env python3
from __future__ import annotations

from contextlib import asynccontextmanager
import json
import re
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from pathlib import Path

from fastapi import Body, FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, ImageOps
from pydantic import BaseModel

from face_identity import identify_people
from person_features import (
    DEFAULT_PERSON_PROMPT,
    PersonFeatureStore,
    extract_person_features,
)
from qwen_pipeline import (
    DEFAULT_SYSTEM,
    DEFAULT_TASK,
    DEFAULT_VL_PROMPT,
    InferenceCancelled,
    PipelineConfig,
    QwenPipeline,
)


BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
PERSON_FEATURES_DIR = BASE_DIR / "outputs" / "person_features"
SUPPORTED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".heic", ".heif"}
MIN_LIBRARY_IMAGE_SIZE_BYTES = 100 * 1024
MIN_LIBRARY_IMAGE_LONG_EDGE = 360
pipeline = QwenPipeline()
person_store = PersonFeatureStore(PERSON_FEATURES_DIR)
jobs: dict[str, dict] = {}
jobs_lock = threading.Lock()
stream_cancellations: dict[str, threading.Event] = {}
stream_cancellations_lock = threading.Lock()


class PersonProfileCreate(BaseModel):
    name: str
    notes: str = ""


class PersonProfileUpdate(BaseModel):
    name: str
    notes: str = ""


class LibraryScanRequest(BaseModel):
    directories: list[str] = []
    recursive: bool = False
    directory: str | None = None


class StreamPathRequest(BaseModel):
    image_path: str
    vl_prompt: str = DEFAULT_VL_PROMPT
    task: str = DEFAULT_TASK
    system: str = DEFAULT_SYSTEM
    vl_max_tokens: int = 131072
    llm_max_tokens: int = 131072
    temperature: float = 0.2
    top_p: float = 0.9
    seed: int = 7
    resize_max_edge: int = 768
    strip_thinking: bool = True
    inference_id: str | None = None


class StreamCancelRequest(BaseModel):
    inference_id: str


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


def encode_sse(event: str, payload: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"


def list_identity_profiles() -> list[dict]:
    return person_store.list_profiles(include_embedding=True)


def describe_people_layout(identified_people: list[dict]) -> list[dict]:
    if not identified_people:
        return []
    centers_x = []
    centers_y = []
    for item in identified_people:
        bbox = item["bbox"]
        centers_x.append(bbox["x"] + bbox["width"] / 2)
        centers_y.append(bbox["y"] + bbox["height"] / 2)
    min_x, max_x = min(centers_x), max(centers_x)
    min_y, max_y = min(centers_y), max(centers_y)
    span_x = max(max_x - min_x, 1.0)
    span_y = max(max_y - min_y, 1.0)

    ordered = sorted(
        identified_people,
        key=lambda item: (
            item["bbox"]["x"] + item["bbox"]["width"] / 2,
            item["bbox"]["y"] + item["bbox"]["height"] / 2,
        ),
    )
    described: list[dict] = []
    total = len(ordered)
    for index, item in enumerate(ordered, start=1):
        bbox = item["bbox"]
        center_x = bbox["x"] + bbox["width"] / 2
        center_y = bbox["y"] + bbox["height"] / 2
        x_ratio = (center_x - min_x) / span_x
        y_ratio = (center_y - min_y) / span_y
        if total == 1:
            horizontal = "居中"
        elif x_ratio < 0.2:
            horizontal = "最左侧"
        elif x_ratio < 0.4:
            horizontal = "偏左"
        elif x_ratio < 0.6:
            horizontal = "中间"
        elif x_ratio < 0.8:
            horizontal = "偏右"
        else:
            horizontal = "最右侧"
        if total == 1 or span_y < 80:
            vertical = "同一排"
        elif y_ratio < 0.33:
            vertical = "上排"
        elif y_ratio < 0.66:
            vertical = "中排"
        else:
            vertical = "下排"
        described.append(
            {
                **item,
                "position_index": index,
                "position_label": f"第{index}人",
                "horizontal_position": horizontal,
                "vertical_position": vertical,
            }
        )
    return described


def inject_people_context(vl_prompt: str, identified_people: list[dict]) -> str:
    if not identified_people:
        return vl_prompt
    people_with_layout = describe_people_layout(identified_people)
    lines = []
    for item in people_with_layout:
        if item.get("person_id") == "unknown":
            identity_text = "name=未知人物，person_id=unknown，confidence=unknown"
        else:
            identity_text = (
                f"name={item['name']}，person_id={item['person_id']}，confidence={item['confidence']}"
            )
        lines.append(
            f"- {item['position_label']}：{item['horizontal_position']}，{item['vertical_position']}，"
            f"{identity_text}"
        )
    return (
        f"{vl_prompt.strip()}\n\n"
        "已通过本地人脸向量匹配识别出以下人物，并且已经按画面从左到右编号。"
        "请优先使用这些编号和相对位置来绑定人物身份，不要把已识别身份分配给别的人。"
        "对于未知人物，保持为未知人物，不要擅自套用已知姓名。"
        "如果要描述动作或关系，请先按“第N人”确定对象，再写名字；如果画面信息明显冲突，再说明不确定性。\n"
        + "\n".join(lines)
    )


def detect_people(upload_path: Path) -> list[dict]:
    profiles = list_identity_profiles()
    if not profiles:
        return []
    return identify_people(upload_path, profiles)


MEMORY_SCORE_WEIGHTS = {
    "memorability": 0.35,
    "emotion": 0.20,
    "story": 0.25,
    "anchors": 0.20,
}


def attach_computed_memory_score(payload: dict) -> dict:
    vl_output = payload.get("vl_output")
    if not isinstance(vl_output, str):
        return payload
    match = re.search(r"\{.*\}", vl_output.strip(), re.DOTALL)
    if not match:
        return payload
    try:
        parsed = json.loads(match.group(0))
    except json.JSONDecodeError:
        return payload
    dimension_scores = parsed.get("dimension_scores")
    if not isinstance(dimension_scores, dict):
        return payload
    weighted_total = 0.0
    for key, weight in MEMORY_SCORE_WEIGHTS.items():
        value = dimension_scores.get(key)
        if not isinstance(value, (int, float)):
            return payload
        weighted_total += float(value) * weight
    parsed["memory_score"] = round(weighted_total, 1)
    payload["vl_output"] = json.dumps(parsed, ensure_ascii=False, indent=2)
    payload["memory_score"] = parsed["memory_score"]
    return payload


def enrich_identity_result(payload: dict, identified_people: list[dict] | None = None) -> dict:
    payload["identified_people"] = describe_people_layout(identified_people or [])
    payload = attach_computed_memory_score(payload)
    if isinstance(payload.get("vl_output"), str):
        payload["vl_description"] = payload["vl_output"]
    return payload


def attach_vl_prompt(payload: dict, vl_prompt: str) -> dict:
    payload["effective_vl_prompt"] = vl_prompt
    return payload


def register_stream_cancellation(inference_id: str | None) -> tuple[str | None, threading.Event | None]:
    if not inference_id:
        return None, None
    event = threading.Event()
    with stream_cancellations_lock:
        stream_cancellations[inference_id] = event
    return inference_id, event


def cancel_stream_inference(inference_id: str | None) -> None:
    if not inference_id:
        return
    with stream_cancellations_lock:
        event = stream_cancellations.get(inference_id)
    if event is not None:
        event.set()


def cleanup_stream_cancellation(inference_id: str | None) -> None:
    if not inference_id:
        return
    with stream_cancellations_lock:
        stream_cancellations.pop(inference_id, None)


def save_upload_to_temp(contents: bytes, suffix: str, prefix: str) -> tuple[str, Path]:
    temp_dir = tempfile.mkdtemp(prefix=prefix)
    upload_path = Path(temp_dir) / f"input{suffix}"
    upload_path.write_bytes(contents)
    return temp_dir, upload_path


def prepare_inference(
    image_path: Path,
    *,
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
) -> tuple[list[dict], PipelineConfig]:
    identified_people = detect_people(image_path)
    config = build_config(
        inject_people_context(vl_prompt, identified_people),
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
    return identified_people, config


def build_streaming_response(
    upload_path: Path,
    temp_dir: str | None,
    config: PipelineConfig,
    identified_people: list[dict],
    inference_id: str | None = None,
) -> StreamingResponse:
    _, cancel_event = register_stream_cancellation(inference_id)

    def should_cancel() -> bool:
        return cancel_event.is_set() if cancel_event is not None else False

    def event_stream():
        try:
            yield ": stream-start\n\n"
            yield encode_sse(
                "identified_people",
                {
                    "identified_people": identified_people,
                },
            )
            yield encode_sse(
                "effective_vl_prompt",
                {
                    "effective_vl_prompt": config.vl_prompt,
                },
            )
            for item in pipeline.stream_infer(upload_path, config, should_cancel=should_cancel):
                if item["event"] == "complete":
                    item["data"] = attach_vl_prompt(
                        enrich_identity_result(item["data"], identified_people=identified_people),
                        config.vl_prompt,
                    )
                yield encode_sse(item["event"], item["data"])
        except InferenceCancelled:
            yield encode_sse(
                "cancelled",
                {
                    "message": "推理已中断。",
                    "inference_id": inference_id,
                },
            )
        except Exception as exc:  # noqa: BLE001
            yield encode_sse(
                "error",
                {
                    "message": str(exc),
                },
            )
        finally:
            cleanup_stream_cancellation(inference_id)
            if temp_dir is not None:
                try:
                    upload_path.unlink(missing_ok=True)
                    Path(temp_dir).rmdir()
                except OSError:
                    pass

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
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


def run_job(
    job_id: str,
    upload_path: Path,
    temp_dir: str,
    config: PipelineConfig,
    identified_people: list[dict] | None = None,
) -> None:
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
        result = enrich_identity_result(
            pipeline.infer_with_progress(upload_path, config, progress_callback=on_progress),
            identified_people=identified_people,
        )
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


def serialize_library_file(path: Path, base_dir: Path) -> dict:
    stat = path.stat()
    return {
        "name": path.name,
        "path": str(path),
        "relative_path": str(path.relative_to(base_dir)),
        "suffix": path.suffix.lower(),
        "size_bytes": stat.st_size,
        "modified_at": stat.st_mtime,
    }


def open_directory_picker() -> list[str]:
    if sys.platform != "darwin":
        raise HTTPException(status_code=501, detail="Native directory picker is currently only supported on macOS.")
    script = """
set chosenFolders to choose folder with prompt "选择一个或多个相册目录" with multiple selections allowed
set pathList to {}
repeat with oneFolder in chosenFolders
  copy POSIX path of oneFolder to end of pathList
end repeat
set AppleScript's text item delimiters to linefeed
return pathList as text
"""
    try:
        result = subprocess.run(
            ["osascript", "-e", script],
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError as exc:
        raise HTTPException(status_code=500, detail="Failed to open native directory picker.") from exc
    if result.returncode != 0:
        stderr = (result.stderr or "").strip().lower()
        if "user canceled" in stderr or "cancel" in stderr:
            return []
        raise HTTPException(status_code=500, detail="Native directory picker failed.")
    return [line.strip() for line in result.stdout.splitlines() if line.strip()]


def should_include_library_image(path: Path) -> bool:
    stat = path.stat()
    if stat.st_size < MIN_LIBRARY_IMAGE_SIZE_BYTES:
        return False
    try:
        with Image.open(path) as image:
            corrected = ImageOps.exif_transpose(image)
            long_edge = max(corrected.size)
    except Exception:
        return False
    return long_edge >= MIN_LIBRARY_IMAGE_LONG_EDGE


def iter_library_images(base_dir: Path, recursive: bool) -> list[dict]:
    iterator = base_dir.rglob("*") if recursive else base_dir.iterdir()
    files: list[dict] = []
    for item in iterator:
        if not item.is_file():
            continue
        if item.suffix.lower() not in SUPPORTED_IMAGE_EXTENSIONS:
            continue
        if not should_include_library_image(item):
            continue
        files.append(serialize_library_file(item, base_dir))
    return sorted(files, key=lambda item: item["modified_at"], reverse=True)


def count_descendant_supported_images(base_dir: Path) -> int:
    count = 0
    for item in base_dir.rglob("*"):
        if not item.is_file():
            continue
        if item.suffix.lower() not in SUPPORTED_IMAGE_EXTENSIONS:
            continue
        count += 1
    return count


def resolve_scan_directories(payload: LibraryScanRequest) -> list[Path]:
    raw_directories = [item for item in payload.directories if item]
    if payload.directory:
        raw_directories.append(payload.directory)
    if not raw_directories:
        raise HTTPException(status_code=400, detail="At least one directory is required.")
    resolved: list[Path] = []
    seen: set[str] = set()
    for raw in raw_directories:
        directory = Path(raw).expanduser().resolve()
        key = str(directory)
        if key in seen:
            continue
        if not directory.exists():
            raise HTTPException(status_code=404, detail=f"Directory not found: {directory}")
        if not directory.is_dir():
            raise HTTPException(status_code=400, detail=f"Path is not a directory: {directory}")
        seen.add(key)
        resolved.append(directory)
    return resolved


@app.get("/")
def home() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/debug")
def debug_page() -> FileResponse:
    return FileResponse(STATIC_DIR / "debug.html")


@app.get("/people")
def people_page() -> FileResponse:
    return FileResponse(STATIC_DIR / "people.html")


@app.post("/api/library/pick-directories")
def pick_library_directories() -> dict:
    directories = open_directory_picker()
    return {
        "directories": directories,
        "count": len(directories),
    }


@app.post("/api/library/scan")
def scan_library(payload: LibraryScanRequest = Body(...)) -> dict:
    directories = resolve_scan_directories(payload)
    files: list[dict] = []
    hints: list[str] = []
    for directory in directories:
        root_files = iter_library_images(directory, recursive=payload.recursive)
        if not payload.recursive and not root_files:
            nested_count = count_descendant_supported_images(directory)
            if nested_count > 0:
                hints.append(f"{directory.name} 的照片可能都在子目录里；开启递归扫描可读取约 {nested_count} 个候选文件。")
        for item in root_files:
            files.append(
                {
                    **item,
                    "directory": str(directory),
                    "directory_name": directory.name,
                    "display_path": f"{directory.name}/{item['relative_path']}",
                }
            )
    files.sort(key=lambda item: item["modified_at"], reverse=True)
    return {
        "directories": [str(item) for item in directories],
        "recursive": payload.recursive,
        "count": len(files),
        "files": files,
        "hints": hints,
    }


@app.get("/api/library/image")
def library_image(path: str = Query(...)) -> FileResponse:
    image_path = Path(path).expanduser().resolve()
    if not image_path.exists() or not image_path.is_file():
        raise HTTPException(status_code=404, detail="Image not found.")
    if image_path.suffix.lower() not in SUPPORTED_IMAGE_EXTENSIONS:
        raise HTTPException(status_code=400, detail="Unsupported image type.")
    return FileResponse(image_path)


@app.get("/api/status")
def status() -> dict:
    payload = pipeline.get_status()
    payload["people_profiles"] = {
            "count": len(person_store.list_profiles()),
        }
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


@app.get("/api/people")
def list_people() -> dict:
    return {
        "profiles": person_store.list_profiles(),
        "defaults": {
            "prompt": DEFAULT_PERSON_PROMPT,
        },
    }


@app.get("/api/people/{profile_id}/preview")
def person_preview(profile_id: str) -> FileResponse:
    preview_path = person_store.preview_path(profile_id)
    if not preview_path.exists():
        raise HTTPException(status_code=404, detail="Preview not found.")
    return FileResponse(preview_path, media_type="image/jpeg")


@app.post("/api/people")
def create_person(payload: PersonProfileCreate = Body(...)) -> dict:
    name = payload.name.strip()
    if not name:
        raise HTTPException(status_code=400, detail="Name is required.")
    return person_store.create_profile(name=name, notes=payload.notes)


def _update_person(profile_id: str, payload: PersonProfileUpdate) -> dict:
    name = payload.name.strip()
    if not name:
        raise HTTPException(status_code=400, detail="Name is required.")
    try:
        return person_store.update_profile_meta(profile_id=profile_id, name=name, notes=payload.notes)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail="Profile not found.") from exc


@app.patch("/api/people/{profile_id}")
def update_person_patch(profile_id: str, payload: PersonProfileUpdate = Body(...)) -> dict:
    return _update_person(profile_id, payload)


@app.put("/api/people/{profile_id}")
def update_person_put(profile_id: str, payload: PersonProfileUpdate = Body(...)) -> dict:
    return _update_person(profile_id, payload)


@app.post("/api/people/{profile_id}/update")
def update_person_post(profile_id: str, payload: PersonProfileUpdate = Body(...)) -> dict:
    return _update_person(profile_id, payload)


@app.delete("/api/people/{profile_id}")
def delete_person(profile_id: str) -> dict:
    try:
        person_store.delete_profile(profile_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail="Profile not found.") from exc
    return {"deleted": True, "id": profile_id}


@app.post("/api/people/{profile_id}/clear-samples")
def clear_person_samples(profile_id: str) -> dict:
    try:
        profile = person_store.clear_profile_samples(profile_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail="Profile not found.") from exc
    return {
        "cleared": True,
        "profile": profile,
    }


@app.post("/api/people/{profile_id}/extract")
async def extract_person(profile_id: str, images: list[UploadFile] = File(...), prompt: str = Form(DEFAULT_PERSON_PROMPT)) -> dict:
    if not images:
        raise HTTPException(status_code=400, detail="At least one image is required.")
    temp_dir = tempfile.mkdtemp(prefix="person-features-")
    saved_images: list[tuple[Path, str]] = []
    try:
        for image in images:
            contents = await image.read()
            if not contents:
                continue
            suffix = Path(image.filename or "sample.jpg").suffix or ".jpg"
            image_path = Path(temp_dir) / f"{uuid.uuid4().hex}{suffix}"
            image_path.write_bytes(contents)
            saved_images.append((image_path, image.filename or image_path.name))
        if not saved_images:
            raise HTTPException(status_code=400, detail="Uploaded files are empty.")
        try:
            return extract_person_features(store=person_store, profile_id=profile_id, images=saved_images, prompt=prompt)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail="Profile not found.") from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
    finally:
        for image_path, _ in saved_images:
            image_path.unlink(missing_ok=True)
        try:
            Path(temp_dir).rmdir()
        except OSError:
            pass


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

    with tempfile.TemporaryDirectory(prefix="qwen-upload-") as tmp:
        upload_path = Path(tmp) / f"input{suffix}"
        upload_path.write_bytes(contents)
        identified_people, config = prepare_inference(
            upload_path,
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

        try:
            return attach_vl_prompt(
                enrich_identity_result(pipeline.infer(upload_path, config), identified_people=identified_people),
                config.vl_prompt,
            )
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

    temp_dir, upload_path = save_upload_to_temp(contents, suffix, "qwen-job-")
    identified_people, config = prepare_inference(
        upload_path,
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

    job = create_job()
    thread = threading.Thread(
        target=run_job,
        args=(job["id"], upload_path, temp_dir, config, identified_people),
        daemon=True,
    )
    thread.start()
    return get_job(job["id"])


@app.post("/api/stream")
async def stream_infer(
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
    inference_id: str | None = Form(None),
) -> StreamingResponse:
    suffix = Path(image.filename or "upload.jpg").suffix or ".jpg"
    contents = await image.read()
    if not contents:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    temp_dir, upload_path = save_upload_to_temp(contents, suffix, "qwen-stream-")
    identified_people, config = prepare_inference(
        upload_path,
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
    return build_streaming_response(
        upload_path,
        temp_dir,
        config,
        identified_people,
        inference_id=inference_id,
    )


@app.post("/api/stream-path")
async def stream_infer_path(payload: StreamPathRequest = Body(...)) -> StreamingResponse:
    source_path = Path(payload.image_path).expanduser().resolve()
    if not source_path.exists() or not source_path.is_file():
        raise HTTPException(status_code=404, detail="Image not found.")
    if source_path.suffix.lower() not in SUPPORTED_IMAGE_EXTENSIONS:
        raise HTTPException(status_code=400, detail="Unsupported image type.")
    identified_people, config = prepare_inference(
        source_path,
        vl_prompt=payload.vl_prompt,
        task=payload.task,
        system=payload.system,
        vl_max_tokens=payload.vl_max_tokens,
        llm_max_tokens=payload.llm_max_tokens,
        temperature=payload.temperature,
        top_p=payload.top_p,
        seed=payload.seed,
        resize_max_edge=payload.resize_max_edge,
        strip_thinking=payload.strip_thinking,
    )
    return build_streaming_response(
        source_path,
        None,
        config,
        identified_people,
        inference_id=payload.inference_id,
    )


@app.post("/api/stream/cancel")
def cancel_stream(payload: StreamCancelRequest = Body(...)) -> dict:
    cancel_stream_inference(payload.inference_id)
    return {
        "cancelled": True,
        "inference_id": payload.inference_id,
    }


@app.get("/api/jobs/{job_id}")
def job_status(job_id: str) -> dict:
    try:
        return get_job(job_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Job not found.") from exc


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=False)
