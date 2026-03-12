#!/usr/bin/env python3
from __future__ import annotations

from contextlib import asynccontextmanager
import json
import tempfile
import threading
import time
import uuid
from pathlib import Path

from fastapi import Body, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
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
    PipelineConfig,
    QwenPipeline,
)


BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
PERSON_FEATURES_DIR = BASE_DIR / "outputs" / "person_features"
pipeline = QwenPipeline()
person_store = PersonFeatureStore(PERSON_FEATURES_DIR)
jobs: dict[str, dict] = {}
jobs_lock = threading.Lock()


class PersonProfileCreate(BaseModel):
    name: str
    notes: str = ""


class PersonProfileUpdate(BaseModel):
    name: str
    notes: str = ""


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


def enrich_identity_result(payload: dict, identified_people: list[dict] | None = None) -> dict:
    payload["identified_people"] = describe_people_layout(identified_people or [])
    if isinstance(payload.get("vl_output"), str):
        payload["vl_description"] = payload["vl_output"]
    return payload


def attach_vl_prompt(payload: dict, vl_prompt: str) -> dict:
    payload["effective_vl_prompt"] = vl_prompt
    return payload


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


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/people")
def people_page() -> FileResponse:
    return FileResponse(STATIC_DIR / "people.html")


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
        identified_people = detect_people(upload_path)
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

    temp_dir = tempfile.mkdtemp(prefix="qwen-job-")
    upload_path = Path(temp_dir) / f"input{suffix}"
    upload_path.write_bytes(contents)
    identified_people = detect_people(upload_path)
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
) -> StreamingResponse:
    suffix = Path(image.filename or "upload.jpg").suffix or ".jpg"
    contents = await image.read()
    if not contents:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    temp_dir = tempfile.mkdtemp(prefix="qwen-stream-")
    upload_path = Path(temp_dir) / f"input{suffix}"
    upload_path.write_bytes(contents)
    identified_people = detect_people(upload_path)
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
            for item in pipeline.stream_infer(upload_path, config):
                if item["event"] == "complete":
                    item["data"] = attach_vl_prompt(
                        enrich_identity_result(item["data"], identified_people=identified_people),
                        config.vl_prompt,
                    )
                yield encode_sse(item["event"], item["data"])
        except Exception as exc:  # noqa: BLE001
            yield encode_sse(
                "error",
                {
                    "message": str(exc),
                },
            )
        finally:
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


@app.get("/api/jobs/{job_id}")
def job_status(job_id: str) -> dict:
    try:
        return get_job(job_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Job not found.") from exc


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=False)
