from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageOps

try:
    from pillow_heif import register_heif_opener
except ImportError:  # pragma: no cover - optional dependency
    register_heif_opener = None

try:
    from insightface.app import FaceAnalysis
except ImportError:  # pragma: no cover - dependency installed locally
    FaceAnalysis = None

if register_heif_opener is not None:
    register_heif_opener()


_FACE_ANALYZER: FaceAnalysis | None = None
_FACE_ANALYZER_ERROR: str | None = None
DEFAULT_MODEL_NAME = "buffalo_l"


def _load_image(path: str | Path) -> np.ndarray:
    try:
        with Image.open(path) as image:
            corrected = ImageOps.exif_transpose(image).convert("RGB")
            array = np.array(corrected)
    except Exception as exc:  # noqa: BLE001
        raise ValueError(f"Unable to read image: {path}") from exc
    return array[:, :, ::-1].copy()


def load_pil_image(path: str | Path) -> Image.Image:
    try:
        with Image.open(path) as image:
            corrected = ImageOps.exif_transpose(image).convert("RGB")
            return corrected.copy()
    except Exception as exc:  # noqa: BLE001
        raise ValueError(f"Unable to read image: {path}") from exc


def get_face_analyzer() -> FaceAnalysis:
    global _FACE_ANALYZER, _FACE_ANALYZER_ERROR
    if _FACE_ANALYZER is not None:
        return _FACE_ANALYZER
    if _FACE_ANALYZER_ERROR is not None:
        raise RuntimeError(_FACE_ANALYZER_ERROR)
    if FaceAnalysis is None:
        _FACE_ANALYZER_ERROR = (
            "InsightFace is not installed. Run `.venv/bin/pip install insightface onnxruntime`."
        )
        raise RuntimeError(_FACE_ANALYZER_ERROR)
    try:
        analyzer = FaceAnalysis(name=DEFAULT_MODEL_NAME, providers=["CPUExecutionProvider"])
        analyzer.prepare(ctx_id=0, det_size=(640, 640))
    except Exception as exc:  # noqa: BLE001
        _FACE_ANALYZER_ERROR = f"Failed to initialize InsightFace ({DEFAULT_MODEL_NAME}): {exc}"
        raise RuntimeError(_FACE_ANALYZER_ERROR) from exc
    _FACE_ANALYZER = analyzer
    return analyzer


def _face_to_bbox(face: Any) -> dict[str, int]:
    bbox = [int(round(value)) for value in face.bbox.tolist()]
    x1, y1, x2, y2 = bbox
    return {
        "x": x1,
        "y": y1,
        "width": max(1, x2 - x1),
        "height": max(1, y2 - y1),
    }


def save_face_preview(
    source_path: str | Path,
    bbox: dict[str, int],
    target_path: str | Path,
    size: int = 160,
    padding_ratio: float = 0.35,
) -> None:
    image = load_pil_image(source_path)
    width, height = image.size
    x = int(bbox["x"])
    y = int(bbox["y"])
    w = int(bbox["width"])
    h = int(bbox["height"])
    padding_x = int(w * padding_ratio)
    padding_y = int(h * padding_ratio)
    left = max(0, x - padding_x)
    top = max(0, y - padding_y)
    right = min(width, x + w + padding_x)
    bottom = min(height, y + h + padding_y)
    crop = image.crop((left, top, right, bottom))
    crop_width, crop_height = crop.size
    square_side = max(crop_width, crop_height)
    square = Image.new("RGB", (square_side, square_side), (245, 240, 232))
    offset_x = (square_side - crop_width) // 2
    offset_y = (square_side - crop_height) // 2
    square.paste(crop, (offset_x, offset_y))
    crop = square.resize((size, size), Image.Resampling.LANCZOS)
    Path(target_path).parent.mkdir(parents=True, exist_ok=True)
    crop.save(target_path, format="JPEG", quality=90)


def _analyze_faces(path: str | Path) -> list[Any]:
    analyzer = get_face_analyzer()
    image = _load_image(path)
    try:
        faces = analyzer.get(image)
    except Exception as exc:  # noqa: BLE001
        raise ValueError(f"InsightFace failed on image: {path}") from exc
    return sorted(faces, key=lambda face: (_face_to_bbox(face)["x"], _face_to_bbox(face)["y"]))


def detect_faces(path: str | Path) -> list[dict[str, int]]:
    return [_face_to_bbox(face) for face in _analyze_faces(path)]


def _embedding_from_face(face: Any) -> list[float]:
    vector = getattr(face, "normed_embedding", None)
    if vector is None:
        vector = getattr(face, "embedding", None)
    if vector is None:
        raise ValueError("InsightFace did not return an embedding.")
    array = np.array(vector, dtype=np.float32)
    norm = float(np.linalg.norm(array))
    if norm == 0:
        raise ValueError("InsightFace returned a zero-norm embedding.")
    return (array / norm).astype(np.float32).tolist()


def compute_face_embedding(path: str | Path, bbox: dict[str, int] | None = None) -> tuple[list[float], dict[str, int]]:
    faces = _analyze_faces(path)
    if not faces:
        raise ValueError("No face detected in image.")
    if bbox is None:
        target_face = max(faces, key=lambda face: (_face_to_bbox(face)["width"] * _face_to_bbox(face)["height"]))
    else:
        def distance(face: Any) -> float:
            candidate = _face_to_bbox(face)
            return abs(candidate["x"] - bbox["x"]) + abs(candidate["y"] - bbox["y"])

        target_face = min(faces, key=distance)
    target_bbox = _face_to_bbox(target_face)
    return _embedding_from_face(target_face), target_bbox


def cosine_similarity(left: list[float], right: list[float]) -> float:
    a = np.array(left, dtype=np.float32)
    b = np.array(right, dtype=np.float32)
    left_norm = float(np.linalg.norm(a))
    right_norm = float(np.linalg.norm(b))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return float(np.dot(a, b) / (left_norm * right_norm))


def aggregate_embeddings(embeddings: list[list[float]]) -> list[float]:
    if not embeddings:
        raise ValueError("No embeddings to aggregate.")
    matrix = np.array(embeddings, dtype=np.float32)
    prototype = matrix.mean(axis=0)
    norm = float(np.linalg.norm(prototype))
    if norm == 0:
        raise ValueError("Aggregated embedding has zero norm.")
    return (prototype / norm).astype(np.float32).tolist()


def similarity_to_confidence(score: float) -> str:
    if score >= 0.55:
        return "high"
    if score >= 0.40:
        return "medium"
    if score >= 0.28:
        return "low"
    return "unknown"


def similarity_to_percent(score: float) -> float | None:
    if score < 0:
        return None
    clamped = max(0.0, min(1.0, score))
    return round(clamped * 100, 1)


def identify_people(path: str | Path, profiles: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not profiles:
        return []

    faces = _analyze_faces(path)
    if not faces:
        return []

    known_profiles = [
        profile
        for profile in profiles
        if isinstance(profile.get("identity_embedding"), list) and profile["identity_embedding"]
    ]
    if not known_profiles:
        return []

    identified: list[dict[str, Any]] = []
    for index, face in enumerate(faces, start=1):
        bbox = _face_to_bbox(face)
        embedding = _embedding_from_face(face)

        best_profile = None
        best_score = -1.0
        for profile in known_profiles:
            score = cosine_similarity(embedding, profile["identity_embedding"])
            if score > best_score:
                best_score = score
                best_profile = profile

        confidence = similarity_to_confidence(best_score)
        if best_profile is None or confidence == "unknown":
            identified.append(
                {
                    "face_index": index,
                    "person_id": "unknown",
                    "name": "未知人物",
                    "confidence": "unknown",
                    "confidence_score": round(best_score, 4) if best_score >= 0 else None,
                    "confidence_percent": similarity_to_percent(best_score),
                    "similarity": round(best_score, 4) if best_score >= 0 else None,
                    "bbox": bbox,
                }
            )
            continue

        identified.append(
            {
                "face_index": index,
                "person_id": best_profile["person_id"],
                "name": best_profile["name"],
                "confidence": confidence,
                "confidence_score": round(best_score, 4),
                "confidence_percent": similarity_to_percent(best_score),
                "similarity": round(best_score, 4),
                "bbox": bbox,
            }
        )

    return identified
