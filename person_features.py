from __future__ import annotations

import json
import re
import shutil
import time
import uuid
from pathlib import Path
from typing import Any

from face_identity import aggregate_embeddings, compute_face_embedding, save_face_preview


DEFAULT_PERSON_PROMPT = (
    "上传多张同一人的清晰样本照，用于建立本地人物 ID 向量。"
)


class PersonFeatureStore:
    def __init__(self, base_dir: Path) -> None:
        self.base_dir = base_dir
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def list_profiles(self, include_embedding: bool = False) -> list[dict[str, Any]]:
        profiles: list[dict[str, Any]] = []
        for profile_dir in sorted(self.base_dir.iterdir()):
            if not profile_dir.is_dir():
                continue
            profile_path = profile_dir / "profile.json"
            if not profile_path.exists():
                continue
            profile = self._normalize_profile(json.loads(profile_path.read_text(encoding="utf-8")))
            profiles.append(profile if include_embedding else self._sanitize_profile(profile))
        return sorted(profiles, key=lambda item: item.get("updated_at", 0), reverse=True)

    def get_profile(self, profile_id: str, include_embedding: bool = False) -> dict[str, Any]:
        profile_path = self.base_dir / profile_id / "profile.json"
        if not profile_path.exists():
            raise FileNotFoundError(profile_id)
        profile = json.loads(profile_path.read_text(encoding="utf-8"))
        normalized = self._normalize_profile(profile)
        return normalized if include_embedding else self._sanitize_profile(normalized)

    def create_profile(self, name: str, notes: str = "") -> dict[str, Any]:
        profile_id = uuid.uuid4().hex
        profile_dir = self.base_dir / profile_id
        profile_dir.mkdir(parents=True, exist_ok=False)
        person_id = self._generate_person_id(name)
        profile = {
            "id": profile_id,
            "person_id": person_id,
            "name": name.strip(),
            "notes": notes.strip(),
            "feature_summary": "",
            "sample_count": 0,
            "samples": [],
            "identity_embedding": [],
            "has_identity_embedding": False,
            "identity_source": "face_embedding",
            "preview_available": False,
            "created_at": time.time(),
            "updated_at": time.time(),
            "last_extracted_at": None,
        }
        self._write_profile(profile)
        return self._sanitize_profile(profile)

    def delete_profile(self, profile_id: str) -> None:
        profile_dir = self.base_dir / profile_id
        if not profile_dir.exists():
            raise FileNotFoundError(profile_id)
        shutil.rmtree(profile_dir)

    def clear_profile_samples(self, profile_id: str) -> dict[str, Any]:
        profile = self.get_profile(profile_id, include_embedding=True)
        sample_dir = self.base_dir / profile_id / "samples"
        if sample_dir.exists():
            shutil.rmtree(sample_dir)
        self.preview_path(profile_id).unlink(missing_ok=True)
        profile["feature_summary"] = ""
        profile["sample_count"] = 0
        profile["samples"] = []
        profile["identity_embedding"] = []
        profile["has_identity_embedding"] = False
        profile["preview_available"] = False
        profile["last_extracted_at"] = None
        profile["updated_at"] = time.time()
        self._write_profile(profile)
        return self._sanitize_profile(profile)

    def update_profile_meta(self, profile_id: str, name: str, notes: str = "") -> dict[str, Any]:
        profile = self.get_profile(profile_id, include_embedding=True)
        profile["name"] = name.strip()
        profile["notes"] = notes.strip()
        profile["updated_at"] = time.time()
        self._write_profile(profile)
        return self._sanitize_profile(profile)

    def update_profile_identity(
        self,
        profile_id: str,
        sample_records: list[dict[str, Any]],
        identity_embedding: list[float],
        feature_summary: str,
        preview_source_path: str | None = None,
        preview_bbox: dict[str, int] | None = None,
    ) -> dict[str, Any]:
        profile = self.get_profile(profile_id, include_embedding=True)
        profile["feature_summary"] = feature_summary
        profile["sample_count"] = sum(1 for item in sample_records if item.get("face_detected"))
        profile["samples"] = sample_records
        profile["identity_embedding"] = identity_embedding
        profile["has_identity_embedding"] = bool(identity_embedding)
        if preview_source_path and preview_bbox:
            save_face_preview(preview_source_path, preview_bbox, self.preview_path(profile_id))
            profile["preview_available"] = True
        profile["last_extracted_at"] = time.time()
        profile["updated_at"] = time.time()
        self._write_profile(profile)
        return self._sanitize_profile(profile)

    def save_sample(self, profile_id: str, source_path: Path, original_name: str) -> Path:
        sample_dir = self.base_dir / profile_id / "samples"
        sample_dir.mkdir(parents=True, exist_ok=True)
        suffix = Path(original_name).suffix or source_path.suffix or ".jpg"
        filename = f"{uuid.uuid4().hex}{suffix}"
        target_path = sample_dir / filename
        shutil.copy2(source_path, target_path)
        return target_path

    def preview_path(self, profile_id: str) -> Path:
        return self.base_dir / profile_id / "preview.jpg"

    def _write_profile(self, profile: dict[str, Any]) -> None:
        profile = self._normalize_profile(profile)
        profile_dir = self.base_dir / profile["id"]
        profile_dir.mkdir(parents=True, exist_ok=True)
        profile_path = profile_dir / "profile.json"
        profile_path.write_text(
            json.dumps(profile, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def _generate_person_id(self, name: str) -> str:
        slug = re.sub(r"[^a-z0-9]+", "-", name.strip().lower())
        slug = slug.strip("-") or "person"
        return f"{slug}-{uuid.uuid4().hex[:8]}"

    def _normalize_profile(self, profile: dict[str, Any]) -> dict[str, Any]:
        normalized = dict(profile)
        normalized.setdefault("person_id", self._generate_person_id(normalized.get("name", "person")))
        normalized.setdefault("notes", "")
        normalized.setdefault("feature_summary", "")
        normalized.setdefault("sample_count", 0)
        normalized.setdefault("samples", [])
        normalized.setdefault("identity_embedding", [])
        normalized["has_identity_embedding"] = bool(normalized.get("identity_embedding"))
        normalized.setdefault("identity_source", "face_embedding")
        normalized["preview_available"] = bool(normalized.get("preview_available")) and self.preview_path(normalized["id"]).exists()
        return normalized

    def _sanitize_profile(self, profile: dict[str, Any]) -> dict[str, Any]:
        sanitized = dict(profile)
        sanitized.pop("identity_embedding", None)
        sanitized["has_identity_embedding"] = bool(profile.get("identity_embedding"))
        sanitized["preview_available"] = bool(profile.get("preview_available"))
        return sanitized


def summarize_identity_build(sample_count: int, failed_count: int) -> str:
    summary = f"已建立人物 ID 向量，成功样本 {sample_count} 张。"
    if failed_count:
        summary += f"\n未能提取人脸的样本 {failed_count} 张。"
    return summary


def extract_person_features(
    store: PersonFeatureStore,
    profile_id: str,
    images: list[tuple[Path, str]],
    prompt: str = DEFAULT_PERSON_PROMPT,
) -> dict[str, Any]:
    if not images:
        raise ValueError("No images were provided.")

    sample_records: list[dict[str, Any]] = []
    embeddings: list[list[float]] = []
    failed_count = 0
    best_preview: tuple[str, dict[str, int], int] | None = None
    for index, (image_path, original_name) in enumerate(images, start=1):
        stored_path = store.save_sample(profile_id, image_path, original_name)
        try:
            embedding, bbox = compute_face_embedding(stored_path)
            embeddings.append(embedding)
            area = int(bbox["width"]) * int(bbox["height"])
            if best_preview is None or area > best_preview[2]:
                best_preview = (str(stored_path), bbox, area)
            sample_records.append(
                {
                    "index": index,
                    "original_name": original_name,
                    "stored_path": str(stored_path),
                    "face_detected": True,
                    "bbox": bbox,
                    "embedding_dim": len(embedding),
                }
            )
        except ValueError as exc:
            failed_count += 1
            sample_records.append(
                {
                    "index": index,
                    "original_name": original_name,
                    "stored_path": str(stored_path),
                    "face_detected": False,
                    "error": str(exc),
                }
            )

    if not embeddings:
        raise ValueError("Uploaded samples did not contain a detectable face.")

    identity_embedding = aggregate_embeddings(embeddings)
    feature_summary = summarize_identity_build(len(embeddings), failed_count)
    profile = store.update_profile_identity(
        profile_id,
        sample_records=sample_records,
        identity_embedding=identity_embedding,
        feature_summary=feature_summary,
        preview_source_path=best_preview[0] if best_preview else None,
        preview_bbox=best_preview[1] if best_preview else None,
    )
    return {
        "profile": profile,
        "feature_summary": feature_summary,
        "samples": sample_records,
        "prompt": prompt,
    }
