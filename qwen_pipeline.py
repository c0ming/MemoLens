from __future__ import annotations

import json
import logging
import mimetypes
import os
import plistlib
import subprocess
import struct
import sys
import tempfile
import threading
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import mlx.core as mx
from mlx_lm import generate as lm_generate
from mlx_lm import load as lm_load
from mlx_lm.sample_utils import make_sampler
from mlx_lm.generate import stream_generate as lm_stream_generate
from mlx_vlm import generate as vlm_generate
from mlx_vlm import load as vlm_load
from mlx_vlm.generate import stream_generate as vlm_stream_generate
from PIL import ExifTags, Image, ImageOps, UnidentifiedImageError
from transformers.tokenization_utils_tokenizers import TokenizersBackend

try:
    from pillow_heif import register_heif_opener
except ImportError:  # pragma: no cover - optional dependency
    register_heif_opener = None
else:
    register_heif_opener()


DEFAULT_VL_MODEL = os.path.expanduser(
    "~/.lmstudio/models/lmstudio-community/Qwen3-VL-8B-Instruct-MLX-8bit"
)
DEFAULT_LLM_MODEL = os.path.expanduser(
    "~/.lmstudio/models/nightmedia/"
    "Qwen3.5-27B-Claude-4.6-Opus-Reasoning-Distilled-qx64-hi-mlx"
)
DEFAULT_VL_PROMPT = """请分析这张图片，并严格输出 JSON，不要输出额外说明。

你需要完成 4 件事：

1. 用一句简短中文说明这是什么图片。
不要套用固定类别，如果无法准确概括，就尽量如实描述。

字段：
"image_type": "一句简短中文"

2. 评估这张图片的可回忆度。
“可回忆度”指这张图片在未来是否容易唤起具体、鲜明、可讲述的个人回忆。
它不是摄影质量分，也不是美观分。

请从以下 4 个维度评分，每项 0-10 分：
- memorability：整体是否容易让人记住
- emotion：是否有明显情绪或氛围
- story：是否像一个可讲述的时刻
- anchors：是否有具体细节可作为回忆锚点，如人物、动作、地点、物品、文字

再用一句话解释原因。

字段：
"dimension_scores": {
  "memorability": 0,
  "emotion": 0,
  "story": 0,
  "anchors": 0
},
"memory_reason": "一句话说明为什么这个分数高或低"

不要输出 memory_score，总分会由系统在你返回 dimension_scores 后自动计算，并补回最终结果。

3. 给图片打 tag。
请输出 3 到 8 个简短中文 tag。
tag 要尽量具体，优先基于画面中真实可见的内容，不要空泛，不要为了好看而编造。

字段：
"tags": ["", "", ""]

4. 给图片写一句题注。
要求：
- 只写一句
- 自然，不生硬
- 可以有一点诗意，但不要过度
- 不要像说明文
- 不要空泛，不要鸡汤
- 不要编造图片中看不出来的背景故事
- 不要用“这张照片里……”“画面中……”这类开头

字段：
"caption_line": "一句题注"

请严格输出 JSON，格式如下：
{
  "image_type": "",
  "dimension_scores": {
    "memorability": 0,
    "emotion": 0,
    "story": 0,
    "anchors": 0
  },
  "memory_reason": "",
  "tags": [],
  "caption_line": ""
}"""
DEFAULT_TASK = "请用一句简短的话描述整张图片的内容而不丢失重要细节，如果有识别到的人物，直接用人名替换来描述。"
DEFAULT_SYSTEM = "你是一个严谨的中文助手。请基于提供的视觉解析结果完成任务，不要编造图片中不存在的细节。"


class InferenceCancelled(Exception):
    pass


@dataclass(slots=True)
class PipelineConfig:
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


def require_existing_path(path: str, label: str) -> Path:
    resolved = Path(path).expanduser().resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"{label} not found: {resolved}")
    return resolved


def resolve_optional_path(path: str) -> Path | None:
    resolved = Path(path).expanduser().resolve()
    if not resolved.exists():
        return None
    return resolved


def maybe_write_json(path: str | None, payload: dict[str, Any]) -> None:
    if not path:
        return
    output_path = Path(path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def build_vl_prompt(processor: Any, image_path: str, prompt: str) -> str:
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_path},
                {"type": "text", "text": prompt},
            ],
        }
    ]
    return processor.apply_chat_template(messages, add_generation_prompt=True)


def build_llm_prompt(tokenizer: Any, system_prompt: str, task: str, vl_output: str) -> str:
    messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": (
                f"图片解析结果如下：\n{vl_output.strip()}\n\n"
                f"请完成这个任务：{task.strip()}"
            ),
        },
    ]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )


def split_thinking_output(text: str) -> dict[str, str]:
    cleaned = text.strip()
    reasoning = ""
    answer = cleaned
    if "<think>" in cleaned and "</think>" in cleaned:
        head, tail = cleaned.split("</think>", 1)
        reasoning = head.split("<think>", 1)[-1].strip()
        answer = tail.strip()
    return {
        "raw": cleaned,
        "reasoning": reasoning,
        "answer": answer,
    }


def split_thinking_progress(text: str) -> dict[str, Any]:
    cleaned = text.strip()
    if "<think>" not in cleaned:
        return {
            "raw": cleaned,
            "reasoning": "",
            "answer": cleaned,
            "in_thinking": False,
        }
    before_think, after_think = cleaned.split("<think>", 1)
    if "</think>" in after_think:
        reasoning, answer = after_think.split("</think>", 1)
        return {
            "raw": cleaned,
            "reasoning": reasoning.strip(),
            "answer": answer.strip(),
            "in_thinking": False,
        }
    return {
        "raw": cleaned,
        "reasoning": after_think.strip(),
        "answer": before_think.strip(),
        "in_thinking": True,
    }


def _to_json_safe(value: Any) -> Any:
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, bytes):
        return f"<{len(value)} bytes>"
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _to_json_safe(val) for key, val in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_to_json_safe(item) for item in value]
    if hasattr(value, "numerator") and hasattr(value, "denominator"):
        try:
            return float(value)
        except Exception:  # noqa: BLE001
            return str(value)
    return str(value)


def _decode_gps_info(gps_info: Any) -> dict[str, Any]:
    if not isinstance(gps_info, dict):
        return {}
    return {
        ExifTags.GPSTAGS.get(key, str(key)): _to_json_safe(value)
        for key, value in gps_info.items()
    }


def _extract_embedded_gps_ifd(exif: Any) -> dict[str, Any]:
    get_ifd = getattr(exif, "get_ifd", None)
    if get_ifd is None:
        return {}
    try:
        gps_ifd = get_ifd(ExifTags.IFD.GPSInfo)
    except Exception:  # noqa: BLE001
        return {}
    return _decode_gps_info(gps_ifd)


def _coerce_rational(value: Any) -> float | None:
    if isinstance(value, (int, float)):
        return float(value)
    if hasattr(value, "numerator") and hasattr(value, "denominator"):
        try:
            return float(value)
        except Exception:  # noqa: BLE001
            return None
    return None


def _gps_coordinate_to_decimal(values: Any, ref: Any) -> float | None:
    if not isinstance(values, (list, tuple)) or len(values) != 3:
        return None
    degrees = _coerce_rational(values[0])
    minutes = _coerce_rational(values[1])
    seconds = _coerce_rational(values[2])
    if degrees is None or minutes is None or seconds is None:
        return None
    decimal = degrees + minutes / 60.0 + seconds / 3600.0
    ref_text = str(ref).upper()
    if ref_text in {"S", "W"}:
        decimal *= -1
    return decimal


def _extract_gps_summary(exif_data: dict[str, Any]) -> dict[str, Any]:
    gps = exif_data.get("GPSInfo")
    if not isinstance(gps, dict):
        return {}
    summary: dict[str, Any] = {}
    latitude = _gps_coordinate_to_decimal(
        gps.get("GPSLatitude"),
        gps.get("GPSLatitudeRef"),
    )
    longitude = _gps_coordinate_to_decimal(
        gps.get("GPSLongitude"),
        gps.get("GPSLongitudeRef"),
    )
    altitude = _coerce_rational(gps.get("GPSAltitude"))
    if latitude is not None:
        summary["latitude"] = latitude
    if longitude is not None:
        summary["longitude"] = longitude
    if altitude is not None:
        summary["altitude_m"] = altitude
    return summary


def _summarize_exif_field(name: str, value: Any) -> Any:
    if name in {"ExifOffset", "InteropOffset", "JPEGInterchangeFormat", "JPEGInterchangeFormatLength"}:
        return {
            "raw": _to_json_safe(value),
            "note": "Internal EXIF offset/thumbnail pointer, not human-readable metadata.",
        }
    if name in {"MakerNote", "PrintImageMatching"}:
        return {
            "raw": _to_json_safe(value),
            "note": "Manufacturer/private metadata block; usually not human-readable.",
        }
    if name in {"UserComment", "XPComment", "XPKeywords", "XPSubject", "XPTitle", "XPAuthor"}:
        return {
            "raw": _to_json_safe(value),
            "note": "Comment-like metadata; decoding may depend on vendor-specific encoding.",
        }
    return _to_json_safe(value)


def _build_exif_data(exif: Any) -> dict[str, Any]:
    exif_data = {
        ExifTags.TAGS.get(tag, str(tag)): _summarize_exif_field(
            ExifTags.TAGS.get(tag, str(tag)),
            value,
        )
        for tag, value in exif.items()
    }
    embedded_gps = _extract_embedded_gps_ifd(exif)
    if embedded_gps:
        exif_data["GPSInfo"] = embedded_gps
        return exif_data
    raw_gps = exif_data.get("GPSInfo")
    gps_data = _decode_gps_info(raw_gps)
    if gps_data:
        exif_data["GPSInfo"] = gps_data
    elif raw_gps not in (None, {}):
        exif_data["GPSInfo"] = {
            "raw": _to_json_safe(raw_gps),
            "note": "GPS metadata appears to exist, but Pillow did not decode it into coordinate fields.",
        }
    return exif_data


def extract_macos_metadata(input_path: Path) -> dict[str, Any]:
    if sys.platform != "darwin":
        return {}
    try:
        result = subprocess.run(
            ["mdls", "-plist", "-", str(input_path)],
            capture_output=True,
            check=False,
        )
    except OSError:
        return {}
    if result.returncode != 0 or not result.stdout:
        return {
            "mdls_error": (result.stderr or b"").decode("utf-8", errors="replace").strip() or None,
        }
    try:
        payload = plistlib.loads(result.stdout)
    except Exception:  # noqa: BLE001
        return {
            "mdls_error": "Failed to parse mdls plist output.",
        }
    return _to_json_safe(payload)


def extract_assetsd_metadata(input_path: Path) -> dict[str, Any]:
    if sys.platform != "darwin":
        return {}
    metadata: dict[str, Any] = {}
    def read_attr(name: str) -> bytes | None:
        try:
            result = subprocess.run(
                ["xattr", "-px", name, str(input_path)],
                capture_output=True,
                check=False,
                text=True,
            )
        except OSError:
            return None
        if result.returncode != 0:
            return None
        hex_output = "".join(result.stdout.split())
        if not hex_output:
            return b""
        try:
            return bytes.fromhex(hex_output)
        except ValueError:
            return None

    location_raw = read_attr("com.apple.assetsd.customLocation")
    if location_raw and len(location_raw) >= 64:
        try:
            values = struct.unpack("<8d", location_raw[:64])
            metadata["custom_location"] = {
                "latitude": values[0],
                "longitude": values[1],
                "altitude_m": values[2],
                "horizontal_accuracy_m": values[3],
                "speed_mps": values[4],
                "course_deg": values[5],
                "vertical_accuracy_m": values[6],
                "timestamp_seconds_since_2001": values[7],
            }
        except struct.error:
            metadata["custom_location_raw"] = _to_json_safe(location_raw)

    creation_raw = read_attr("com.apple.assetsd.customCreationDate")
    if creation_raw:
        try:
            parsed = plistlib.loads(creation_raw)
            metadata["custom_creation_date"] = _to_json_safe(parsed)
        except Exception:  # noqa: BLE001
            metadata["custom_creation_date_raw"] = _to_json_safe(creation_raw)

    simple_attrs = {
        "com.apple.assetsd.originalFilename": "original_filename",
        "com.apple.assetsd.importedByDisplayName": "imported_by_display_name",
        "com.apple.assetsd.creatorBundleID": "creator_bundle_id",
        "com.apple.assetsd.cloudAsset.UUID": "cloud_asset_uuid",
        "com.apple.assetsd.mediaGroupUUID": "media_group_uuid",
        "com.apple.assetsd.timeZoneName": "timezone_name",
    }
    for attr_name, output_name in simple_attrs.items():
        raw = read_attr(attr_name)
        if raw is None:
            continue
        metadata[output_name] = raw.decode("utf-8", errors="replace").strip("\x00")

    return metadata


def build_macos_summary(metadata: dict[str, Any]) -> dict[str, Any]:
    if not metadata:
        return {}
    key_map = {
        "kind": "kMDItemKind",
        "display_name": "kMDItemDisplayName",
        "content_type": "kMDItemContentType",
        "content_type_tree": "kMDItemContentTypeTree",
        "file_size_bytes": "kMDItemFSSize",
        "created_at": "kMDItemContentCreationDate",
        "modified_at": "kMDItemContentModificationDate",
        "pixel_width": "kMDItemPixelWidth",
        "pixel_height": "kMDItemPixelHeight",
        "device_model": "kMDItemAcquisitionModel",
        "camera_maker": "kMDItemCreator",
        "color_space": "kMDItemColorSpace",
        "f_number": "kMDItemFNumber",
        "focal_length_mm": "kMDItemFocalLength",
        "iso_speed": "kMDItemISOSpeed",
        "exposure_time_seconds": "kMDItemExposureTimeSeconds",
        "exposure_program": "kMDItemExposureProgram",
        "latitude": "kMDItemLatitude",
        "longitude": "kMDItemLongitude",
        "white_balance": "kMDItemWhiteBalance",
        "flash_on_off": "kMDItemFlashOnOff",
    }
    return {
        label: metadata.get(source_key)
        for label, source_key in key_map.items()
        if metadata.get(source_key) is not None
    }


def merge_location_summary(macos_summary: dict[str, Any], assetsd_metadata: dict[str, Any]) -> dict[str, Any]:
    summary = dict(macos_summary)
    custom_location = assetsd_metadata.get("custom_location") or {}
    if "latitude" not in summary and custom_location.get("latitude") is not None:
        summary["latitude"] = custom_location["latitude"]
    if "longitude" not in summary and custom_location.get("longitude") is not None:
        summary["longitude"] = custom_location["longitude"]
    if "altitude_m" not in summary and custom_location.get("altitude_m") is not None:
        summary["altitude_m"] = custom_location["altitude_m"]
    if "created_at" not in summary and assetsd_metadata.get("custom_creation_date") is not None:
        summary["created_at"] = assetsd_metadata["custom_creation_date"]
    if "timezone_name" not in summary and assetsd_metadata.get("timezone_name") is not None:
        summary["timezone_name"] = assetsd_metadata["timezone_name"]
    return summary


def extract_photo_metadata(input_path: Path, image: Image.Image) -> dict[str, Any]:
    stat = input_path.stat()
    exif = image.getexif()
    exif_data = _build_exif_data(exif)
    exif_gps_summary = _extract_gps_summary(exif_data)
    macos_metadata = extract_macos_metadata(input_path)
    assetsd_metadata = extract_assetsd_metadata(input_path)
    macos_summary = merge_location_summary(
        build_macos_summary(macos_metadata),
        assetsd_metadata,
    )
    for key, value in exif_gps_summary.items():
        macos_summary.setdefault(key, value)

    return {
        "file": {
            "name": input_path.name,
            "suffix": input_path.suffix,
            "extension": input_path.suffix.lower(),
            "absolute_path": str(input_path.resolve()),
            "mime_type": mimetypes.guess_type(input_path.name)[0],
            "size_bytes": stat.st_size,
            "created_at": datetime.fromtimestamp(stat.st_ctime).isoformat(),
            "modified_at": datetime.fromtimestamp(stat.st_mtime).isoformat(),
            "heif_decoder_available": register_heif_opener is not None,
        },
        "image": {
            "format": image.format,
            "mode": image.mode,
            "bands": list(image.getbands()),
            "width": image.width,
            "height": image.height,
            "is_animated": bool(getattr(image, "is_animated", False)),
            "n_frames": getattr(image, "n_frames", 1),
            "has_transparency": image.has_transparency_data if hasattr(image, "has_transparency_data") else None,
        },
        "media_hints": {
            "is_heic": input_path.suffix.lower() == ".heic",
            "is_heif": input_path.suffix.lower() == ".heif",
            "live_photo_still_supported": input_path.suffix.lower() in {".heic", ".heif"},
            "live_photo_motion_supported": False,
            "note": (
                "HEIC/HEIF still image can be processed when pillow-heif is installed. "
                "Live Photo motion/video sidecar is not used by this app."
            ),
        },
        "macos_summary": macos_summary,
        "macos_mdls": macos_metadata,
        "apple_assetsd": assetsd_metadata,
        "exif_gps_summary": exif_gps_summary,
        "pil_info": _to_json_safe(dict(image.info)),
        "exif": exif_data,
    }


def normalize_image(input_path: Path, tmp_dir: Path, max_edge: int) -> tuple[Path, dict[str, Any]]:
    try:
        with Image.open(input_path) as image:
            photo_metadata = extract_photo_metadata(input_path, image)
            exif_orientation = photo_metadata["exif"].get("Orientation")
            image = ImageOps.exif_transpose(image)
            original_size = image.size
            resized = False

            if max_edge > 0 and max(original_size) > max_edge:
                scale = max_edge / max(original_size)
                resized_size = (
                    max(1, round(original_size[0] * scale)),
                    max(1, round(original_size[1] * scale)),
                )
                image = image.resize(resized_size, Image.Resampling.LANCZOS)
                resized = True

            if image.mode != "RGB":
                image = image.convert("RGB")

            output_path = tmp_dir / "normalized.jpg"
            image.save(output_path, format="JPEG", quality=92, optimize=True)
            processed_size = image.size
    except UnidentifiedImageError as exc:
        suffix = input_path.suffix.lower()
        if suffix in {".heic", ".heif"} and register_heif_opener is None:
            raise ValueError(
                "HEIC/HEIF image is not supported in the current environment. "
                "Install pillow-heif in the app environment to enable it."
            ) from exc
        raise

    return output_path, {
        "photo_metadata": photo_metadata,
        "original_size": {"width": original_size[0], "height": original_size[1]},
        "processed_size": {"width": processed_size[0], "height": processed_size[1]},
        "resized": resized,
        "resize_max_edge": max_edge,
        "orientation_corrected": exif_orientation is not None,
    }


class QwenPipeline:
    def __init__(
        self,
        vl_model_path: str = DEFAULT_VL_MODEL,
        llm_model_path: str = DEFAULT_LLM_MODEL,
        lazy: bool = False,
    ) -> None:
        self.vl_model_path = require_existing_path(vl_model_path, "VL model")
        self._configured_llm_model_path = Path(llm_model_path).expanduser().resolve()
        self.llm_model_path = resolve_optional_path(llm_model_path)
        self.lazy = lazy
        self._load_lock = threading.Lock()
        self._infer_lock = threading.Lock()
        self._vl_model = None
        self._processor = None
        self._llm_model = None
        self._tokenizer = None
        self._loading = False
        self._last_error: str | None = None
        self._last_infer_error: str | None = None
        self._notices: list[str] = []
        self._load_timings = {
            "load_vl_model": 0.0,
            "load_llm_model": 0.0,
        }

    @property
    def loaded(self) -> bool:
        return self.vl_loaded and (not self.llm_available or self.llm_loaded)

    @property
    def vl_loaded(self) -> bool:
        return self._vl_model is not None and self._processor is not None

    @property
    def llm_loaded(self) -> bool:
        return self._llm_model is not None and self._tokenizer is not None

    @property
    def llm_available(self) -> bool:
        return self.llm_model_path is not None

    @property
    def llm_enabled(self) -> bool:
        return self.llm_available and self.llm_loaded

    @property
    def mode(self) -> str:
        return "vl+llm" if self.llm_available else "vl_only"

    @property
    def load_state(self) -> str:
        if self._loading:
            return "loading"
        if self.vl_loaded:
            return "ready"
        if self._last_error:
            return "error"
        return "idle"

    @property
    def capabilities(self) -> dict[str, bool]:
        return (
            {
                "vl": True,
                "llm": self.llm_available,
            }
        )

    def _set_notice(self, message: str) -> None:
        if message not in self._notices:
            self._notices.append(message)

    def get_status(self) -> dict[str, Any]:
        supported_limits = {
            "vl_context_tokens": None,
            "llm_context_tokens": None,
            "vl_max_new_tokens_max": None,
            "llm_max_new_tokens_max": None,
        }
        if self.vl_loaded:
            vl_context = getattr(getattr(self._processor, "tokenizer", None), "model_max_length", None)
            supported_limits = {
                "vl_context_tokens": vl_context,
                "llm_context_tokens": None,
                "vl_max_new_tokens_max": vl_context,
                "llm_max_new_tokens_max": None,
            }
        if self.llm_loaded:
            llm_context = getattr(self._tokenizer, "model_max_length", None)
            supported_limits["llm_context_tokens"] = llm_context
            supported_limits["llm_max_new_tokens_max"] = llm_context
        if not self.llm_available:
            self._set_notice(
                f"LLM model not found; running in VL-only mode: {self._configured_llm_model_path}"
            )
        return {
            "loaded": self.loaded,
            "vl_loaded": self.vl_loaded,
            "llm_loaded": self.llm_loaded,
            "llm_available": self.llm_available,
            "mode": self.mode,
            "load_state": self.load_state,
            "capabilities": self.capabilities,
            "loading": self._loading,
            "load_error": self._last_error,
            "last_infer_error": self._last_infer_error,
            "notices": list(self._notices),
            "vl_model": str(self.vl_model_path),
            "llm_model": str(self.llm_model_path) if self.llm_model_path else None,
            "configured_llm_model": str(self._configured_llm_model_path),
            "lazy": self.lazy,
            "timing_seconds": dict(self._load_timings),
            "supported_limits": supported_limits,
        }

    def _load_llm_with_fixed_tokenizer(self) -> tuple[Any, Any]:
        if self.llm_model_path is None:
            raise FileNotFoundError(
                f"LLM model not found: {self._configured_llm_model_path}"
            )
        logger = logging.getLogger("transformers.tokenization_utils_tokenizers")

        class _RegexWarningFilter(logging.Filter):
            def filter(self, record: logging.LogRecord) -> bool:
                return "incorrect regex pattern" not in record.getMessage()

        warning_filter = _RegexWarningFilter()
        logger.addFilter(warning_filter)
        try:
            llm_model, tokenizer = lm_load(
                str(self.llm_model_path),
                lazy=self.lazy,
            )
        finally:
            logger.removeFilter(warning_filter)

        wrapped = tokenizer._tokenizer
        TokenizersBackend._patch_mistral_regex(
            wrapped,
            str(self.llm_model_path),
            is_local=True,
            init_kwargs=getattr(wrapped, "init_kwargs", None),
            fix_mistral_regex=True,
        )
        self._set_notice(
            "Qwen3.5 tokenizer regex fix was applied in memory to avoid the known fast-tokenizer warning."
        )
        return llm_model, tokenizer

    def load_models(self) -> dict[str, float]:
        if self.loaded:
            return dict(self._load_timings)

        with self._load_lock:
            if self.loaded:
                return dict(self._load_timings)
            self._loading = True
            self._last_error = None
            try:
                started = time.perf_counter()
                self._vl_model, self._processor = vlm_load(
                    str(self.vl_model_path),
                    lazy=self.lazy,
                )
                self._load_timings["load_vl_model"] = round(
                    time.perf_counter() - started,
                    2,
                )

                if self.llm_available:
                    started = time.perf_counter()
                    self._llm_model, self._tokenizer = self._load_llm_with_fixed_tokenizer()
                    self._load_timings["load_llm_model"] = round(
                        time.perf_counter() - started,
                        2,
                    )
                else:
                    self._load_timings["load_llm_model"] = 0.0
            except Exception as exc:
                self._last_error = str(exc)
                raise
            finally:
                self._loading = False

        return dict(self._load_timings)

    def load_vl_model(self) -> dict[str, float]:
        if self.vl_loaded:
            return dict(self._load_timings)

        with self._load_lock:
            if self.vl_loaded:
                return dict(self._load_timings)
            self._loading = True
            self._last_error = None
            try:
                started = time.perf_counter()
                self._vl_model, self._processor = vlm_load(
                    str(self.vl_model_path),
                    lazy=self.lazy,
                )
                self._load_timings["load_vl_model"] = round(
                    time.perf_counter() - started,
                    2,
                )
            except Exception as exc:
                self._last_error = str(exc)
                raise
            finally:
                self._loading = False

        return dict(self._load_timings)

    def infer(self, image_path: str | Path, config: PipelineConfig | None = None) -> dict[str, Any]:
        return self._infer(image_path, config, progress_callback=None)

    def _emit_progress(
        self,
        progress_callback: Callable[[dict[str, Any]], None] | None,
        **payload: Any,
    ) -> None:
        if progress_callback is not None:
            progress_callback(payload)

    def _infer(
        self,
        image_path: str | Path,
        config: PipelineConfig | None = None,
        progress_callback: Callable[[dict[str, Any]], None] | None = None,
    ) -> dict[str, Any]:
        cfg = config or PipelineConfig()
        source_path = require_existing_path(str(image_path), "Image")
        load_timings = self.load_models()
        self._emit_progress(
            progress_callback,
            state="models_ready",
            message="模型已加载，开始处理图片。",
            timing_seconds=load_timings,
        )

        try:
            with self._infer_lock:
                with tempfile.TemporaryDirectory(prefix="qwen-pipeline-") as tmp:
                    tmp_dir = Path(tmp)
                    normalized_image, image_info = normalize_image(
                        source_path,
                        tmp_dir,
                        cfg.resize_max_edge,
                    )
                    self._emit_progress(
                        progress_callback,
                        state="image_ready",
                        message="图片预处理完成，开始运行 Qwen3-VL。",
                        image_info=image_info,
                        timing_seconds=load_timings,
                    )

                    mx.random.seed(cfg.seed)
                    sampler = make_sampler(cfg.temperature, cfg.top_p)

                    vl_prompt = build_vl_prompt(
                        self._processor,
                        str(normalized_image),
                        cfg.vl_prompt,
                    )
                    self._emit_progress(
                        progress_callback,
                        state="running_vl",
                        message="Qwen3-VL 推理中。",
                        image_info=image_info,
                        timing_seconds=load_timings,
                    )

                    started = time.perf_counter()
                    vl_result = vlm_generate(
                        self._vl_model,
                        self._processor,
                        prompt=vl_prompt,
                        image=str(normalized_image),
                        max_tokens=cfg.vl_max_tokens,
                        temperature=cfg.temperature,
                        top_p=cfg.top_p,
                        sampler=sampler,
                        verbose=False,
                    )
                    vision_seconds = round(time.perf_counter() - started, 2)
                    current_timings = {
                        **load_timings,
                        "vision_generate": vision_seconds,
                    }
                    if not self.llm_enabled:
                        self._emit_progress(
                            progress_callback,
                            state="completed",
                            message="Qwen3-VL 已完成。",
                            image_info=image_info,
                            timing_seconds=current_timings,
                            vl_output=vl_result.text.strip(),
                        )
                        return {
                            "image": str(source_path),
                            "vl_model": str(self.vl_model_path),
                            "llm_model": None,
                            "lazy": self.lazy,
                            "config": asdict(cfg),
                            "image_info": image_info,
                            "timing_seconds": current_timings,
                            "vl_output": vl_result.text.strip(),
                            "llm_output": None,
                            "llm_raw_output": None,
                            "llm_reasoning": None,
                        }

                    self._emit_progress(
                        progress_callback,
                        state="vl_done",
                        message="Qwen3-VL 已完成，开始运行 Qwen3.5。",
                        image_info=image_info,
                        timing_seconds=current_timings,
                        vl_output=vl_result.text.strip(),
                    )

                    llm_prompt = build_llm_prompt(
                        self._tokenizer,
                        cfg.system,
                        cfg.task,
                        vl_result.text,
                    )
                    self._emit_progress(
                        progress_callback,
                        state="running_llm",
                        message="Qwen3.5 推理中。",
                        image_info=image_info,
                        timing_seconds=current_timings,
                        vl_output=vl_result.text.strip(),
                    )

                    started = time.perf_counter()
                    llm_raw = lm_generate(
                        self._llm_model,
                        self._tokenizer,
                        prompt=llm_prompt,
                        max_tokens=cfg.llm_max_tokens,
                        sampler=sampler,
                        verbose=False,
                    )
                    llm_seconds = round(time.perf_counter() - started, 2)
        except Exception as exc:
            self._last_infer_error = str(exc)
            raise
        else:
            self._last_infer_error = None

        llm_parts = split_thinking_output(llm_raw)
        llm_output = llm_parts["answer"] if cfg.strip_thinking else llm_parts["raw"]

        return {
            "image": str(source_path),
            "vl_model": str(self.vl_model_path),
            "llm_model": str(self.llm_model_path) if self.llm_model_path else None,
            "lazy": self.lazy,
            "config": asdict(cfg),
            "image_info": image_info,
            "timing_seconds": {
                **load_timings,
                "vision_generate": vision_seconds,
                "llm_generate": llm_seconds,
            },
            "vl_output": vl_result.text.strip(),
            "llm_output": llm_output.strip(),
            "llm_raw_output": llm_parts["raw"],
            "llm_reasoning": llm_parts["reasoning"],
        }

    def infer_with_progress(
        self,
        image_path: str | Path,
        config: PipelineConfig | None = None,
        progress_callback: Callable[[dict[str, Any]], None] | None = None,
    ) -> dict[str, Any]:
        return self._infer(image_path, config, progress_callback=progress_callback)

    def infer_vl_only(
        self,
        image_path: str | Path,
        config: PipelineConfig | None = None,
    ) -> dict[str, Any]:
        cfg = config or PipelineConfig()
        source_path = require_existing_path(str(image_path), "Image")
        load_timings = self.load_vl_model()
        try:
            with self._infer_lock:
                with tempfile.TemporaryDirectory(prefix="qwen-pipeline-vl-") as tmp:
                    tmp_dir = Path(tmp)
                    normalized_image, image_info = normalize_image(
                        source_path,
                        tmp_dir,
                        cfg.resize_max_edge,
                    )
                    mx.random.seed(cfg.seed)
                    sampler = make_sampler(cfg.temperature, cfg.top_p)
                    vl_prompt = build_vl_prompt(
                        self._processor,
                        str(normalized_image),
                        cfg.vl_prompt,
                    )
                    started = time.perf_counter()
                    vl_result = vlm_generate(
                        self._vl_model,
                        self._processor,
                        prompt=vl_prompt,
                        image=str(normalized_image),
                        max_tokens=cfg.vl_max_tokens,
                        temperature=cfg.temperature,
                        top_p=cfg.top_p,
                        sampler=sampler,
                        verbose=False,
                    )
                    vision_seconds = round(time.perf_counter() - started, 2)
        except Exception as exc:
            self._last_infer_error = str(exc)
            raise
        else:
            self._last_infer_error = None

        return {
            "image": str(source_path),
            "vl_model": str(self.vl_model_path),
            "lazy": self.lazy,
            "config": asdict(cfg),
            "image_info": image_info,
            "timing_seconds": {
                **load_timings,
                "vision_generate": vision_seconds,
            },
            "vl_output": vl_result.text.strip(),
        }

    def stream_infer(
        self,
        image_path: str | Path,
        config: PipelineConfig | None = None,
        should_cancel: Callable[[], bool] | None = None,
    ) -> Any:
        def check_cancel() -> None:
            if should_cancel is not None and should_cancel():
                raise InferenceCancelled("Inference cancelled.")

        cfg = config or PipelineConfig()
        source_path = require_existing_path(str(image_path), "Image")
        load_timings = self.load_models()
        yield {
            "event": "status",
            "data": {
                "state": "models_ready",
                "message": "模型已加载，开始处理图片。",
                "timing_seconds": load_timings,
            },
        }

        try:
            with self._infer_lock:
                with tempfile.TemporaryDirectory(prefix="qwen-pipeline-") as tmp:
                    check_cancel()
                    tmp_dir = Path(tmp)
                    normalized_image, image_info = normalize_image(
                        source_path,
                        tmp_dir,
                        cfg.resize_max_edge,
                    )
                    yield {
                        "event": "image_info",
                        "data": {
                            "image_info": image_info,
                            "state": "image_ready",
                            "message": "图片预处理完成，开始运行 Qwen3-VL。",
                            "timing_seconds": load_timings,
                        },
                    }

                    mx.random.seed(cfg.seed)
                    sampler = make_sampler(cfg.temperature, cfg.top_p)
                    vl_prompt = build_vl_prompt(
                        self._processor,
                        str(normalized_image),
                        cfg.vl_prompt,
                    )
                    yield {
                        "event": "status",
                        "data": {
                            "state": "running_vl",
                            "message": "Qwen3-VL 推理中。",
                            "image_info": image_info,
                            "timing_seconds": load_timings,
                        },
                    }

                    started = time.perf_counter()
                    vl_text = ""
                    vl_stats: dict[str, Any] = {}
                    for chunk in vlm_stream_generate(
                        self._vl_model,
                        self._processor,
                        prompt=vl_prompt,
                        image=str(normalized_image),
                        max_tokens=cfg.vl_max_tokens,
                        temperature=cfg.temperature,
                        top_p=cfg.top_p,
                        sampler=sampler,
                    ):
                        check_cancel()
                        delta = chunk.text or ""
                        if not delta:
                            continue
                        vl_text += delta
                        vl_stats = {
                            "prompt_tokens": chunk.prompt_tokens,
                            "generation_tokens": chunk.generation_tokens,
                            "prompt_tps": chunk.prompt_tps,
                            "generation_tps": chunk.generation_tps,
                            "peak_memory": chunk.peak_memory,
                        }
                        yield {
                            "event": "vl_chunk",
                            "data": {
                                "text": delta,
                                "accumulated_text": vl_text,
                                "stats": vl_stats,
                            },
                        }
                    vision_seconds = round(time.perf_counter() - started, 2)
                    current_timings = {
                        **load_timings,
                        "vision_generate": vision_seconds,
                    }

                    if not self.llm_enabled:
                        result = {
                            "image": str(source_path),
                            "vl_model": str(self.vl_model_path),
                            "llm_model": None,
                            "lazy": self.lazy,
                            "config": asdict(cfg),
                            "image_info": image_info,
                            "timing_seconds": current_timings,
                            "vl_output": vl_text.strip(),
                            "llm_output": None,
                            "llm_raw_output": None,
                            "llm_reasoning": None,
                        }
                        yield {
                            "event": "complete",
                            "data": result,
                        }
                        return

                    yield {
                        "event": "status",
                        "data": {
                            "state": "running_llm",
                            "message": "Qwen3.5 推理中。",
                            "image_info": image_info,
                            "timing_seconds": current_timings,
                            "vl_output": vl_text.strip(),
                        },
                    }
                    llm_prompt = build_llm_prompt(
                        self._tokenizer,
                        cfg.system,
                        cfg.task,
                        vl_text,
                    )
                    started = time.perf_counter()
                    llm_raw = ""
                    llm_stats: dict[str, Any] = {}
                    for chunk in lm_stream_generate(
                        self._llm_model,
                        self._tokenizer,
                        prompt=llm_prompt,
                        max_tokens=cfg.llm_max_tokens,
                        sampler=sampler,
                    ):
                        check_cancel()
                        delta = chunk.text or ""
                        if not delta:
                            continue
                        llm_raw += delta
                        llm_stats = {
                            "prompt_tokens": chunk.prompt_tokens,
                            "generation_tokens": chunk.generation_tokens,
                            "prompt_tps": chunk.prompt_tps,
                            "generation_tps": chunk.generation_tps,
                            "peak_memory": chunk.peak_memory,
                            "finish_reason": chunk.finish_reason,
                        }
                        llm_parts = split_thinking_progress(llm_raw)
                        llm_output = (
                            llm_parts["answer"]
                            if cfg.strip_thinking
                            else llm_parts["raw"]
                        )
                        yield {
                            "event": "llm_chunk",
                            "data": {
                                "text": delta,
                                "llm_output": llm_output,
                                "llm_raw_output": llm_parts["raw"],
                                "llm_reasoning": llm_parts["reasoning"],
                                "in_thinking": llm_parts["in_thinking"],
                                "stats": llm_stats,
                            },
                        }
                    llm_seconds = round(time.perf_counter() - started, 2)
        except Exception as exc:
            self._last_infer_error = str(exc)
            raise
        else:
            self._last_infer_error = None

        llm_parts = split_thinking_output(llm_raw)
        llm_output = llm_parts["answer"] if cfg.strip_thinking else llm_parts["raw"]
        yield {
            "event": "complete",
            "data": {
                "image": str(source_path),
                "vl_model": str(self.vl_model_path),
                "llm_model": str(self.llm_model_path) if self.llm_model_path else None,
                "lazy": self.lazy,
                "config": asdict(cfg),
                "image_info": image_info,
                "timing_seconds": {
                    **load_timings,
                    "vision_generate": vision_seconds,
                    "llm_generate": llm_seconds,
                },
                "vl_output": vl_text.strip(),
                "llm_output": llm_output.strip(),
                "llm_raw_output": llm_parts["raw"],
                "llm_reasoning": llm_parts["reasoning"],
            },
        }
