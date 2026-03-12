from __future__ import annotations

import json
import logging
import os
import tempfile
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

import mlx.core as mx
from mlx_lm import generate as lm_generate
from mlx_lm import load as lm_load
from mlx_lm.sample_utils import make_sampler
from mlx_vlm import generate as vlm_generate
from mlx_vlm import load as vlm_load
from PIL import Image, ImageOps
from transformers.tokenization_utils_tokenizers import TokenizersBackend


DEFAULT_VL_MODEL = os.path.expanduser(
    "~/.lmstudio/models/lmstudio-community/Qwen3-VL-8B-Instruct-MLX-8bit"
)
DEFAULT_LLM_MODEL = os.path.expanduser(
    "~/.lmstudio/models/nightmedia/"
    "Qwen3.5-27B-Claude-4.6-Opus-Reasoning-Distilled-qx64-hi-mlx"
)
DEFAULT_VL_PROMPT = "请用中文提取这张图片里的关键信息，优先描述可见文字、界面元素、主体内容和上下文。"
DEFAULT_TASK = "基于上面的图片解析结果，整理成一段清晰的中文说明。如果有文字信息，请总结重点。"
DEFAULT_SYSTEM = "你是一个严谨的中文助手。请基于提供的视觉解析结果完成任务，不要编造图片中不存在的细节。"


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


def normalize_image(input_path: Path, tmp_dir: Path, max_edge: int) -> tuple[Path, dict[str, Any]]:
    with Image.open(input_path) as image:
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

    return output_path, {
        "original_size": {"width": original_size[0], "height": original_size[1]},
        "processed_size": {"width": processed_size[0], "height": processed_size[1]},
        "resized": resized,
        "resize_max_edge": max_edge,
    }


class QwenPipeline:
    def __init__(
        self,
        vl_model_path: str = DEFAULT_VL_MODEL,
        llm_model_path: str = DEFAULT_LLM_MODEL,
        lazy: bool = False,
    ) -> None:
        self.vl_model_path = require_existing_path(vl_model_path, "VL model")
        self.llm_model_path = require_existing_path(llm_model_path, "LLM model")
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
        return (
            self._vl_model is not None
            and self._processor is not None
            and self._llm_model is not None
            and self._tokenizer is not None
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
        if self.loaded:
            vl_context = getattr(getattr(self._processor, "tokenizer", None), "model_max_length", None)
            llm_context = getattr(self._tokenizer, "model_max_length", None)
            supported_limits = {
                "vl_context_tokens": vl_context,
                "llm_context_tokens": llm_context,
                "vl_max_new_tokens_max": vl_context,
                "llm_max_new_tokens_max": llm_context,
            }
        return {
            "loaded": self.loaded,
            "loading": self._loading,
            "load_error": self._last_error,
            "last_infer_error": self._last_infer_error,
            "notices": list(self._notices),
            "vl_model": str(self.vl_model_path),
            "llm_model": str(self.llm_model_path),
            "lazy": self.lazy,
            "timing_seconds": dict(self._load_timings),
            "supported_limits": supported_limits,
        }

    def _load_llm_with_fixed_tokenizer(self) -> tuple[Any, Any]:
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

                started = time.perf_counter()
                self._llm_model, self._tokenizer = self._load_llm_with_fixed_tokenizer()
                self._load_timings["load_llm_model"] = round(
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
            "llm_model": str(self.llm_model_path),
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
