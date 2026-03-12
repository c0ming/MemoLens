#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from qwen_pipeline import (
    DEFAULT_LLM_MODEL,
    DEFAULT_SYSTEM,
    DEFAULT_TASK,
    DEFAULT_VL_MODEL,
    DEFAULT_VL_PROMPT,
    PipelineConfig,
    QwenPipeline,
    maybe_write_json,
    require_existing_path,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Load a local MLX VL model and a local MLX LLM in one process, "
        "then feed the VL output into the LLM."
    )
    parser.add_argument("--image", required=True, help="Path to the input image.")
    parser.add_argument(
        "--vl-model",
        default=DEFAULT_VL_MODEL,
        help=f"Path to the Qwen VL MLX model. Default: {DEFAULT_VL_MODEL}",
    )
    parser.add_argument(
        "--llm-model",
        default=DEFAULT_LLM_MODEL,
        help=f"Path to the Qwen3.5 MLX model. Default: {DEFAULT_LLM_MODEL}",
    )
    parser.add_argument(
        "--vl-prompt",
        default=DEFAULT_VL_PROMPT,
        help="Prompt used for the vision model.",
    )
    parser.add_argument(
        "--task",
        default=DEFAULT_TASK,
        help="Prompt used for the text model after the VL step.",
    )
    parser.add_argument(
        "--system",
        default=DEFAULT_SYSTEM,
        help="System prompt for the text model.",
    )
    parser.add_argument(
        "--vl-max-tokens",
        type=int,
        default=256,
        help="Max new tokens for the vision model.",
    )
    parser.add_argument(
        "--llm-max-tokens",
        type=int,
        default=512,
        help="Max new tokens for the text model.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.2,
        help="Sampling temperature used for both stages.",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.9,
        help="Top-p used for both stages.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=7,
        help="Random seed used for both stages.",
    )
    parser.add_argument(
        "--lazy",
        action="store_true",
        help="Enable lazy model loading. By default both models are fully loaded.",
    )
    parser.add_argument(
        "--resize-max-edge",
        type=int,
        default=1280,
        help="Resize the input image to this max edge before VL inference. Use 0 to disable.",
    )
    parser.add_argument(
        "--keep-thinking",
        action="store_true",
        help="Keep raw reasoning text from the Qwen3.5 output.",
    )
    parser.add_argument(
        "--save-json",
        help="Optional path to save the full run result as JSON.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    try:
        image_path = require_existing_path(args.image, "Image")
        vl_model_path = require_existing_path(args.vl_model, "VL model")
        llm_model_path = require_existing_path(args.llm_model, "LLM model")
    except FileNotFoundError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    print(f"[info] image={image_path}", file=sys.stderr)
    print(f"[info] vl_model={vl_model_path}", file=sys.stderr)
    print(f"[info] llm_model={llm_model_path}", file=sys.stderr)
    print(f"[info] lazy={args.lazy}", file=sys.stderr)
    pipeline = QwenPipeline(
        vl_model_path=str(vl_model_path),
        llm_model_path=str(llm_model_path),
        lazy=args.lazy,
    )
    config = PipelineConfig(
        vl_prompt=args.vl_prompt,
        task=args.task,
        system=args.system,
        vl_max_tokens=args.vl_max_tokens,
        llm_max_tokens=args.llm_max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        seed=args.seed,
        resize_max_edge=args.resize_max_edge,
        strip_thinking=not args.keep_thinking,
    )
    payload = pipeline.infer(str(image_path), config)

    maybe_write_json(args.save_json, payload)
    if args.save_json:
        print(f"[saved] {args.save_json}", file=sys.stderr)

    print("=== VL OUTPUT ===")
    print(payload["vl_output"])
    print()
    print("=== QWEN3.5 OUTPUT ===")
    print(payload["llm_output"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
