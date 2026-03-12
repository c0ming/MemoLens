# Local MLX Qwen Pipeline

This directory contains:

1. A CLI pipeline that loads `Qwen3-VL` and `Qwen3.5` locally.
2. A FastAPI service that keeps both models loaded in one process.
3. A browser demo page for uploading an image, streaming progress, and reading both outputs.
4. A local person ID workflow backed by `InsightFace` embeddings and a browser management page at `/people`.

## Environment

The current setup was verified with a local virtual environment:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -U pip setuptools wheel
.venv/bin/pip install "mlx-lm==0.31.0" "mlx-vlm==0.4.0" "transformers==5.3.0" pillow pillow-heif
.venv/bin/pip install torch torchvision python-multipart insightface onnxruntime
```

`torch` and `torchvision` are still needed because the `Qwen3-VL` processor initializes its video processor even if you only run image inference.

`pillow-heif` is needed if you want to open `HEIC` / `HEIF` images. For iPhone `Live Photo`, this app currently only uses the still image asset and does not process the paired motion video.

`insightface` and `onnxruntime` are used for local person ID creation and face matching. The first time the face pipeline runs it will download the `buffalo_l` model bundle into `~/.insightface/models/`.

## CLI

If your models are already in the same LM Studio directories found on this machine, you can run:

```bash
.venv/bin/python run_vl_to_qwen.py --image ./test.jpg
```

You can also pass explicit model paths:

```bash
.venv/bin/python run_vl_to_qwen.py \
  --image /absolute/path/to/image.jpg \
  --vl-model /absolute/path/to/Qwen3-VL-MLX \
  --llm-model /absolute/path/to/Qwen3.5-MLX
```

Optional JSON output:

```bash
.venv/bin/python run_vl_to_qwen.py \
  --image ./test.jpg \
  --resize-max-edge 768 \
  --save-json outputs/run.json
```

`--resize-max-edge` now defaults to `768` because image size has a larger impact on VL latency than generation length.

Quick verification:

```bash
.venv/bin/python -m py_compile app.py qwen_pipeline.py run_vl_to_qwen.py person_features.py face_identity.py
```

## API And Demo

Start the local API:

```bash
.venv/bin/python app.py
```

Then open:

```text
http://127.0.0.1:8000
```

API endpoints:

```text
GET  /api/status
POST /api/jobs
GET  /api/jobs/{job_id}
POST /api/infer
POST /api/stream
GET  /people
GET  /api/people
POST /api/people
PUT  /api/people/{profile_id}
DELETE /api/people/{profile_id}
POST /api/people/{profile_id}/extract
POST /api/people/{profile_id}/clear-samples
```

Synchronous `curl` upload:

```bash
curl -X POST http://127.0.0.1:8000/api/infer \
  -F image=@./test.jpg \
  -F resize_max_edge=768 \
  -F vl_max_tokens=131072 \
  -F llm_max_tokens=131072
```

Async job example:

```bash
curl -X POST http://127.0.0.1:8000/api/jobs \
  -F image=@./test.jpg \
  -F resize_max_edge=768
```

Streaming example:

```bash
curl -N -X POST http://127.0.0.1:8000/api/stream \
  -F image=@./test.jpg \
  -F resize_max_edge=768
```

## Person IDs

Open:

```text
http://127.0.0.1:8000/people
```

The management page lets you:

- create a person profile with a stable `person_id`
- edit name and notes
- upload multiple sample photos to build a local face embedding
- clear samples while keeping the profile
- delete the full profile

Built profiles are stored under:

```text
outputs/person_features/
```

Each profile may contain:

- `profile.json`
- `preview.jpg`
- saved sample images

## Notes

- The FastAPI app preloads both models on startup, so the first browser request does not pay the full load cost.
- The inference path is serialized through a lock because the current MLX model objects are reused in-process.
- The browser demo now uses the streaming API so `Identified People` and `VL Output` can update before `Qwen3.5` finishes.
- `Identified People` comes from local `InsightFace` matching, not from `Qwen-VL`.
- The effective prompt shown in the UI is the final prompt after person-ID context has been injected.
- The page includes `速度优先 / 平衡 / 细节优先` presets. Use `速度优先` first if one photo is taking around 20 seconds.
- By default the CLI strips reasoning text from the final `Qwen3.5` answer. Use `--keep-thinking` if you want the raw reasoning output.
- By default the script fully loads both models. Use `--lazy` if you want lazy loading instead.
- Default UI/API values are currently `resize_max_edge=768`, `vl_max_tokens=131072`, and `llm_max_tokens=131072`.
- The default text model path currently points to:

```text
~/.lmstudio/models/nightmedia/Qwen3.5-27B-Claude-4.6-Opus-Reasoning-Distilled-qx64-hi-mlx
```

- The default vision model path currently points to:

```text
~/.lmstudio/models/lmstudio-community/Qwen3-VL-8B-Instruct-MLX-8bit
```
