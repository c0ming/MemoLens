# Repository Guidelines

## Project Structure & Module Organization

- `app.py`: FastAPI entry point. Exposes `/api/status`, `/api/jobs`, and static demo hosting.
- `face_identity.py`: Local face detection / embedding / matching helpers built on `InsightFace`.
- `person_features.py`: Person profile storage, sample management, and local ID building.
- `qwen_pipeline.py`: Shared inference pipeline, image normalization, model loading, and progress reporting.
- `run_vl_to_qwen.py`: CLI wrapper for one-off local runs.
- `static/index.html`: Single-page browser demo.
- `static/people.html`: Person ID management page.
- `README.md`: Setup and local run instructions.
- `outputs/`: Local run artifacts and debug JSON. Ignored by Git by default.

Keep new Python modules at the repository root unless a clear package split is needed. Put browser assets under `static/`.

## Build, Test, and Development Commands

- `python3 -m venv .venv`: Create the local virtual environment.
- `.venv/bin/pip install -U pip setuptools wheel`: Update core tooling.
- `.venv/bin/python -m py_compile app.py qwen_pipeline.py run_vl_to_qwen.py person_features.py face_identity.py`: Fast syntax check before committing.
- `.venv/bin/python app.py`: Start the local API and demo page at `http://127.0.0.1:8000`.
- `.venv/bin/python run_vl_to_qwen.py --image ./test.jpg`: Run the CLI pipeline directly.

When changing inference behavior, verify both the API and CLI paths. When changing person-ID behavior, verify `/people` plus one homepage inference.

## Coding Style & Naming Conventions

- Use Python 3 with 4-space indentation and type hints where practical.
- Prefer small, explicit helper functions over large nested blocks.
- Use `snake_case` for Python variables/functions and lowercase file names such as `qwen_pipeline.py`.
- Keep frontend code minimal and framework-free; use clear DOM ids like `vl-output` and `service-state`.
- Preserve ASCII unless a file already contains Chinese UI text.

## Testing Guidelines

There is no dedicated test suite yet. Use these checks:

- Run `py_compile` on edited Python files.
- Smoke test the API with `curl http://127.0.0.1:8000/api/status`.
- For UI or pipeline changes, run one image through the browser or CLI and confirm `VL Output` and `Qwen3.5 Output` update as expected.
- For person-ID changes, verify one profile can be created, built from multiple samples, and matched during homepage inference.

If you add tests later, place them in `tests/` and name files `test_*.py`.

## Commit & Pull Request Guidelines

This repository currently has no commit history. Use short imperative commit subjects with a prefix, for example:

- `feat: add async job polling for VL progress`
- `fix: reduce default resize edge for faster vision inference`

Pull requests should include:

- A short description of the user-facing change
- Verification steps you ran
- Screenshots for UI updates
- Notes about model defaults, performance, or local environment assumptions

## Security & Configuration Tips

- Do not commit `.venv/`, logs, or personal photo assets unless explicitly intended.
- `outputs/` is still ignored by default, but `outputs/person_features/` may be force-added intentionally when the user explicitly wants local person-ID data versioned.
- Treat model paths under `~/.lmstudio/models/` as local machine configuration.
- Keep family photo processing local; avoid external identity APIs.
- `InsightFace` downloads model files into `~/.insightface/models/`; do not assume those are vendored in the repo.
