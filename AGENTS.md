# Repository Guidelines

## Project Structure & Module Organization
- `app/main.py`: FastAPI entrypoint; serves `/api/classify` (and `/` in dev).
- `app/services/`: Orchestrates classification flow and result merging.
- `app/providers/`: Model backends (OpenAI, Gemini, TF‑IDF, keyword, etc.).
- `app/models/`: Pydantic request/response schemas.
- `app/core/`: Config and cache helpers.
- `app/utils/`: Logging, backoff/retry utilities.
- `app/prompts/` and `app/data/`: Prompt templates and taxonomy files.
- `tests/`: Pytest suite for API auth and service logic.
- Root: `requirements.txt`, `Dockerfile`, `.env.example`, `index.html`, `promptfooconfig.yaml`.

## Build, Test, and Development Commands
- Install deps: `pip install -r requirements.txt`
- Run API (dev): `ENV=dev uvicorn app.main:app --reload`
- Run API (prod-ish): `uvicorn app.main:app --host 0.0.0.0 --port 8080`
- Tests: `pytest -q` (single test: `pytest tests/test_api_auth.py::test_classify_valid_token`)
- Type check (optional): `pyright` (config in `pyrightconfig.json`)
- Docker: `docker build -t fetch .` then `docker run -p 8080:8080 --env-file .env fetch`

## Coding Style & Naming Conventions
- Language: Python 3.12 with type hints; 4‑space indent.
- Formatting: Black (e.g., `black app tests`).
- Names: modules/files `snake_case.py`, classes `PascalCase`, functions/vars `snake_case`.
- Imports: prefer absolute within `app.`; keep sections stdlib/third‑party/local.

## Testing Guidelines
- Framework: pytest + pytest‑asyncio.
- Layout: tests under `tests/`, files named `test_*.py`.
- Scope: unit tests for providers/services; mock external APIs; add regression tests for taxonomy logic.
- Running in CI/local: `pytest -q`; keep tests deterministic and offline.

## Commit & Pull Request Guidelines
- Commits: imperative, concise subjects (≤72 chars). Conventional prefixes welcome (`feat:`, `fix:`, `chore:`). Avoid `WIP` in final history.
- PRs: clear description, linked issues, steps to validate (commands or curl), note config changes (`.env` keys like `API_TOKENS`, `OPENAI_API_KEY`), and update README/AGENTS if behavior changes.

## Security & Configuration Tips
- Secrets: never commit `.env`. Copy `.env.example` and set `OPENAI_API_KEY`, `GEMINI_API_KEY`, `OPENROUTER_API_KEY`, and `API_TOKENS`.
- Dev mode: set `ENV=dev` to bypass auth and serve `index.html` locally.
- Data: place taxonomy files under `app/data/`; don’t commit proprietary datasets.
- PromptFoo & local eval notes: see `promptfoo/README.md` (or create `docs/PROMPTFOO.md`) for venv/caching/run instructions.
  - **Always activate venv** before running any tests: `source .venv/bin/activate`
  - **Run evals without output redirection** to inspect ongoing progress: `promptfoo eval -c config.yaml` (not `... 2>&1 | tail`)
  - **PromptFoo can be installed system-wide** and doesn't usually require `npx` (use `pip install promptfoo` globally or in venv)

