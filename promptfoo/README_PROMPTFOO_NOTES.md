# PromptFoo: Local eval notes

This file documents the specific steps and troubleshooting tips to ensure PromptFoo runs reliably in this repo (especially when using GitHub Copilot / VS Code).

## Quick checklist ✅
- Activate the project venv before running PromptFoo: `source .venv/bin/activate`
- Install dev deps (if not already): `pip install -r requirements-dev.txt`
- Ensure `OPENAI_API_KEY` is set in your environment for real LLM runs, or set `OFFLINE_MODE=1` for offline testing.

## Common commands
- Run a single-case eval (debug):
  `promptfoo eval -c promptfoo/followup_questions_eval.yaml --filter-first-n 1`
- Run full eval (no cache):
  `promptfoo eval -c promptfoo/followup_questions_eval.yaml --no-cache`

## Caching
- Enable provider-level caching in `followup_questions_eval.yaml` by setting `cache_enabled: true` (the default cache dir is `./cache/provider_responses_cache`).
- Inspect cache DB: `sqlite3 ./cache/provider_responses_cache/cache.db 'select key from Cache limit 5;'` or use Python to unpickle values.
- Clear cache: `python clear_all_caches.py`
- Signs of cache usage in PromptFoo output:
  - `Cache hit for <provider>`
  - `Grading: <tokens> (cached)`

## Troubleshooting
- Error: `ModuleNotFoundError: No module named 'openai'` → activate the repo venv and ensure `openai` is installed there: `pip install -r requirements-dev.txt`.
- Error: `NameError: name 'List' is not defined` → run `python -m py_compile` on `promptfoo/assertions` to spot missing typing imports.
- PromptFoo worker logs: `~/.promptfoo/logs/promptfoo-debug-*.log` and `promptfoo-error-*.log`
- Classification eval data lives in `promptfoo/iresa_sample_data.csv`
- Follow-up eval data lives in `promptfoo/iresa_followup_questions_only.csv`

## Notes for Copilot users
- Make sure VS Code's interpreter is set to the repo `.venv` so Copilot / tasks use the same environment.
- If running evals from the VS Code terminal, remember to activate `.venv` first.

---
