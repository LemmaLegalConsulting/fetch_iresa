# Follow-up question eval (Promptfoo)

This eval checks follow-up question generation quality using:
- LLM-graded `llm-rubric` assertions
- A Python `textstat` readability check (Flesch–Kincaid grade < 9)

## Files

- `promptfoo/followup_questions_eval.yaml` – Promptfoo suite config (uses `iresa_followup_questions_only.csv`)
- `promptfoo/assertions/followup_textstat_grade.py` – Python assertion using `textstat`

## Prereqs

- Promptfoo installed (`promptfoo` or `npx promptfoo`)
- API keys as needed:
  - `OPENAI_API_KEY` for the rubric grader (and for any enabled OpenAI providers)
  - If you keep `enabled_providers` as-is, you may also need `GEMINI_API_KEY`, `OPENROUTER_API_KEY`, `SPOT_API_KEY`.

### Python deps for readability check

Because many Linux distros block system-wide `pip install` (PEP 668), use a venv:

```bash
cd /home/quinten/fetch_iresa
python3 -m venv .venv
. .venv/bin/activate
pip install -r requirements-dev.txt
```

## Run

Validate config:

```bash
cd /home/quinten/fetch_iresa
promptfoo validate -c promptfoo/followup_questions_eval.yaml
```

Run eval:

```bash
cd /home/quinten/fetch_iresa
promptfoo eval -c promptfoo/followup_questions_eval.yaml
```

Human review/rating should be done in the Promptfoo web UI (promptfoo.dev), not via local HTML exports.

Tip: override grader model on the CLI if desired:

```bash
promptfoo eval -c promptfoo/followup_questions_eval.yaml --grader openai:gpt-4.1-mini
```

The tracked IRESA CSV is a safe placeholder for this public repo. Replace it locally with your real IRESA follow-up eval rows before running meaningful evals.
