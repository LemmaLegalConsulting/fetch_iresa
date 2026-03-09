# Follow-up Question Evaluation (Promptfoo)

This folder contains the Promptfoo eval config and helpers for follow-up question quality.
Human review/rating should be done in the Promptfoo web UI (promptfoo.dev), not via local HTML exports.

## Quick Start

```bash
cd /home/quinten/fetch_iresa
promptfoo eval -c promptfoo/followup_questions_eval.yaml
```

> Note: Promptfoo's Python worker must have optional packages available for readability assertions.
> If needed:
>
> ```bash
> .venv/bin/python -m pip install scireadability nltk
> ```
>
> To skip the preflight check, set `RUN_EVAL_FORCE=1` when using `./promptfoo/run_eval.sh`.

## Files

- `followup_questions_eval.yaml` — main eval config (heuristics + LLM rubrics + readability)
- `iresa_sample_data.csv` — placeholder classification eval cases for the IRESA taxonomy
- `iresa_followup_questions_only.csv` — placeholder follow-up eval cases for the IRESA taxonomy
- `promptfoo_classifier_provider.py` — provider wrapper used by Promptfoo
- `assertions/` — custom heuristic and readability assertions
- `config/legal_easy_words.txt` — custom easy-word list for readability

## Public Copy Notes

This repo copy intentionally excludes the original sample classification CSVs and eval artifacts.
The tracked IRESA CSVs are safe placeholders so the Promptfoo configs resolve cleanly in a public repo.
Replace those placeholder rows locally with IRESA-specific data before running real evaluations.

## Human Review

Use the Promptfoo web UI for review and rating. This repo no longer generates or maintains local HTML review forms.
