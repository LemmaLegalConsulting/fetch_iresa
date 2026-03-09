#!/usr/bin/env python3
"""Check tests that have no follow-up questions and verify assertions pass.

This uses cached provider outputs when possible (ClassificationService with cache_enabled=True).
Run with repo venv activated: `source .venv/bin/activate && python scripts/check_no_followup_failures.py`.
"""
from __future__ import annotations

import csv
import os
import asyncio
from typing import List

import sys, os
# Ensure repo root is on sys.path so `import app` works when running scripts
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from app.services.classification_service import ClassificationService
from app.models.api_models import ClassificationRequest
from promptfoo.assertions.followup_heuristics import sentence_length_ok
from promptfoo.assertions.followup_textstat_grade import get_assert, get_fkgl_assert

CSV_PATH = "promptfoo/iresa_followup_questions_only.csv"

async def main():
    svc = ClassificationService(cache_enabled=True, cache_dir='./cache')

    no_followup_cases: List[str] = []
    failures = []

    with open(CSV_PATH, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        rows = list(reader)

    for i, row in enumerate(rows):
        problem = row.get('problem_description') or row.get('prompt') or ''
        if not problem:
            continue
        req = ClassificationRequest(problem_description=problem, taxonomy_name='iresa')
        resp = await svc.classify(req)

        # Normalize followups
        fus = resp.follow_up_questions
        has_followups = bool(fus)
        has_nonempty_question = False
        first_q_text = None
        if has_followups:
            for q in fus:
                q_text = getattr(q, 'question', '')
                if isinstance(q_text, str) and q_text.strip():
                    has_nonempty_question = True
                    break
            first_q_text = (fus[0].question if len(fus) > 0 else '')

        if not has_followups or not has_nonempty_question:
            # Count as no-valid-followup (either none or empty questions)
            no_followup_cases.append({'idx': i, 'snippet': problem[:120], 'first_q': first_q_text})
            # Build raw output JSON like provider
            labels_as_dicts = [l.model_dump() for l in resp.labels]
            follow_up_questions_as_dicts = [q.model_dump() for q in resp.follow_up_questions]
            output = {
                'labels': labels_as_dicts,
                'follow_up_questions': follow_up_questions_as_dicts,
            }
            import json
            raw_output = json.dumps(output)

            # Run assertions
            sl = sentence_length_ok(raw_output, {})
            dc = get_assert(raw_output, {})
            fk = get_fkgl_assert(raw_output, {})

            if not (sl.get('pass') and dc.get('pass') and fk.get('pass')):
                failures.append({'idx': i, 'problem': problem[:120], 'first_q': first_q_text, 'sentence_length': sl, 'dale_chall': dc, 'fkgl': fk})

    print(f'Total cases checked: {len(rows)}')
    print(f'No-followup cases: {len(no_followup_cases)}')
    print(f'Failures among no-followup cases: {len(failures)}')
    if failures:
        for f in failures[:10]:
            print('\n---')
            print('idx:', f['idx'])
            print('problem snippet:', f['problem'])
            print('sentence_length:', f['sentence_length'])
            print('dale_chall:', f['dale_chall'])
            print('fkgl:', f['fkgl'])

if __name__ == '__main__':
    asyncio.run(main())
