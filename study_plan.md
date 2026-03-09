Study Plan: Follow‑Up Question Quality
======================================

Overview
--------
- Goal: evaluate LLM-generated follow-up questions for completeness, relevance, and efficiency in eliciting missing facts from narratives.
- Approach: “digital twin” fact release—facts are chunked, hidden, and only released when a question is responsive. Tests run via promptfoo with a custom evaluator.
- Outcome: scoring rubric, logged decisions for manual audit, and a repeatable harness to iterate prompts and thresholds.

Scope and Assumptions
---------------------
- Corpus: existing narratives + initial user questions (or primary tasks).
- Scale: ~1,000 runs, ~3 turns each; token budgets remain small (fits gpt-4o-mini comfortably).
- Secrets: use `.env` with keys for chosen provider(s); never commit `.env`.
- Determinism: prefer temperature 0 for evaluator/selector; small temp (0–0.3) for the follow-up agent if you want variety.

Data Preparation
----------------
1) Normalize narratives (strip boilerplate, fix whitespace, preserve order).
2) Sentence segmentation with spaCy sentencizer (parser/NER disabled for speed).
3) Chunking heuristics (per narrative):
   - Merge sentences into ~30–80 token chunks.
   - Do not split inside quotes, numbers, enumerations; keep causal connectors together.
   - Keep Q/A pairs intact if present.
4) Emit TSV/CSV: `data/chunks.tsv` with columns `narrative_id,chunk_id,tokens,text`.
5) Manual QA: spot-check 10–20 narratives; adjust `min_tok/max_tok` if splits feel too fine/coarse.

Chunking Implementation Notes
-----------------------------
- Location: `scripts/chunk_narratives.py`.
- Inputs: JSONL with `{"narrative_id": "...", "text": "..."}` (default) or TSV with `narrative_id<TAB>text` via CLI flag `--tsv`.
- Output: `data/chunks.tsv`.
- CLI example: `python scripts/chunk_narratives.py --input data/narratives.jsonl --output data/chunks.tsv --min-tok 25 --max-tok 80`.
- Dependencies: `spacy` + `en_core_web_sm`; add guard to error if model not downloaded and print install hint.
- Implementation details:
  - Disable parser/ner: `spacy.load("en_core_web_sm", disable=["parser", "ner"])`; add `nlp.add_pipe("sentencizer")`.
  - Token counting: use `len(sent)` to approximate tokens; ok for thresholding.
  - Buffering: accumulate sentences until adding the next would exceed `max_tok` *and* current buffer ≥ `min_tok`; then flush.
  - Chunk ids: stable per narrative: `chunk_id = f"{narrative_id}__{i:04d}"`.
  - Write TSV with header; escape tabs/newlines by replacing with spaces.
  - Log stats: total narratives, total chunks, avg tokens per chunk; print to stdout for sanity.

Digital Twin (Fact Release) Design
----------------------------------
- Hidden store: per narrative, list of `{chunk_id, text}`.
- At each turn:
  1) Take the model’s follow-up question.
  2) Selector LLM sees the question + hidden fact list; returns only ids of responsive facts (or `none`).
  3) Code maps ids → text; append to visible context.
- Guardrails:
  - Selector must never echo fact text; only ids.
  - If nothing is responsive, return `none`.
  - Cap facts shown to selector (e.g., 200–800 tokens) if narratives get long.
  - Log every turn: question, candidate ids considered, ids released.

Digital Twin Implementation Notes
---------------------------------
- Language: JS/TypeScript for promptfoo evaluator convenience; Python equivalent possible if preferred.
- Fact store: preload `data/chunks.tsv` into an in-memory map `{narrative_id: [{id, text}]}` in the evaluator init.
- Selector call: use same provider as tests (e.g., `openai:gpt-4o-mini`); temperature 0.
- Safety: before sending to selector, cap to `max_candidates` (e.g., first 80 ids or top-k if embedding filter is later added) to avoid prompt bloat.
- Parsing: selector returns CSV string or `none`; split on commas, trim, validate against known ids; reject/flag any unknown id or leaked text.
- Released context format: store as list of `{id, text}`; provide to the follow-up agent as concatenated bullet list: `- [id] text`.

Selector Prompt Sketch
----------------------
System: You are a fact-release oracle. You ONLY reveal ids of facts that directly answer or materially support the user’s question. If none apply, respond with `none`. Never reveal fact text.

Facts (hidden):
- [c1] {fact text}
- [c2] {fact text}
- ...

User question: "{question}"

Return: a comma-separated list of ids (e.g., `c2, c5`) or `none`.

Follow-Up Agent Prompt Sketch
-----------------------------
Instruction: Given the initial question and the facts revealed so far, ask one concise follow-up that would most reduce uncertainty. Do not repeat known facts. One question only.

Context so far:
{released_facts_text_or_ids}

Initial user question: {initial_question}

Optional Embedding Hybrid
-------------------------
- Pre-filter facts by cosine similarity within the same `narrative_id` (top-k).
- Only pass top-k texts + ids to the selector LLM to cut tokens and variance.
- Can be skipped for small corpora; add if selector shows drift or cost increases.

Embedding Filter Implementation Notes (optional)
-----------------------------------------------
- Library: `openai` embeddings or local (e.g., `nomic-embed-text` via `sentence-transformers` if offline).
- Precompute: embed all chunks once; persist to `data/chunks_embeddings.npy` + `data/chunks_meta.json` (id → idx mapping).
- Retrieval: given question, embed; filter to same `narrative_id`; compute cosine sim; take top-k (e.g., 15); send only those to selector.
- Guard: if embeddings unavailable, fallback to all-chunk selector with a warning log.

Scoring Heuristic (per turn)
----------------------------
- +1 if the question unlocks at least one new responsive chunk.
- -1 if it targets already-released info (redundant).
- 0 if nothing is unlocked.
- Optional +1 bonus if it unlocks a manually tagged high-priority chunk.
- Aggregate: average per run or sum over turns; keep metadata with released ids.

promptfoo Configuration Plan
----------------------------
- File: `promptfooconfig.yaml`.
- Provider: `openai:gpt-4o-mini` (configurable).
- Tests: one per narrative (or per scenario) with vars:
  - `narrative_id`, `initial_question`.
- Prompt: single follow-up agent prompt consuming `context` and `initial_question`.
- Evaluator: custom JS (e.g., `evals/digitalTwin.js`) that:
  - Receives model output (follow-up question).
  - Calls selector (LLM-only or hybrid).
  - Updates `context` with newly released facts.
  - Computes score and returns metadata (question, released ids).
- Turns: run for fixed N turns (e.g., 3) per test; use promptfoo stateful evaluator or simulate by looping in the evaluator.

Evaluator Skeleton (JS)
-----------------------
- Inputs: `vars` (narrative_id, initial_question), `state.context` (released facts), `response.output` (question).
- Steps:
  - Build selector prompt with hidden facts for that narrative.
  - Call selector LLM; parse ids or `none`.
  - Append new facts to `context`.
  - Compute score (per heuristic).
  - Return `{score, metadata: {...}, state: {context}}`.
- Safeguards: assert selector output contains only known ids or `none`; if not, fail the test to catch leakage.

Evaluator Implementation Notes (JS, promptfoo)
----------------------------------------------
- File: `evals/digitalTwin.js`.
- Exports: `const evaluator = { name: "digitalTwin", async evaluate({ prompt, vars, response, state }) { ... } }; module.exports = evaluator;`
- State handling:
  - Initialize `state.context = state.context ?? []`.
  - Facts map loaded once at module scope from `data/chunks.tsv`.
- Selector call example (OpenAI):
  ```js
  const { OpenAI } = require("openai");
  const client = new OpenAI({ apiKey: process.env.OPENAI_API_KEY });

  async function selectFacts(question, facts) {
    const hidden = facts.map(f => `- [${f.id}] ${f.text}`).join("\n");
    const messages = [
      { role: "system", content: "You are a fact-release oracle. Return ids only." },
      { role: "user", content: `Facts:\n${hidden}\n\nUser question: "${question}"\nReturn comma-separated ids or 'none'.` },
    ];
    const res = await client.chat.completions.create({
      model: "gpt-4o-mini",
      messages,
      temperature: 0,
      max_tokens: 20,
    });
    return res.choices[0].message.content.trim();
  }
  ```
- Scoring calc:
  ```js
  const asked = response.output.trim();
  const selection = await selectFacts(asked, candidateFacts);
  const ids = selection.toLowerCase() === "none" ? [] : selection.split(",").map(s => s.trim()).filter(Boolean);
  const newFacts = candidateFacts.filter(f => ids.includes(f.id) && !state.context.some(c => c.id === f.id));
  const redundant = ids.some(id => state.context.some(c => c.id === id));
  const score = (newFacts.length ? 1 : 0) - (redundant ? 1 : 0);
  const context = [...state.context, ...newFacts];
  return { score, state: { context }, metadata: { asked, released: newFacts.map(f => f.id), raw_selector: selection } };
  ```
- Validation:
  - If `ids` contains unknown ids, throw an error to fail the test (catches leakage or parsing errors).
  - If selector response length > 50 chars, fail to catch text leakage.
- Wiring in `promptfooconfig.yaml`:
  ```yaml
  description: "Follow-up question quality"
  providers:
    - id: openai:gpt-4o-mini
      config: { apiKey: env:OPENAI_API_KEY }
  prompts:
    - label: followup-agent
      prompt: |
        You ask concise follow-up questions to reduce uncertainty.
        Known facts:
        {{#each context}}
        - [{{this.id}}] {{this.text}}
        {{/each}}
        Initial user question: {{initial_question}}
        Ask one follow-up question. One sentence.
  tests:
    - vars:
        narrative_id: "narrative_001"
        initial_question: "How did the outage start?"
  evaluators:
    - path: ./evals/digitalTwin.js
  turns: 3
  ```

Manual Audit Plan
-----------------
- Before full run: dry-run 10 narratives × 3 turns; inspect logs for:
  - Selector leakage (text in ids), redundant releases, missed obvious facts.
  - Follow-up questions that ignore missing high-priority facts.
- Maintain a small golden set: map sample questions → correct chunk ids; run as regression to catch prompt drift.

Cost Check (baseline)
---------------------
- Assumptions: gpt-4o-mini, ~800 hidden-fact tokens/turn, 3 turns/run.
- Estimated: ~\$0.0006 per run → ~\$0.60 for 1,000 runs. gpt-4o ≈ \$19 for the same setup.

Risks and Mitigations
---------------------
- Leakage (selector reveals text): keep ids-only protocol; assert outputs.
- Variance (LLM picks wrong facts): set temperature 0; optionally add embedding pre-filter or a second-pass judge.
- Over-chunking/under-chunking: adjust `min_tok/max_tok` after QA; re-run chunker.
- Redundancy: scoring penalty and prompt instruction to avoid repeating known facts.

Next Actions (execution order)
------------------------------
1) Write chunker script (`scripts/chunk_narratives.py`) and generate `data/chunks.tsv`; spot-check.
2) Draft selector and follow-up agent prompts; bake into evaluator.
3) Implement `evals/digitalTwin.js` evaluator and `promptfooconfig.yaml` with 3-turn runs.
4) Dry-run 10 cases; inspect logs; tune thresholds (k for pre-filter, scoring weights).
5) Scale to full set; export scores and logs for manual review.
