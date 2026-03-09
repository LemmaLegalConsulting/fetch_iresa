# Legal Classification Service

A FastAPI-based multi-provider legal problem classification API that aggregates results from multiple LLM providers (OpenAI, Gemini, Mistral) and specialized classifiers (SPOT, keyword-based) using weighted voting.

## Features

- **Multi-provider classification**: Combines results from multiple AI providers for better accuracy
- **Weighted voting**: Configurable weights for each provider based on empirical performance
- **Taxonomy-based**: Classifies problems into a configurable legal taxonomy
- **Follow-up questions**: Generates clarifying questions with semantic deduplication
- **Caching**: Optional disk-based caching for provider responses
- **Authentication**: Bearer token-based API authentication (bypassable in dev mode)

## Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

For development (tests, linting, etc.) install the development dependencies:

```bash
pip install -r requirements-dev.txt
```

### 2. Configure environment variables

Copy `.env.example` to `.env` and configure your API keys:

```bash
cp .env.example .env
```

Required environment variables:

| Variable | Description |
|----------|-------------|
| `OPENAI_API_KEY` | OpenAI API key |
| `GEMINI_API_KEY` | Google Gemini API key |
| `OPENROUTER_API_KEY` | OpenRouter API key (for Mistral) |
| `SPOT_API_KEY` | SPOT taxonomy API key |
| `API_TOKENS` | Comma-separated list of valid API tokens for authentication |
| `ENV` | Set to `dev` to bypass authentication and serve the demo UI |

### 3. Place taxonomy files

Place your taxonomy CSV files in `app/data/`.

- Default taxonomy: `app/data/taxonomy.csv`
- IRESA taxonomy placeholder: `app/data/taxonomy_iresa.csv`

This public copy intentionally omits the original private sample eval CSVs. Replace the tracked IRESA placeholder files in `app/data/` and `promptfoo/` with your local IRESA data before running evals.

## Running the Application

### Development mode (with hot reload and auth bypass)

```bash
ENV=dev uvicorn app.main:app --reload
```

### Production mode

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8080
```

### Docker

```bash
docker build -t fetch .
docker run -p 8080:8080 --env-file .env fetch
```

## API Reference

### POST `/api/classify`

Classify a legal problem description.

#### Request Headers

```
Authorization: Bearer <your_api_token>
```

#### Request Body

```json
{
  "text": "My landlord refuses to return my security deposit after I moved out.",
  "taxonomy_name": "default",
  "decision_mode": "vote",
  "enabled_models": ["gemini", "gpt-5.2", "keyword", "spot"],
  "include_debug_details": false,
  "skip_semantic_merge": false
}
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `text` | string | *required* | The legal problem description to classify |
| `taxonomy_name` | string | `"default"` | Name of the taxonomy to use (`default`, `iresa`, or `list`) |
| `decision_mode` | string | `"vote"` | `"vote"` for weighted voting, `"first"` for first successful result |
| `enabled_models` | array | `null` | List of provider names to use. If null, uses configured defaults |
| `include_debug_details` | boolean | `false` | Include raw provider results and weighted scores |
| `skip_semantic_merge` | boolean | `false` | Skip LLM-based deduplication of follow-up questions (reduces latency) |
| `conversation_id` | string | `null` | Optional client-supplied identifier to correlate multiple queries in the same conversation. Included in telemetry metadata and searchable in Langfuse. |

#### Response

```json
{
  "labels": [
    {"label": "Real Property > Tenant (Residential)", "confidence": 2.45},
    {"label": "Consumer Law > Debt Collection", "confidence": 1.2}
  ],
  "follow_up_questions": [
    {
      "question": "How long ago did you move out?",
      "format": "text",
      "options": null
    },
    {
      "question": "Did your landlord provide a written reason for withholding the deposit?",
      "format": "radio",
      "options": ["Yes", "No", "Not sure"]
    }
  ],
  "likely_no_legal_problem": {
    "value": false,
    "vote_weight": 0.0,
    "total_weight": 3.25,
    "pct": 0.0,
    "threshold": 1.625,
    "high_disagreement": false
  }
}
```

## Telemetry & Langfuse 🔍

- You can supply an optional `conversation_id` in the request body to correlate multiple queries that belong to the same user session. This value is attached to the request span and provider generations as metadata.
- In the Langfuse web UI you can search or filter by that metadata (for example: `conversation_id:conv-123` or `metadata.conversation_id:conv-123`) to find all related traces and generations quickly.
- The API also uses the (truncated) user `text` as the request span name so you can scan recent queries in the Langfuse timeline without expanding each trace.

## Available Providers

| Provider | Instance Name | Description |
|----------|---------------|-------------|
| OpenAI GPT-4.1 Mini | `gpt-4.1-mini` | OpenAI's GPT-4.1-mini model |
| OpenAI GPT-4.1 Nano | `gpt-4.1-nano` | OpenAI's GPT-4.1-nano model |
| OpenAI GPT-5 | `gpt-5` | OpenAI's GPT-5 model |
| OpenAI GPT-5.2 (Azure) | `gpt-5.2` | Azure OpenAI GPT-5.2 deployment |
| Gemini | `gemini` | Google Gemini 2.5 Flash |
| Mistral | `mistral` | Mistral Small via OpenRouter |
| SPOT | `spot` | Suffolk LIT Lab SPOT taxonomy API |
| Keyword | `keyword` | Simple keyword-based classifier |

## Configuration

Provider weights and enabled classifiers can be configured in `app/core/config.py`:

```python
CLASSIFIER_WEIGHTS = {
    "gemini": 0.9,
    "gpt-4.1-mini": 0.8,
    "gpt-5.2": 0.8,
    "spot": 0.75,
    "keyword": 0.5,
}

ENABLED_CLASSIFIERS = ["gemini", "mistral", "keyword", "spot", "gpt-5.2"]
```

Taxonomy files are mapped by name in `TAXONOMY_MAPPING`, including the IRESA placeholder:

```python
TAXONOMY_MAPPING = {
    "default": "app/data/taxonomy.csv",
    "iresa": "app/data/taxonomy_iresa.csv",
    "list": "app/data/list-taxonomy.csv",
}
```

## Testing

Make sure the development dependencies are installed:

```bash
pip install -r requirements-dev.txt
```

Run tests:

```bash
pytest -q
```

Run a specific test:

```bash
pytest tests/test_api_auth.py::test_classify_valid_token
```

## Project Structure

```
app/
├── main.py                 # FastAPI entrypoint
├── core/
│   ├── config.py           # Weights, enabled classifiers, taxonomy mapping
│   └── cache.py            # Disk cache helpers
├── data/                   # Taxonomy CSV files
├── models/
│   └── api_models.py       # Pydantic request/response schemas
├── prompts/                # Prompt templates
├── providers/
│   ├── base.py             # Abstract provider class
│   ├── openai.py           # OpenAI provider
│   ├── gemini.py           # Gemini provider
│   ├── mistral.py          # Mistral provider
│   ├── spot.py             # SPOT API provider
│   └── keyword.py          # Keyword-based classifier
├── services/
│   └── classification_service.py  # Orchestration and voting
└── utils/
    ├── backoff.py          # Retry logic with exponential backoff
    └── logging.py          # Logging utilities
```

## License

See LICENSE file.
