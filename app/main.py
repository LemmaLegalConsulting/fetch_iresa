import os
import json
import html

if os.getenv("ENV") != "production":
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except ImportError:
        pass

from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.responses import HTMLResponse
from typing import Optional
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from app.models.api_models import ClassificationRequest, ClassificationResponse
from app.services.classification_service import ClassificationService
from app.utils.logging import get_logger

app = FastAPI(
    title="Legal Problem Classification API",
    description="""
    API for classifying legal problem descriptions using multiple AI providers.

    ## Features
    - Aggregates results from keyword, LLM, and taxonomy-based classifiers
    - Uses the German IRESA social-law taxonomy
    - Provides consensus scoring for "no legal problem" detection
    - Includes disagreement metrics for category uncertainty

    ## Authentication
    - Dev mode: No auth required (set ENV=dev)
    - Production: Bearer token required (set API_TOKENS env var)

    ## Endpoints
    - POST /api/classify: Classify a problem description
    - GET /test: Public interactive test page (no auth)
    """,
    version="1.0.0",
)

classification_service = ClassificationService()

logger = get_logger(__name__)

security = HTTPBearer(auto_error=False)


def verify_token(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),
):
    """Validate a Bearer token from the `Authorization` header.

    In dev mode (ENV=dev), auth is bypassed.
    In production, a valid Bearer token is required.

    Args:
      credentials: Injected by FastAPI's `HTTPBearer` security dependency.

    Returns:
      The validated token string, or None if in dev mode.

    Raises:
      HTTPException: If not in dev mode and token is invalid.
    """
    # In dev mode, bypass authentication
    if os.getenv("ENV") == "dev":
        return None

    # In production, require valid token
    if credentials is None:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not authenticated",
        )

    token = credentials.credentials
    valid_tokens_str = os.getenv("API_TOKENS")
    if not valid_tokens_str:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="API_TOKENS environment variable not set.",
        )
    valid_tokens = [t.strip() for t in valid_tokens_str.split(",")]
    if token not in valid_tokens:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return token


@app.post("/api/classify", response_model=ClassificationResponse)
async def classify(payload: ClassificationRequest, token: str = Depends(verify_token)):
    """Classify a legal problem description and return relevant categories and questions.

    This endpoint analyzes the provided problem description using multiple classifier providers
    (e.g., keyword matching, LLMs like Gemini/Mistral, and taxonomy APIs). Results are aggregated
    with weighted voting to determine the most relevant legal categories and follow-up questions.

    The response includes:
    - **labels**: Top legal categories with confidence scores
    - **follow_up_questions**: Clarifying questions to refine classification
    - **likely_no_legal_problem**: Consensus analysis with weighted result and disagreement score

    Args:
      payload: Request containing problem description, taxonomy options, and debug settings.
      token: Bearer token for authentication (bypassed in dev mode).

    Returns:
      ClassificationResponse with aggregated results from all enabled providers.

    Raises:
      HTTPException: For authentication failures or invalid requests.
    """
    # Token is already validated by the verify_token dependency above
    return await classification_service.classify(
        payload, enabled_models=payload.enabled_models
    )


@app.get("/test", response_class=HTMLResponse)
async def test_page(text: Optional[str] = None):
    """Public interactive test UI (no auth) for quick end-user checks."""
    labels_html = ""
    questions_html = ""
    raw_json = ""
    error_html = ""

    if text and text.strip():
        try:
            response = await classification_service.classify(
                ClassificationRequest(
                    text=text.strip(),
                    taxonomy_name="iresa",
                )
            )
            if response.labels:
                items = "".join(
                    f"<li><strong>{html.escape(label.label)}</strong>"
                    f" (confidence: {label.confidence:.3f})</li>"
                    if label.confidence is not None
                    else f"<li><strong>{html.escape(label.label)}</strong></li>"
                    for label in response.labels
                )
                labels_html = f"<ul>{items}</ul>"
            else:
                labels_html = "<p>No labels returned.</p>"

            if response.follow_up_questions:
                items = "".join(
                    f"<li>{html.escape(q.question)}</li>"
                    for q in response.follow_up_questions
                )
                questions_html = f"<ol>{items}</ol>"
            else:
                questions_html = "<p>No follow-up questions returned.</p>"

            raw_json = html.escape(
                json.dumps(response.model_dump(), ensure_ascii=False, indent=2)
            )
        except Exception as exc:
            error_html = f"<p><strong>Error:</strong> {html.escape(str(exc))}</p>"

    safe_text = html.escape(text or "")
    return f"""
<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>IRESA Classifier Test</title>
    <style>
      body {{ font-family: sans-serif; max-width: 840px; margin: 2rem auto; padding: 0 1rem; }}
      textarea {{ width: 100%; min-height: 120px; }}
      button {{ margin-top: 0.5rem; padding: 0.5rem 0.9rem; }}
      pre {{ background: #f5f5f5; padding: 1rem; overflow: auto; }}
      .muted {{ color: #666; font-size: 0.95rem; }}
      .thinking-backdrop {{
        position: fixed;
        inset: 0;
        background: rgba(0, 0, 0, 0.35);
        display: none;
        align-items: center;
        justify-content: center;
        z-index: 9999;
      }}
      .thinking-box {{
        background: #fff;
        border-radius: 8px;
        padding: 1rem 1.25rem;
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.2);
      }}
    </style>
  </head>
  <body>
    <div id="thinking-modal" class="thinking-backdrop" aria-live="polite" aria-busy="true">
      <div class="thinking-box">Thinking...</div>
    </div>
    <h1>IRESA Classifier Test</h1>
    <p class="muted">This page is unauthenticated and runs classification directly on the server.</p>
    <form id="classify-form" method="get" action="/test">
      <label for="text">Problem description</label>
      <textarea id="text" name="text" placeholder="Beschreibe dein sozialrechtliches Problem...">{safe_text}</textarea>
      <br />
      <button type="submit">Classify</button>
    </form>
    {error_html}
    <h2>Labels</h2>
    {labels_html or "<p>Submit text to classify.</p>"}
    <h2>Follow-up Questions</h2>
    {questions_html or "<p>Submit text to generate questions.</p>"}
    <h2>Raw Response</h2>
    <pre>{raw_json or "No result yet."}</pre>
    <script>
      const form = document.getElementById("classify-form");
      const modal = document.getElementById("thinking-modal");
      if (form && modal) {{
        form.addEventListener("submit", () => {{
          modal.style.display = "flex";
        }});
      }}
    </script>
  </body>
</html>
"""
