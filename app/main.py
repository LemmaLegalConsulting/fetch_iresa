import os

if os.getenv("ENV") != "production":
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except ImportError:
        pass

from fastapi import FastAPI, Depends, HTTPException, status
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
    - Supports OSB and LIST taxonomies
    - Provides consensus scoring for "no legal problem" detection
    - Includes disagreement metrics for category uncertainty

    ## Authentication
    - Dev mode: No auth required (set ENV=dev)
    - Production: Bearer token required (set API_TOKENS env var)

    ## Endpoints
    - POST /api/classify: Classify a problem description
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
