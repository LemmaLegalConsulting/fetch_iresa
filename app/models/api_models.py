from pydantic import BaseModel, Field, ConfigDict
from typing import List, Optional, Dict, Any, Union


class Label(BaseModel):
    """A predicted taxonomy label with optional confidence and LIST ID."""

    label: str
    confidence: Optional[float] = None
    id: Optional[str] = Field(
        default=None, description="Optional LIST taxonomy ID (e.g., 'HO-00-00-00-00')"
    )


class FollowUpQuestion(BaseModel):
    """A follow-up question to refine classification."""

    question: str
    format: Optional[str] = None
    options: Optional[List[str]] = None


class FollowUpAnswer(BaseModel):
    """A follow-up question paired with the user's answer."""

    question: str = Field(description="The follow-up question that was asked")
    answer: str = Field(description="The user's answer to the question")


class ClassificationRequest(BaseModel):
    """Request payload for the classification endpoint."""

    model_config = ConfigDict(populate_by_name=True)

    # Accept legacy/short `text` field used by tests/clients while keeping
    # the explicit `problem_description` name internally.
    problem_description: str = Field(..., alias="text")
    conversation_id: Optional[str] = Field(
        default=None,
        description=(
            "Optional conversation identifier supplied by the client to correlate multiple queries "
            "in a multi-turn session. This is included in telemetry metadata and searchable in Langfuse."
        ),
    )
    taxonomy_name: str = Field(
        default="default",
        description="Name of the taxonomy to use (e.g., 'default', 'list')",
    )
    taxonomy_format: str = Field(
        default="osb",
        description=(
            "Output format for classification labels. Options: 'osb' (default OSB taxonomy), "
            "'list' (LIST taxonomy codes). When taxonomy_format='list' and taxonomy_name='default', "
            "results are converted from OSB to LIST codes using a mapping. When taxonomy_name='list', "
            "classification is done directly against the LIST taxonomy."
        ),
    )
    include_debug_details: bool = Field(
        default=False, description="Include full voting details in the response"
    )
    decision_mode: str = Field(
        default="vote",
        description="Decision mode for combining classifier results: 'vote' or 'first'",
    )
    skip_followups: bool = Field(
        default=False,
        description="Skip generating follow-up questions. When True, uses a simplified prompt and returns empty follow_up_questions.",
    )
    enabled_models: Optional[List[str]] = Field(
        default=None,
        description="List of enabled classifier models. If None, all enabled classifiers will be used.",
    )
    skip_semantic_merge: bool = Field(
        default=False,
        description="Skip semantic merging of follow-up questions. Reduces latency but may return duplicate questions.",
    )
    followup_answers: Optional[List["FollowUpAnswer"]] = Field(
        default=None,
        description="Answers to follow-up questions from a previous classification. When provided, refines the classification based on user responses.",
    )


class ClassificationResponse(BaseModel):
    """Response payload with aggregated labels, questions, and consensus analysis."""

    labels: List[Label] = Field(
        description="Top 2 most agreed-upon legal categories from provider aggregation"
    )
    follow_up_questions: List[FollowUpQuestion] = Field(
        description="Up to 3 follow-up questions prioritized by semantic relevance"
    )

    class LikelyNoLegalProblem(BaseModel):
        """Consensus analysis for non-legal problem detection.

        - value: Final decision (True if weighted_result meets threshold)
        - weighted_result: Fraction of total provider weight voting 'no legal problem' (0.0-1.0)
        - disagreement_score: Measure of category disagreement (0.0 = full consensus, 1.0 = max disagreement)
        """

        value: bool = Field(
            description="Whether the problem is likely not legal (based on 30% consensus threshold)"
        )
        weighted_result: float = Field(
            description="Weighted fraction voting for 'no legal problem' (0.0-1.0)"
        )
        disagreement_score: float = Field(
            description="Category disagreement level (0.0-1.0, higher = more disagreement)"
        )

    likely_no_legal_problem: LikelyNoLegalProblem = Field(
        description="Consensus metrics for non-legal problem detection"
    )

    # Debug fields, populated when include_debug_details is True
    raw_provider_results: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Raw results from each classifier provider (debug mode only)",
    )
    weighted_label_scores: Optional[Dict[str, float]] = Field(
        default=None, description="Weighted scores for each label (debug mode only)"
    )
    weighted_question_scores: Optional[Dict[str, float]] = Field(
        default=None, description="Weighted scores for each question (debug mode only)"
    )
