import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from app.services.classification_service import ClassificationService
from app.models.api_models import ClassificationRequest, Label, FollowUpQuestion
from app.providers.base import ClassifierProvider, LLMClassifierProvider


@pytest.mark.asyncio
async def test_get_voted_results_dict_labels():
    # Arrange
    service = ClassificationService(enabled_providers_override=[])
    # Force online code path while still mocking the client and parser.
    service.offline_mode = False
    results_with_providers = [
        ("provider1", {"labels": [{"label": "Label A", "confidence": 0.9}]}),
        ("provider2", {"labels": [Label(label="Label A", confidence=0.8)]}),
    ]
    include_debug_details = False

    # Act
    response = await service._get_voted_results(
        results_with_providers, include_debug_details
    )

    # Assert
    assert len(response.labels) == 1
    assert response.labels[0].label == "Label A"
    assert response.likely_no_legal_problem.value == False


@pytest.mark.asyncio
async def test_likely_no_legal_problem_strong_consensus():
    """Test that likely_no_legal_problem is True when most providers vote 'no legal problem'."""
    # Arrange
    service = ClassificationService(enabled_providers_override=[])
    results_with_providers = [
        ("provider1", {"labels": [], "questions": [], "likely_no_legal_problem": True}),
        ("provider2", {"labels": [], "questions": [], "likely_no_legal_problem": True}),
        (
            "provider3",
            {
                "labels": [{"label": "Family Law"}],
                "questions": [],
                "likely_no_legal_problem": False,
            },
        ),
    ]
    include_debug_details = False

    # Act
    response = await service._get_voted_results(
        results_with_providers, include_debug_details
    )

    # Assert - 2 out of 3 providers voted "no legal problem" (66.7%) which is >= 50%
    assert response.likely_no_legal_problem.value == True


@pytest.mark.asyncio
async def test_likely_no_legal_problem_low_label_agreement():
    """Test that likely_no_legal_problem is True when top label has < 40% of total weight."""
    # Arrange
    service = ClassificationService(enabled_providers_override=[])
    # Each provider has weight 1.0, total = 3.0
    # Label A gets 0.9 (30% of 3.0), Label B gets 0.8 (26.7%), Label C gets 0.7 (23.3%)
    # Top label has < 40%, so low agreement
    results_with_providers = [
        (
            "provider1",
            {
                "labels": [{"label": "Label A", "confidence": 0.9}],
                "questions": [],
                "likely_no_legal_problem": False,
            },
        ),
        (
            "provider2",
            {
                "labels": [{"label": "Label B", "confidence": 0.8}],
                "questions": [],
                "likely_no_legal_problem": False,
            },
        ),
        (
            "provider3",
            {
                "labels": [{"label": "Label C", "confidence": 0.7}],
                "questions": [],
                "likely_no_legal_problem": False,
            },
        ),
    ]
    include_debug_details = False

    # Act
    response = await service._get_voted_results(
        results_with_providers, include_debug_details
    )

    # Assert - top label score (0.9) is 30% of total (3.0), which is < 40%.
    # Disagreement score should reflect low agreement across providers.
    assert response.likely_no_legal_problem.value is False
    assert response.likely_no_legal_problem.disagreement_score == pytest.approx(
        0.7, rel=1e-2
    )


@pytest.mark.asyncio
async def test_likely_no_legal_problem_close_top_two_labels():
    """Test that close top-2 labels under the same top-level category are NOT
    considered high disagreement (e.g., subcategory disagreement should be treated as agreement).
    """
    # Arrange
    service = ClassificationService(enabled_providers_override=[])
    # Two providers return very close scores but both under the same top-level
    # category 'Real Property' (different subcategories). This should NOT be
    # treated as likely_no_legal_problem.
    results_with_providers = [
        (
            "provider1",
            {
                "labels": [
                    {
                        "label": "Real Property > General (residential)",
                        "confidence": 1.0,
                    }
                ],
                "questions": [],
                "likely_no_legal_problem": False,
            },
        ),
        (
            "provider2",
            {
                "labels": [
                    {
                        "label": "Real Property > Tenant (Residential)",
                        "confidence": 0.95,
                    }
                ],
                "questions": [],
                "likely_no_legal_problem": False,
            },
        ),
    ]
    include_debug_details = False

    # Act
    response = await service._get_voted_results(
        results_with_providers, include_debug_details
    )

    # Assert - subcategory disagreement should NOT flip to likely_no_legal_problem
    assert response.likely_no_legal_problem.value == False


@pytest.mark.asyncio
async def test_likely_no_legal_problem_close_top_two_different_top_levels():
    """Test that close top-2 labels from different top-level categories are considered disagreement."""
    # Arrange
    service = ClassificationService(enabled_providers_override=[])
    # Two providers return very close scores but from different top-level
    # categories. This should be treated as low agreement across broad areas.
    results_with_providers = [
        (
            "provider1",
            {
                "labels": [{"label": "Family Law", "confidence": 1.0}],
                "questions": [],
                "likely_no_legal_problem": False,
            },
        ),
        (
            "provider2",
            {
                "labels": [{"label": "Taxation", "confidence": 0.95}],
                "questions": [],
                "likely_no_legal_problem": False,
            },
        ),
    ]
    include_debug_details = False

    # Act
    response = await service._get_voted_results(
        results_with_providers, include_debug_details
    )

    # Assert - different top-level categories close in score should surface disagreement.
    assert response.likely_no_legal_problem.value is False
    assert response.likely_no_legal_problem.disagreement_score == pytest.approx(
        0.5, rel=1e-2
    )


@pytest.mark.asyncio
async def test_likely_no_legal_problem_clear_consensus():
    """Test that likely_no_legal_problem is False when there's clear agreement."""
    # Arrange
    service = ClassificationService(enabled_providers_override=[])
    # All providers agree on same label
    results_with_providers = [
        (
            "provider1",
            {
                "labels": [{"label": "Family Law", "confidence": 1.0}],
                "questions": [],
                "likely_no_legal_problem": False,
            },
        ),
        (
            "provider2",
            {
                "labels": [{"label": "Family Law", "confidence": 1.0}],
                "questions": [],
                "likely_no_legal_problem": False,
            },
        ),
        (
            "provider3",
            {
                "labels": [{"label": "Family Law", "confidence": 1.0}],
                "questions": [],
                "likely_no_legal_problem": False,
            },
        ),
    ]
    include_debug_details = False

    # Act
    response = await service._get_voted_results(
        results_with_providers, include_debug_details
    )

    # Assert - top label score (3.0) is 100% of total (3.0), which is > 40%
    # and there's only one label so no "close" check applies
    assert response.likely_no_legal_problem.value == False
    assert len(response.labels) == 1
    assert response.labels[0].label == "Family Law"


@pytest.mark.asyncio
async def test_semantic_merging():
    # Arrange
    service = ClassificationService(enabled_providers_override=[])

    with patch(
        "app.services.classification_service.ClassificationService._semantically_merge_questions",
        new_callable=AsyncMock,
    ) as mock_merge_questions:
        mock_merge_questions.return_value = [
            FollowUpQuestion(
                question="What is the specific legal issue?", format="text"
            ),
            FollowUpQuestion(
                question="Do you have any documents related to this case?",
                format="boolean",
            ),
        ]

        results_with_providers = [
            (
                "provider1",
                {
                    "labels": [],
                    "questions": [
                        {
                            "question": "What is the specific legal issue?",
                            "type": "text",
                        },
                        {
                            "question": "Could you elaborate on the legal problem?",
                            "type": "text",
                        },
                    ],
                    "likely_no_legal_problem": False,
                },
            ),
            (
                "provider2",
                {
                    "labels": [],
                    "questions": [
                        {
                            "question": "Do you have any documents related to this case?",
                            "type": "boolean",
                        },
                        {
                            "question": "Are there any relevant papers or files?",
                            "type": "boolean",
                        },
                    ],
                    "likely_no_legal_problem": False,
                },
            ),
        ]
        include_debug_details = False

        # Act
        response = await service._get_voted_results(
            results_with_providers, include_debug_details
        )

        # Assert
        assert len(response.follow_up_questions) == 2
        assert (
            response.follow_up_questions[0].question
            == "What is the specific legal issue?"
        )
        assert response.follow_up_questions[0].format == "text"
        assert (
            response.follow_up_questions[1].question
            == "Do you have any documents related to this case?"
        )
        assert response.follow_up_questions[1].format == "boolean"

        mock_merge_questions.assert_called_once()


@pytest.mark.asyncio
async def test_semantic_merge_handles_list_response():
    # Arrange
    service = ClassificationService(enabled_providers_override=[])
    # Force online code path while still mocking the client and parser.
    service.offline_mode = False
    service.openai_client = MagicMock()
    service.openai_client.responses.create = AsyncMock(
        return_value=MagicMock(output_text="[]")
    )

    with patch(
        "app.services.classification_service.parse_json_from_llm_response"
    ) as mock_parse:
        mock_parse.return_value = [
            {"question": "What happened?", "type": "text"},
            {"question": "Do you have documents?", "format": "boolean"},
        ]

        questions = [
            FollowUpQuestion(question="Original question", format="text"),
            FollowUpQuestion(question="Original question 2", format="text"),
        ]

        # Act
        merged = await service._semantically_merge_questions(
            questions,
            request_span=None,
            request_id="test-request",
            taxonomy_name="test-taxonomy",
        )

    # Assert
    assert [q.question for q in merged] == [
        "What happened?",
        "Do you have documents?",
    ]
    assert merged[0].format == "text"
    assert merged[1].format == "boolean"


@pytest.mark.asyncio
async def test_semantic_merge_uses_configured_reasoning_effort():
    # Arrange
    service = ClassificationService(enabled_providers_override=[])
    service.offline_mode = False
    service.openai_client = MagicMock()
    service.openai_client.responses.create = AsyncMock(
        return_value=MagicMock(output_text="[]")
    )

    questions = [FollowUpQuestion(question="Original question", format="text")]

    with (
        patch("app.services.classification_service.OPENAI_SUPPORTS_REASONING_OBJECT", True),
        patch("app.services.classification_service.OPENAI_SUPPORTS_REASONING_EFFORT", False),
        patch("app.services.classification_service.GPT_5_REASONING_EFFORT", "low"),
        patch("app.services.classification_service.parse_json_from_llm_response", return_value=[]),
    ):
        # Act
        await service._semantically_merge_questions(
            questions,
            request_span=None,
            request_id="test-request",
            taxonomy_name="test-taxonomy",
        )

    # Assert
    called_kwargs = service.openai_client.responses.create.call_args.kwargs
    assert called_kwargs["reasoning"]["effort"] == "low"
    assert "minimal" not in str(called_kwargs)


# =============================================================================
# Tests for skip_followups feature
# =============================================================================


@pytest.mark.asyncio
async def test_skip_followups_returns_empty_questions_voted_mode():
    """Test that skip_followups=True returns empty follow_up_questions in voted mode."""
    # Arrange
    service = ClassificationService(enabled_providers_override=[])
    results_with_providers = [
        (
            "provider1",
            {
                "labels": [{"label": "Family Law", "confidence": 0.9}],
                "questions": [
                    {"question": "What is the specific legal issue?", "type": "text"}
                ],
                "likely_no_legal_problem": False,
            },
        ),
        (
            "provider2",
            {
                "labels": [{"label": "Family Law", "confidence": 0.8}],
                "questions": [
                    {"question": "Do you have any documents?", "type": "text"}
                ],
                "likely_no_legal_problem": False,
            },
        ),
    ]

    # Act - with skip_followups=True
    response = await service._get_voted_results(
        results_with_providers, include_debug_details=False, skip_followups=True
    )

    # Assert - follow_up_questions should be empty
    assert response.follow_up_questions == []
    # Labels should still be present
    assert len(response.labels) == 1
    assert response.labels[0].label == "Family Law"


@pytest.mark.asyncio
async def test_skip_followups_returns_empty_questions_first_mode():
    """Test that skip_followups=True returns empty follow_up_questions in first mode."""
    # Arrange
    service = ClassificationService(enabled_providers_override=[])
    results_with_providers = [
        (
            "provider1",
            {
                "labels": [{"label": "Family Law", "confidence": 0.9}],
                "questions": [
                    {"question": "What is the specific legal issue?", "type": "text"}
                ],
                "likely_no_legal_problem": False,
            },
        ),
    ]

    # Act - with skip_followups=True
    response = await service._get_first_result(
        results_with_providers, include_debug_details=False, skip_followups=True
    )

    # Assert - follow_up_questions should be empty
    assert response.follow_up_questions == []
    # Labels should still be present
    assert len(response.labels) == 1
    assert response.labels[0].label == "Family Law"


@pytest.mark.asyncio
async def test_skip_followups_false_still_returns_questions():
    """Test that skip_followups=False (default) still returns follow_up_questions."""
    # Arrange
    service = ClassificationService(enabled_providers_override=[])

    with patch(
        "app.services.classification_service.ClassificationService._semantically_merge_questions",
        new_callable=AsyncMock,
    ) as mock_merge_questions:
        mock_merge_questions.return_value = [
            FollowUpQuestion(
                question="What is the specific legal issue?", format="text"
            ),
        ]

        # Note: We need >3 questions to trigger the semantic merge (optimization skips it for ≤3)
        results_with_providers = [
            (
                "provider1",
                {
                    "labels": [{"label": "Family Law", "confidence": 0.9}],
                    "questions": [
                        {
                            "question": "What is the specific legal issue?",
                            "type": "text",
                        },
                        {"question": "What state are you in?", "type": "text"},
                    ],
                    "likely_no_legal_problem": False,
                },
            ),
            (
                "provider2",
                {
                    "labels": [{"label": "Family Law", "confidence": 0.85}],
                    "questions": [
                        {
                            "question": "Do you have children?",
                            "type": "radio",
                            "options": ["Yes", "No"],
                        },
                        {
                            "question": "Are you married?",
                            "type": "radio",
                            "options": ["Yes", "No"],
                        },
                    ],
                    "likely_no_legal_problem": False,
                },
            ),
        ]

        # Act - with skip_followups=False (default)
        response = await service._get_voted_results(
            results_with_providers, include_debug_details=False, skip_followups=False
        )

        # Assert - follow_up_questions should be present (from mock)
        assert len(response.follow_up_questions) == 1
        assert (
            response.follow_up_questions[0].question
            == "What is the specific legal issue?"
        )
        mock_merge_questions.assert_called_once()


@pytest.mark.asyncio
async def test_skip_followups_skips_semantic_merge():
    """Test that skip_followups=True skips the semantic merge call entirely."""
    # Arrange
    service = ClassificationService(enabled_providers_override=[])

    with patch(
        "app.services.classification_service.ClassificationService._semantically_merge_questions",
        new_callable=AsyncMock,
    ) as mock_merge_questions:
        results_with_providers = [
            (
                "provider1",
                {
                    "labels": [{"label": "Family Law", "confidence": 0.9}],
                    "questions": [
                        {
                            "question": "What is the specific legal issue?",
                            "type": "text",
                        }
                    ],
                    "likely_no_legal_problem": False,
                },
            ),
        ]

        # Act - with skip_followups=True
        response = await service._get_voted_results(
            results_with_providers, include_debug_details=False, skip_followups=True
        )

        # Assert - semantic merge should NOT be called
        mock_merge_questions.assert_not_called()
        assert response.follow_up_questions == []


@pytest.mark.asyncio
async def test_skip_followups_fallback_when_no_labels():
    """Test that skip_followups=True returns empty questions even in fallback case."""
    # Arrange
    service = ClassificationService(enabled_providers_override=[])
    results_with_providers = [
        (
            "provider1",
            {"labels": [], "questions": [], "likely_no_legal_problem": False},
        ),
    ]

    # Act - with skip_followups=True
    response = await service._get_voted_results(
        results_with_providers, include_debug_details=False, skip_followups=True
    )

    # Assert - fallback should return empty questions instead of error message
    assert response.follow_up_questions == []


@pytest.mark.asyncio
async def test_skip_followups_first_mode_fallback():
    """Test that skip_followups=True returns empty questions in first mode fallback."""
    # Arrange
    service = ClassificationService(enabled_providers_override=[])
    # All providers fail
    results_with_providers = [
        ("provider1", Exception("Provider failed")),
        ("provider2", Exception("Provider also failed")),
    ]

    # Act - with skip_followups=True
    response = await service._get_first_result(
        results_with_providers, include_debug_details=False, skip_followups=True
    )

    # Assert - fallback should return empty questions instead of error message
    assert response.follow_up_questions == []
    assert response.labels == []


# ================================================================================
# Tests for skip_semantic_merge optimization
# ================================================================================


@pytest.mark.asyncio
async def test_skip_semantic_merge():
    """Test that skip_semantic_merge=True bypasses the semantic merge LLM call."""
    # Arrange
    service = ClassificationService(enabled_providers_override=[])

    with patch(
        "app.services.classification_service.ClassificationService._semantically_merge_questions",
        new_callable=AsyncMock,
    ) as mock_merge_questions:
        mock_merge_questions.return_value = [
            FollowUpQuestion(question="Merged question", format="text"),
        ]

        # Use more than 3 questions to ensure the skip logic is tested (not the <=3 optimization)
        results_with_providers = [
            (
                "provider1",
                {
                    "labels": [{"label": "Family Law", "confidence": 1.0}],
                    "questions": [
                        {"question": "Question 1", "type": "text"},
                        {"question": "Question 2", "type": "text"},
                        {"question": "Question 3", "type": "text"},
                        {"question": "Question 4", "type": "text"},
                        {"question": "Question 5", "type": "text"},
                    ],
                    "likely_no_legal_problem": False,
                },
            ),
        ]

        # Act - with skip_semantic_merge=True
        response = await service._get_voted_results(
            results_with_providers,
            include_debug_details=False,
            skip_semantic_merge=True,
        )

        # Assert - semantic merge should NOT be called even with >3 questions
        mock_merge_questions.assert_not_called()
        # Questions should be returned without merging (up to 3)
        assert len(response.follow_up_questions) == 3


@pytest.mark.asyncio
async def test_skip_semantic_merge_false_calls_merge():
    """Test that skip_semantic_merge=False (default) calls the semantic merge when >3 questions."""
    # Arrange
    service = ClassificationService(enabled_providers_override=[])

    with patch(
        "app.services.classification_service.ClassificationService._semantically_merge_questions",
        new_callable=AsyncMock,
    ) as mock_merge_questions:
        mock_merge_questions.return_value = [
            FollowUpQuestion(question="Merged question", format="text"),
        ]

        # Use more than 3 questions to trigger semantic merge
        results_with_providers = [
            (
                "provider1",
                {
                    "labels": [{"label": "Family Law", "confidence": 1.0}],
                    "questions": [
                        {"question": "Question 1", "type": "text"},
                        {"question": "Question 2", "type": "text"},
                        {"question": "Question 3", "type": "text"},
                        {"question": "Question 4", "type": "text"},
                    ],
                    "likely_no_legal_problem": False,
                },
            ),
        ]

        # Act - with skip_semantic_merge=False (default)
        response = await service._get_voted_results(
            results_with_providers,
            include_debug_details=False,
            skip_semantic_merge=False,
        )

        # Assert - semantic merge should be called
        mock_merge_questions.assert_called_once()


# ================================================================================
# Tests for prompt caching
# ================================================================================


def test_prompt_caching_different_taxonomies():
    """Test that different taxonomies produce different cached prompts."""
    from app.providers.base import load_prompt, _rendered_prompt_cache

    # Arrange - use simple list-of-dicts taxonomies
    taxonomy1 = [{"Category": "Family Law"}]
    taxonomy2 = [{"Category": "Criminal Law"}]

    # Clear any existing cache
    _rendered_prompt_cache.clear()

    # Act
    prompt1, _ = load_prompt("openai", taxonomy1)
    prompt2, _ = load_prompt("openai", taxonomy2)

    # Assert - different taxonomies should produce different prompts
    assert prompt1 != prompt2
    assert "Family Law" in prompt1
    assert "Criminal Law" in prompt2


@pytest.mark.asyncio
async def test_classify_passes_conversation_id_to_telemetry(monkeypatch):
    service = ClassificationService(enabled_providers_override=[])

    mock_start = MagicMock(return_value=None)
    monkeypatch.setattr(
        "app.services.classification_service.start_request_trace", mock_start
    )

    request = ClassificationRequest(
        text="My important question", conversation_id="conv-123"
    )

    # Act
    await service.classify(request)

    # Assert that start_request_trace was called and metadata contains the conversation_id
    assert mock_start.called
    _, kwargs = mock_start.call_args
    assert kwargs.get("name") == service._truncate_text(
        request.problem_description, limit=200
    )
    assert kwargs.get("metadata", {}).get("conversation_id") == "conv-123"


@pytest.mark.asyncio
async def test_provider_generation_includes_conversation_id(monkeypatch):
    service = ClassificationService(enabled_providers_override=[])

    mock_start_gen = MagicMock(return_value=None)
    monkeypatch.setattr(
        "app.services.classification_service.start_provider_generation", mock_start_gen
    )

    # Create a minimal fake provider
    class FakeProvider:
        instance_name = "fake"
        provider_type = "fake"

        async def classify(self, *args, **kwargs):
            return {"labels": [], "questions": [], "likely_no_legal_problem": False}

    provider = FakeProvider()
    taxonomy_df = None
    final_prompt = "prompt"
    request_span = None
    request_id = "rid-1"
    request = ClassificationRequest(text="Sample", conversation_id="conv-999")

    # Act
    result = await service._classify_with_telemetry(
        provider=provider,
        request=request,
        taxonomy_df=taxonomy_df,
        final_prompt=final_prompt,
        request_span=request_span,
        request_id=request_id,
    )

    # Assert
    assert mock_start_gen.called
    _, kwargs = mock_start_gen.call_args
    input_payload = kwargs.get("input_payload", {})
    metadata = kwargs.get("metadata", {})
    assert input_payload.get("conversation_id") == "conv-999"
    assert metadata.get("conversation_id") == "conv-999"


@pytest.mark.asyncio
async def test_classify_with_telemetry_times_out_slow_provider(monkeypatch):
    service = ClassificationService(enabled_providers_override=[])
    monkeypatch.setattr(
        "app.services.classification_service.CLASSIFIER_TIMEOUT_SECONDS", 0.01
    )

    class SlowProvider:
        instance_name = "slow"
        provider_type = "slow"

        async def classify(self, *args, **kwargs):
            await asyncio.sleep(0.05)
            return {"labels": [], "questions": [], "likely_no_legal_problem": False}

    request = ClassificationRequest(text="Sample timeout request")

    with pytest.raises(TimeoutError, match="timed out after"):
        await service._classify_with_telemetry(
            provider=SlowProvider(),
            request=request,
            taxonomy_df=[],
            final_prompt="prompt",
            request_span=None,
            request_id="rid-timeout",
        )


@pytest.mark.asyncio
async def test_classify_drops_timed_out_provider_from_vote(monkeypatch):
    service = ClassificationService(enabled_providers_override=[])
    monkeypatch.setattr(
        "app.services.classification_service.CLASSIFIER_TIMEOUT_SECONDS", 0.01
    )

    class FastProvider:
        instance_name = "fast"
        provider_type = "fast"
        model_name = "fast-model"

        async def classify(self, *args, **kwargs):
            return {
                "labels": [{"label": "Family Law", "confidence": 1.0}],
                "questions": [],
                "likely_no_legal_problem": False,
            }

    class SlowProvider:
        instance_name = "slow"
        provider_type = "slow"
        model_name = "slow-model"

        async def classify(self, *args, **kwargs):
            await asyncio.sleep(0.05)
            return {
                "labels": [{"label": "Taxation", "confidence": 1.0}],
                "questions": [],
                "likely_no_legal_problem": False,
            }

    service.providers = [FastProvider(), SlowProvider()]

    response = await service.classify(
        ClassificationRequest(
            text="Need legal help",
            include_debug_details=True,
            skip_followups=True,
        )
    )

    assert response.labels
    assert response.labels[0].label == "Family Law"
    assert response.raw_provider_results is not None
    assert "slow" in response.raw_provider_results
    assert "timed out after" in response.raw_provider_results["slow"]["error"]


@pytest.mark.asyncio
async def test_followup_answers_default_to_llm_providers_only():
    service = ClassificationService(enabled_providers_override=[])

    class FakeLLMProvider(LLMClassifierProvider):
        def __init__(self):
            super().__init__(provider_type="openai", model_name="fake-llm")

        def _get_client(self):
            return None

        async def classify(self, *args, **kwargs):
            return {"labels": [], "questions": [], "likely_no_legal_problem": False}

    class FakeNonLLMProvider(ClassifierProvider):
        def __init__(self):
            super().__init__(provider_type="keyword")
            self.instance_name = "fake-keyword"

        async def classify(self, *args, **kwargs):
            return {"labels": [], "questions": [], "likely_no_legal_problem": False}

    service.providers = [FakeLLMProvider(), FakeNonLLMProvider()]

    async def _fake_classify_with_telemetry(*, provider, **kwargs):
        return {
            "labels": [{"label": provider.instance_name, "confidence": 1.0}],
            "questions": [],
            "likely_no_legal_problem": False,
        }

    with patch.object(
        service,
        "_classify_with_telemetry",
        AsyncMock(side_effect=_fake_classify_with_telemetry),
    ) as mock_classify:
        response = await service.classify(
            ClassificationRequest(
                text="Need help",
                followup_answers=[{"question": "Q1", "answer": "A1"}],
                skip_followups=True,
                include_debug_details=True,
            )
        )

    called_provider_names = {
        call.kwargs["provider"].instance_name for call in mock_classify.call_args_list
    }
    assert called_provider_names == {"fake-llm"}
    assert response.labels
    assert response.labels[0].label == "fake-llm"


@pytest.mark.asyncio
async def test_followup_answers_enabled_models_override_allows_non_llm():
    service = ClassificationService(enabled_providers_override=[])

    class FakeLLMProvider(LLMClassifierProvider):
        def __init__(self):
            super().__init__(provider_type="openai", model_name="fake-llm")

        def _get_client(self):
            return None

        async def classify(self, *args, **kwargs):
            return {"labels": [], "questions": [], "likely_no_legal_problem": False}

    class FakeNonLLMProvider(ClassifierProvider):
        def __init__(self):
            super().__init__(provider_type="keyword")
            self.instance_name = "fake-keyword"

        async def classify(self, *args, **kwargs):
            return {"labels": [], "questions": [], "likely_no_legal_problem": False}

    llm_provider = FakeLLMProvider()
    keyword_provider = FakeNonLLMProvider()
    service._all_providers = {
        llm_provider.instance_name: llm_provider,
        keyword_provider.instance_name: keyword_provider,
    }
    service.providers = [llm_provider]

    async def _fake_classify_with_telemetry(*, provider, **kwargs):
        return {
            "labels": [{"label": provider.instance_name, "confidence": 1.0}],
            "questions": [],
            "likely_no_legal_problem": False,
        }

    with patch.object(
        service,
        "_classify_with_telemetry",
        AsyncMock(side_effect=_fake_classify_with_telemetry),
    ) as mock_classify:
        response = await service.classify(
            ClassificationRequest(
                text="Need help",
                followup_answers=[{"question": "Q1", "answer": "A1"}],
                enabled_models=["fake-llm", "fake-keyword"],
                skip_followups=True,
                include_debug_details=True,
            )
        )

    called_provider_names = {
        call.kwargs["provider"].instance_name for call in mock_classify.call_args_list
    }
    assert called_provider_names == {"fake-llm", "fake-keyword"}
    assert response.raw_provider_results is not None
    assert "fake-keyword" in response.raw_provider_results
