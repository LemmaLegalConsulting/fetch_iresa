import os
import time
from uuid import uuid4
from openai import AsyncAzureOpenAI, AsyncOpenAI
import json
from app.models.api_models import (
    ClassificationRequest,
    ClassificationResponse,
    Label,
    FollowUpQuestion,
    FollowUpAnswer,
)
from app.providers.base import ClassifierProvider, LLMClassifierProvider, load_prompt
from app.providers.openai import OpenAIProvider
from app.providers.gemini import GeminiProvider
from app.providers.mistral import MistralProvider
from app.providers.keyword import KeywordClassifierProvider
from app.providers.spot import SpotProvider
from app.providers.openai import GPT41NanoProvider
from app.providers.openai import GPT5Provider
from app.providers.openai import GPT52Provider
from app.providers.utils import extract_usage_details
from app.utils.json_helpers import parse_json_from_llm_response
from app.core.config import (
    CLASSIFIER_WEIGHTS,
    ENABLED_CLASSIFIERS,
    TAXONOMY_MAPPING,
    DECISION_MODE,
    GPT_5_REASONING_EFFORT,
    CLASSIFIER_TIMEOUT_SECONDS,
    OPENAI_SUPPORTS_REASONING_OBJECT,
    OPENAI_SUPPORTS_REASONING_EFFORT,
    OPENAI_SDK_VERSION,
)
from app.data.list_taxonomy import (
    load_list_taxonomy_simple,
    lookup_list_code_from_title,
    convert_osb_label_to_list,
    format_list_label,
    get_list_code_to_title_mapping,
)
from app.utils.csv_helpers import read_csv_as_list_of_dicts, dedupe_and_clean_rows
import asyncio
from collections import defaultdict
from typing import List, Dict, Union, Optional, Tuple, Any
from diskcache import Cache
from app.utils.logging import get_logger
import inspect
from threading import Lock
from app.telemetry import (
    finalize_provider_generation,
    finalize_request_trace,
    flush_telemetry_async,
    start_provider_generation,
    start_request_trace,
)


def _env_flag_enabled(name: str) -> bool:
    """Return True when the env var is set to a truthy value."""
    return os.getenv(name, "").strip().lower() in {"1", "true", "yes", "on"}


logger = get_logger(__name__)


class ClassificationService:
    """Coordinates multiple classifier providers and aggregates results."""

    def __init__(
        self,
        enabled_providers_override: Optional[List[str]] = None,
        cache_enabled: bool = False,
        cache_dir: str = "./cache",
    ):
        """Create a new service instance.

        Args:
          enabled_providers_override: Optional list of provider instance names to enable.
          cache_enabled: Whether to cache provider responses to disk.
          cache_dir: Path to the directory for caching.
        """
        # Offline mode disables external providers and semantic merge (network calls)
        self.offline_mode = _env_flag_enabled("OFFLINE_MODE") or _env_flag_enabled(
            "DISABLE_EXTERNAL_PROVIDERS"
        )
        # Cache of all initialized provider instances (keyed by instance_name)
        self._all_providers: Dict[str, ClassifierProvider] = {}
        self.providers = self._init_providers(
            enabled_providers_override=enabled_providers_override
        )
        # Lock to protect taxonomy cache updates if we ever load on-miss at runtime
        self._tax_lock = Lock()
        # Preload all taxonomies to reduce per-request latency
        self.taxonomies = self._load_all_taxonomies()
        # Initialize OpenAI client for semantic merging (required dependency)
        self.openai_client = None
        if not self.offline_mode:
            azure_endpoint = os.environ.get("OPENAI_BASE_URL_GPT_5_2")
            azure_api_key = os.environ.get("OPENAI_GPT_5_2_API_KEY")
            azure_api_version = os.environ.get("OPENAI_GPT_5_2_API_VERSION")
            if azure_endpoint and azure_api_key and azure_api_version:
                self.openai_client = AsyncAzureOpenAI(
                    api_key=azure_api_key,
                    api_version=azure_api_version,
                    azure_endpoint=azure_endpoint,
                )
            else:
                self.openai_client = AsyncOpenAI(
                    api_key=os.environ.get("OPENAI_API_KEY")
                )
        self.cache_enabled = cache_enabled
        self._client_closed = False
        if self.cache_enabled:
            cache_path = os.path.join(cache_dir, "provider_responses_cache")
            logger.info(f"Initializing cache at: {os.path.abspath(cache_path)}")
            if os.path.exists(cache_path):
                logger.info(
                    f"Cache directory exists. Contents: {os.listdir(cache_path)}"
                )
            else:
                logger.info("Cache directory does not exist.")
            self.cache = Cache(cache_path)
            logger.info(f"Caching enabled. Cache directory: {self.cache.directory}")
        else:
            self.cache = {}
            logger.info("Caching disabled.")

    async def cleanup(self) -> None:
        """Cleanup async resources, particularly the OpenAI client.
        
        This should be called when the service is no longer needed to ensure
        proper cleanup of HTTP connections before the event loop closes.
        """
        if not self._client_closed and self.openai_client is not None:
            try:
                await self.openai_client.close()
                self._client_closed = True
                logger.debug("OpenAI client closed successfully")
            except RuntimeError as e:
                if "Event loop is closed" not in str(e):
                    logger.debug(f"Error closing OpenAI client: {e}")
            except Exception as e:
                logger.debug(f"Error closing OpenAI client: {e}")

        # Close provider clients to prevent httpx/anyio cleanup after loop shutdown.
        seen: set[int] = set()
        for provider in list(self._all_providers.values()):
            if id(provider) in seen:
                continue
            seen.add(id(provider))
            try:
                await provider.aclose()
            except RuntimeError as e:
                if "Event loop is closed" not in str(e):
                    logger.debug(
                        f"Error closing provider {provider.instance_name}: {e}"
                    )
            except Exception as e:
                logger.debug(f"Error closing provider {provider.instance_name}: {e}")

    def _load_all_taxonomies(self) -> Dict[str, List[Dict[str, Any]]]:
        """Load all taxonomies from the mapping into memory as lists of dicts."""
        taxonomies = {}
        for name, file_path in TAXONOMY_MAPPING.items():
            try:
                rows = read_csv_as_list_of_dicts(file_path)
                rows = dedupe_and_clean_rows(rows)
                taxonomies[name] = rows
                logger.info(f"Successfully loaded taxonomy '{name}' from {file_path}")
            except FileNotFoundError:
                logger.warning(f"Taxonomy file not found for '{name}': {file_path}")
            except Exception as e:
                logger.error(
                    f"Error loading taxonomy file for '{name}' at {file_path}: {e}"
                )
        return taxonomies

    def _init_providers(
        self, enabled_providers_override: Optional[List[str]] = None
    ) -> List[ClassifierProvider]:
        """Instantiate and filter providers based on config or override.

        Uses cached provider instances when available to avoid re-initialization overhead.

        Args:
          enabled_providers_override: Restrict enabled providers to these instance names.

        Returns:
          List of enabled `ClassifierProvider` instances.
        """
        if self.offline_mode:
            if "keyword" not in self._all_providers:
                try:
                    self._all_providers["keyword"] = KeywordClassifierProvider()
                except Exception as e:
                    logger.error(
                        f"Could not initialize KeywordClassifierProvider in offline mode: {e}"
                    )
                    return []
            return [self._all_providers["keyword"]]

        # Initialize all providers once and cache them
        if not self._all_providers:
            provider_classes = [
                ("openai", OpenAIProvider),
                ("gemini", GeminiProvider),
                ("mistral", MistralProvider),
                ("keyword", KeywordClassifierProvider),
                ("spot", SpotProvider),
                ("gpt-41-nano", GPT41NanoProvider),
                ("gpt-5", GPT5Provider),
                ("gpt-5.2", GPT52Provider),
            ]
            for name, provider_class in provider_classes:
                try:
                    provider = provider_class()
                    # Cache by configured name (e.g., "openai") so ENABLED_CLASSIFIERS lookups work
                    self._all_providers[name] = provider
                    # Also cache by instance_name when it differs, for backward compatibility
                    if provider.instance_name != name:
                        self._all_providers[provider.instance_name] = provider
                except Exception as e:
                    logger.error(f"Could not initialize {provider_class.__name__}: {e}")

        if enabled_providers_override:
            enabled_classifier_names = enabled_providers_override
        else:
            enabled_classifier_names = ENABLED_CLASSIFIERS

        # Filter cached providers based on enabled list
        enabled_providers = [
            self._all_providers[name]
            for name in enabled_classifier_names
            if name in self._all_providers
        ]
        return enabled_providers

    async def _semantically_merge_questions(
        self,
        questions: List[FollowUpQuestion],
        *,
        request_span,
        request_id: Optional[str],
        taxonomy_name: Optional[str],
        conversation_id: Optional[str] = None,
    ) -> List[FollowUpQuestion]:
        """Merge similar follow-up questions using an LLM call.

        Args:
          questions: A list of candidate follow-up questions.

        Returns:
          A deduplicated list of merged questions.
        """
        if self.offline_mode:
            return questions

        if not questions:
            return []

        questions_data = [q.model_dump() for q in questions]
        questions_json = json.dumps(questions_data, indent=2)

        final_prompt, _ = load_prompt("semantic_merge", [])

        model_name = os.getenv("OPENAI_GPT_5_2_MODEL", "gpt-5.2")
        generation = start_provider_generation(
            request_span,
            name="semantic_merge",
            model=model_name,
            input_payload={
                "questions_preview": questions_data[:5],
                "question_count": len(questions_data),
                "taxonomy": taxonomy_name,
                "conversation_id": conversation_id,
            },
            metadata={
                "provider": "semantic_merge",
                "request_id": request_id,
                "taxonomy": taxonomy_name,
                **({"conversation_id": conversation_id} if conversation_id else {}),
            },
        )

        start_time = time.perf_counter()
        response = None
        parsed_response: Any = None
        telemetry_error: Optional[BaseException] = None

        try:
            # Use Responses API for semantic merge so we can pass reasoning effort.
            # The Responses API accepts reasoning parameter (nested object in SDK v2.x).
            params: Dict[str, Any] = {
                "model": model_name,
                "input": f"SYSTEM:\n{final_prompt}\n\nUSER:\n{questions_json}",
            }

            # Add reasoning parameter based on SDK support.
            # Use configured effort level so this stays valid as model support evolves.
            if GPT_5_REASONING_EFFORT is not None:
                if OPENAI_SUPPORTS_REASONING_OBJECT:
                    # SDK v2.x expects nested reasoning object
                    params["reasoning"] = {"effort": GPT_5_REASONING_EFFORT}
                elif OPENAI_SUPPORTS_REASONING_EFFORT:
                    # Older SDK might support flat reasoning_effort parameter
                    params["reasoning_effort"] = GPT_5_REASONING_EFFORT

            logger.info(
                f"[ClassificationService] semantic_merge calling responses.create with reasoning={'object' if OPENAI_SUPPORTS_REASONING_OBJECT else 'effort' if OPENAI_SUPPORTS_REASONING_EFFORT else 'omitted'}"
            )
            response = await self.openai_client.responses.create(**params)

            # Extract text from Responses API response.
            # The Responses API returns the actual output text in response.output_text,
            # not in response.text (which is the ResponseTextConfig specification).
            content = None
            if hasattr(response, "output_text") and response.output_text:
                content = response.output_text
            elif hasattr(response, "output") and response.output:
                # Fallback: iterate over output list and extract text
                for item in response.output:
                    if hasattr(item, "text"):
                        content = item.text
                        break

            if not content or not str(content).strip():
                logger.error(
                    f"[ClassificationService] semantic_merge received empty content from Responses API: {response}"
                )
                return questions

            content = str(content).strip()
            logger.info(
                f"[ClassificationService] semantic_merge content preview: {content[:200]}"
            )
            # Use robust JSON parsing that handles fenced code blocks and common LLM JSON errors
            parsed_response = parse_json_from_llm_response(content)

            if isinstance(parsed_response, list):
                merged_questions_data = parsed_response
            elif isinstance(parsed_response, dict):
                merged_questions_data = parsed_response.get("merged_questions", [])
            else:
                merged_questions_data = []

            merged_questions = []
            for q_data in merged_questions_data:
                if not isinstance(q_data, dict):
                    logger.warning(
                        f"Merged question item is not a dict, skipping: {q_data}"
                    )
                    continue

                if "question" in q_data:
                    # Normalize keys from the merge output: many prompts/models
                    # use 'type' while our API model expects 'format'.
                    if "format" not in q_data and "type" in q_data:
                        q_data["format"] = q_data.pop("type")

                    # If options are present but no format was provided, default to 'radio'
                    # to ensure UI widgets render correctly.
                    if q_data.get("format") in (None, "") and q_data.get("options"):
                        q_data["format"] = "radio"

                    try:
                        merged_questions.append(FollowUpQuestion(**q_data))
                    except Exception as e:
                        logger.warning(
                            f"Failed to parse merged question, falling back to text-only. Data: {q_data}. Error: {e}"
                        )
                        merged_questions.append(
                            FollowUpQuestion(
                                question=str(q_data.get("question", "")),
                                format=(
                                    q_data.get("format")
                                    or ("radio" if q_data.get("options") else None)
                                ),
                                options=q_data.get("options"),
                            )
                        )
                else:
                    logger.warning(
                        f"Merged question data missing 'question' key: {q_data}"
                    )

            return merged_questions

        except (json.JSONDecodeError, ValueError) as e:
            telemetry_error = e
            error_message = (
                f"JSON decoding error during semantic merging: {e} - Content: {content}"
            )
            logger.error(error_message)
            return questions
        except Exception as e:
            telemetry_error = e
            error_message = f"An unexpected error occurred during semantic merging: {e}"
            logger.error(error_message)
            return questions
        finally:
            latency_ms = (time.perf_counter() - start_time) * 1000.0
            metadata = {
                "provider": "semantic_merge",
                "request_id": request_id,
                "taxonomy": taxonomy_name,
                "latency_ms": latency_ms,
            }
            if isinstance(parsed_response, list):
                merged_questions_for_log = parsed_response[:3]
                raw_keys = []
            elif isinstance(parsed_response, dict):
                merged_questions_for_log = parsed_response.get("merged_questions", [])[
                    :3
                ]
                raw_keys = list(parsed_response.keys())
            else:
                merged_questions_for_log = []
                raw_keys = []

            output_payload = {
                "merged_questions": [
                    q.get("question")
                    for q in merged_questions_for_log
                    if isinstance(q, dict)
                ],
                "raw_keys": raw_keys,
            }
            usage_details = extract_usage_details(response)
            finalize_provider_generation(
                generation,
                output_payload=output_payload if parsed_response else None,
                metadata=metadata,
                usage_details=usage_details,
                error=telemetry_error,
            )

    async def _get_voted_results(
        self,
        results_with_providers: List[Tuple[str, Union[Dict[str, Any], Exception]]],
        include_debug_details: bool,
        *,
        request_span=None,
        request_id: Optional[str] = None,
        taxonomy_name: Optional[str] = None,
        skip_followups: bool = False,
        skip_semantic_merge: bool = False,
        conversation_id: Optional[str] = None,
    ) -> ClassificationResponse:
        """Combine multiple provider results by weighted voting.

        Args:
          results_with_providers: A list of (provider_instance_name, result_or_exception) tuples.
          include_debug_details: Whether to include raw and intermediate scoring data.
          skip_semantic_merge: If True, skip semantic merging of follow-up questions for lower latency.

        Returns:
          A `ClassificationResponse` with aggregated labels and questions.
        """
        label_scores = defaultdict(float)
        question_scores = defaultdict(float)
        no_legal_problem_scores = 0.0
        total_provider_weight = 0.0
        raw_provider_results = {}

        for provider_instance_name, result in results_with_providers:
            base_weight = CLASSIFIER_WEIGHTS.get(
                provider_instance_name, 1.0
            )  # Default to 1.0 if weight not defined
            total_provider_weight += base_weight
            if isinstance(result, Exception):
                logger.error(
                    f"Classifier provider '{provider_instance_name}' failed: {result}"
                )
                raw_provider_results[provider_instance_name] = {"error": str(result)}
            else:
                # Serialize labels within the raw result to ensure JSON compatibility
                serialized_result = result.copy()
                if "labels" in serialized_result and isinstance(
                    serialized_result["labels"], list
                ):
                    serialized_result["labels"] = [
                        label.model_dump() if isinstance(label, Label) else label
                        for label in serialized_result["labels"]
                    ]
                if "questions" in serialized_result and isinstance(
                    serialized_result["questions"], list
                ):
                    serialized_result["questions"] = [
                        q.model_dump() if isinstance(q, FollowUpQuestion) else q
                        for q in serialized_result["questions"]
                    ]
                raw_provider_results[provider_instance_name] = serialized_result

                # Aggregate the likely_no_legal_problem vote
                voted_no_legal = result.get("likely_no_legal_problem", False)
                if voted_no_legal:
                    no_legal_problem_scores += base_weight
                    logger.info(
                        f"Provider {provider_instance_name} voted 'likely_no_legal_problem=True' with weight {base_weight}"
                    )
                else:
                    logger.info(
                        f"Provider {provider_instance_name} voted 'likely_no_legal_problem=False' with weight {base_weight}"
                    )

                for label_entry in result.get("labels", []):
                    if isinstance(label_entry, Label):
                        label_str = label_entry.label
                        confidence = (
                            label_entry.confidence
                            if label_entry.confidence is not None
                            else 1.0
                        )
                    else:  # It's a dict
                        label_str = label_entry.get("label")
                        confidence = label_entry.get("confidence", 1.0)

                    weighted_score = base_weight * confidence

                    if label_str:
                        label_scores[label_str] += weighted_score

                for question_entry in result.get("questions", []):
                    if isinstance(question_entry, dict):
                        question_text = question_entry.get("question")
                    else:  # Assume it's a string
                        question_text = question_entry

                    if question_text:
                        question_scores[question_text] += base_weight

        # Sort labels and questions by their scores in descending order
        sorted_labels = sorted(
            label_scores.items(), key=lambda item: item[1], reverse=True
        )

        # For questions, we need to preserve format and options, so we'll collect them directly
        # from the raw results, prioritizing questions from higher-weighted providers if there are conflicts.
        all_questions_with_details = {}
        for provider_instance_name, result in results_with_providers:
            provider_weight = CLASSIFIER_WEIGHTS.get(provider_instance_name, 1.0)
            if not isinstance(result, Exception):
                for question_entry in result.get("questions", []):
                    if isinstance(question_entry, dict):
                        question_text = question_entry.get("question")
                        if question_text:
                            # If question doesn't exist or current provider has higher weight, update it
                            if (
                                question_text not in all_questions_with_details
                                or provider_weight
                                > CLASSIFIER_WEIGHTS.get(
                                    all_questions_with_details[question_text].get(
                                        "provider", ""
                                    ),
                                    0,
                                )
                            ):
                                all_questions_with_details[question_text] = {
                                    "question_obj": FollowUpQuestion(
                                        question=question_text,
                                        format=question_entry.get("format")
                                        or question_entry.get("type"),
                                        options=question_entry.get("options"),
                                    ),
                                    "provider": provider_instance_name,
                                }
                    elif isinstance(question_entry, str):
                        if (
                            question_entry not in all_questions_with_details
                            or provider_weight
                            > CLASSIFIER_WEIGHTS.get(
                                all_questions_with_details[question_entry].get(
                                    "provider", ""
                                ),
                                0,
                            )
                        ):
                            all_questions_with_details[question_entry] = {
                                "question_obj": FollowUpQuestion(
                                    question=question_entry
                                ),
                                "provider": provider_instance_name,
                            }

        # Skip follow-up question processing entirely when skip_followups is True
        if skip_followups:
            final_top_questions = []
        else:
            all_collected_questions = list(all_questions_with_details.values())
            all_question_objs = [
                item["question_obj"] for item in all_collected_questions
            ]

            # Semantic merge handles both deduplication and ranking
            # Skip if requested or if we have 3 or fewer unique questions (no benefit from merge)
            if skip_semantic_merge or len(all_question_objs) <= 3:
                # Simple deduplication by question text without LLM call
                seen_questions = set()
                merged_questions = []
                for q in all_question_objs:
                    if q.question not in seen_questions:
                        seen_questions.add(q.question)
                        merged_questions.append(q)
            else:
                merged_questions = await self._semantically_merge_questions(
                    all_question_objs,
                    request_span=request_span,
                    request_id=request_id,
                    taxonomy_name=taxonomy_name,
                    conversation_id=conversation_id,
                )

            # Take the top 3 questions from the semantic merge result
            # The semantic merge is responsible for both combining similar questions
            # and ranking them by relevance
            final_top_questions = merged_questions[:3] if merged_questions else []

            # Final safety normalization: if a question has options but no format,
            # default to 'radio' so clients can render it correctly.
            for q in final_top_questions:
                if getattr(q, "options", None) and not getattr(q, "format", None):
                    q.format = "radio"

        # Take the top 2 most common labels
        top_labels = [
            Label(label=label, confidence=score) for label, score in sorted_labels[:2]
        ]

        # Determine if there's likely no legal problem based on:
        # 1. Strong majority of providers voting "no legal problem"
        # 2. Very low agreement on the problem category (high entropy), but
        #    treat parent/child (top-level/subcategory) matches as agreement
        likely_no_legal_problem = False
        disagreement_score = 0.0

        # Check consensus on no legal problem (threshold: 30% of total weight)
        no_legal_problem_threshold = total_provider_weight * 0.3
        if (
            no_legal_problem_scores >= no_legal_problem_threshold
            and no_legal_problem_scores > 0
        ):
            likely_no_legal_problem = True
            logger.info(
                f"Setting likely_no_legal_problem=True: {no_legal_problem_scores}/{total_provider_weight} providers voted 'no legal problem'"
            )

        # Helper: extract the top-level category from a label string.
        # Supports formats like 'Top > Sub' or 'Top,Sub' or just 'Top'
        def _extract_top_level(label_str: str) -> str:
            if not label_str:
                return ""
            if ">" in label_str:
                return label_str.split(">")[0].strip()
            if "," in label_str:
                return label_str.split(",")[0].strip()
            return label_str.strip()

        # Helper: hierarchy distance (0 = exact same, 1 = same top-level, 2 = different top-level)
        def _hierarchy_distance(a: str, b: str) -> int:
            if not a or not b:
                return 2
            if a.strip() == b.strip():
                return 0
            if _extract_top_level(a) == _extract_top_level(b):
                return 1
            return 2

        # Check for very low agreement on categories (if labels exist)
        # Aggregate scores at the top-level category first. This avoids
        # penalizing cases where providers disagree on subcategories but
        # agree on the broader area of law.
        if sorted_labels and not likely_no_legal_problem:
            # Compute top-level aggregated scores
            top_level_scores = defaultdict(float)
            for label_str, score in label_scores.items():
                top = _extract_top_level(label_str)
                top_level_scores[top] += score

            sorted_top_levels = sorted(
                top_level_scores.items(), key=lambda item: item[1], reverse=True
            )

            top_level_score = sorted_top_levels[0][1] if sorted_top_levels else 0

            # Compute disagreement score: 1.0 - (top category weight / total weight)
            # Higher score indicates more disagreement (less consensus on top category)
            disagreement_score = (
                1.0 - (top_level_score / total_provider_weight)
                if total_provider_weight > 0
                else 0.0
            )

        # Build structured likely_no_legal_problem info
        lnlp_obj = {
            "value": likely_no_legal_problem,
            "weighted_result": (
                (no_legal_problem_scores / total_provider_weight)
                if total_provider_weight > 0
                else 0.0
            ),
            "disagreement_score": disagreement_score,
        }

        if not top_labels and not final_top_questions:
            # Build fallback response - empty follow_up_questions if skip_followups is True
            if skip_followups:
                fallback_questions = []
            else:
                fallback_questions = [
                    FollowUpQuestion(
                        question="All classifier providers failed to return a response or no labels/questions were found."
                    )
                ]
            response_data = {
                "labels": [],
                "follow_up_questions": fallback_questions,
                "likely_no_legal_problem": lnlp_obj,
            }
        else:
            response_data = {
                "labels": top_labels,
                "follow_up_questions": final_top_questions,
                "likely_no_legal_problem": lnlp_obj,
            }

        if include_debug_details:
            response_data["raw_provider_results"] = raw_provider_results
            response_data["weighted_label_scores"] = dict(label_scores)
            response_data["weighted_question_scores"] = dict(question_scores)

        logger.info(f"_get_voted_results response_data: {response_data}")
        return ClassificationResponse(**response_data)

    async def _get_first_result(
        self,
        results_with_providers: List[Tuple[str, Union[Dict[str, Any], Exception]]],
        include_debug_details: bool,
        skip_followups: bool = False,
    ) -> ClassificationResponse:
        """Return the first successful provider result, optionally with debug data.

        Args:
          results_with_providers: A list of (provider_instance_name, result_or_exception) tuples.
          include_debug_details: Whether to include raw results for debugging.
          skip_followups: Whether to skip follow-up questions in the response.

        Returns:
          A `ClassificationResponse` based on the first non-error provider output,
          or a response indicating that no providers succeeded.
        """
        raw_provider_results = {}
        for provider_instance_name, result in results_with_providers:
            if isinstance(result, Exception):
                logger.error(
                    f"Classifier provider '{provider_instance_name}' failed: {result}"
                )
                raw_provider_results[provider_instance_name] = {"error": str(result)}
            elif result and isinstance(result, dict) and "error" in result:
                logger.error(
                    f"Classifier provider '{provider_instance_name}' returned an error: {result['error']}"
                )
                raw_provider_results[provider_instance_name] = {
                    "error": result["error"]
                }
            else:
                # Serialize labels within the raw result to ensure JSON compatibility
                serialized_result = result.copy()
                if "labels" in serialized_result and isinstance(
                    serialized_result["labels"], list
                ):
                    serialized_result["labels"] = [
                        label.model_dump() if isinstance(label, Label) else label
                        for label in serialized_result["labels"]
                    ]
                if "questions" in serialized_result and isinstance(
                    serialized_result["questions"], list
                ):
                    serialized_result["questions"] = [
                        q.model_dump() if isinstance(q, FollowUpQuestion) else q
                        for q in serialized_result["questions"]
                    ]
                raw_provider_results[provider_instance_name] = serialized_result

                # Build follow-up questions list, or empty if skip_followups is True
                if skip_followups:
                    follow_up_questions = []
                else:
                    follow_up_questions = [
                        (
                            FollowUpQuestion(
                                question=str(q.get("question")),
                                format=(q.get("format") or q.get("type")),
                                options=q.get("options"),
                            )
                            if isinstance(q, dict)
                            else FollowUpQuestion(question=str(q))
                        )
                        for q in result.get("questions", [])
                    ]
                    # Final safety normalization for first-result path as well
                    for fq in follow_up_questions:
                        if getattr(fq, "options", None) and not getattr(
                            fq, "format", None
                        ):
                            fq.format = "radio"

                # Build likely_no_legal_problem from provider result
                lnlp_value = result.get("likely_no_legal_problem", False)
                lnlp_obj = {
                    "value": lnlp_value if isinstance(lnlp_value, bool) else False,
                    "weighted_result": 1.0 if lnlp_value else 0.0,
                    "disagreement_score": 0.0,
                }

                response_data = {
                    "labels": result.get("labels", []),
                    "follow_up_questions": follow_up_questions,
                    "likely_no_legal_problem": lnlp_obj,
                }
                if include_debug_details:
                    response_data["raw_provider_results"] = raw_provider_results
                logger.info(f"_get_first_result response_data: {response_data}")
                return ClassificationResponse(**response_data)

        # Return fallback response - empty follow_up_questions if skip_followups is True
        if skip_followups:
            fallback_questions = []
        else:
            fallback_questions = [
                FollowUpQuestion(
                    question="No classifier providers returned a successful response."
                )
            ]

        # Default likely_no_legal_problem for fallback (no providers succeeded)
        fallback_lnlp = {
            "value": False,
            "weighted_result": 0.0,
            "disagreement_score": 0.0,
        }

        logger.info(
            f"_get_first_result response_data: {{'labels': [], 'follow_up_questions': {fallback_questions}}}"
        )
        return ClassificationResponse(
            labels=[],
            follow_up_questions=fallback_questions,
            likely_no_legal_problem=fallback_lnlp,
            raw_provider_results=(
                raw_provider_results if include_debug_details else None
            ),
        )

    def _get_cache_key(
        self, problem_description: str, provider_instance_name: str, prompt: str
    ) -> str:
        return f"{problem_description}|{provider_instance_name}|{prompt}"

    async def classify(
        self, request: ClassificationRequest, enabled_models: Optional[List[str]] = None
    ) -> ClassificationResponse:
        """Run classification with enabled providers and aggregate results."""

        request_id = str(uuid4())
        trace_input = {
            "request_id": request_id,
            "problem_description": self._truncate_text(request.problem_description),
            "taxonomy_name": request.taxonomy_name,
            "taxonomy_format": request.taxonomy_format,
            "decision_mode": request.decision_mode,
            "include_debug_details": request.include_debug_details,
            "skip_followups": request.skip_followups,
            "enabled_models": enabled_models or request.enabled_models,
            "conversation_id": request.conversation_id,
        }

        # Use user's query (truncated) as the span name so it is visible in Langfuse web view
        span_name = (
            self._truncate_text(request.problem_description, limit=200)
            or "classification.request"
        )

        metadata = {
            "request_id": request_id,
            "taxonomy": request.taxonomy_name,
            "taxonomy_format": request.taxonomy_format,
            "decision_mode": request.decision_mode,
            "skip_followups": request.skip_followups,
        }
        if request.conversation_id:
            metadata["conversation_id"] = request.conversation_id

        request_span = start_request_trace(
            name=span_name,
            input_payload=trace_input,
            metadata=metadata,
        )

        response: Optional[ClassificationResponse] = None
        error: Optional[BaseException] = None

        try:
            taxonomy_file = TAXONOMY_MAPPING.get(request.taxonomy_name)
            if not taxonomy_file:
                response = ClassificationResponse(
                    labels=[],
                    follow_up_questions=(
                        []
                        if request.skip_followups
                        else [
                            FollowUpQuestion(
                                question=f"Taxonomy '{request.taxonomy_name}' not found in configuration."
                            )
                        ]
                    ),
                )
                return response

            # For LIST taxonomy, use the simplified version (Title column only)
            # to avoid confusing LLMs with the codes
            if request.taxonomy_name == "list":
                taxonomy_df = load_list_taxonomy_simple()
                logger.info("Using simplified LIST taxonomy (Title column only)")
            else:
                taxonomy_df = self.taxonomies.get(request.taxonomy_name)
                if taxonomy_df is None:
                    logger.info(
                        f"Taxonomy '{request.taxonomy_name}' not preloaded; attempting to load from disk: {taxonomy_file}"
                    )
                    taxonomy_df = self._load_taxonomy(taxonomy_file)
                    if taxonomy_df is None:
                        response = ClassificationResponse(
                            labels=[],
                            follow_up_questions=(
                                []
                                if request.skip_followups
                                else [
                                    FollowUpQuestion(
                                        question="Taxonomy could not be loaded."
                                    )
                                ]
                            ),
                        )
                        return response
                    with self._tax_lock:
                        self.taxonomies[request.taxonomy_name] = taxonomy_df

            explicit_enabled_models = (
                enabled_models if enabled_models is not None else request.enabled_models
            )
            if explicit_enabled_models is not None:
                current_providers = self._init_providers(
                    enabled_providers_override=explicit_enabled_models
                )
            else:
                current_providers = self.providers
                # For refinement calls with follow-up answers, default to LLM providers only.
                if request.followup_answers:
                    current_providers = [
                        p
                        for p in current_providers
                        if isinstance(p, LLMClassifierProvider)
                    ]
                    logger.info(
                        "followup_answers present: defaulting to LLM providers only "
                        "(excluding non-LLM providers like spot/keyword)"
                    )

            tasks = []
            results_with_providers: List[
                Tuple[str, Union[Dict[str, Any], Exception]]
            ] = []
            prompts_for_caching: Dict[str, str] = {}

            for provider in current_providers:
                final_prompt, prompt_template = load_prompt(
                    provider.provider_type,
                    taxonomy_df,
                    skip_followups=request.skip_followups,
                    taxonomy_name=request.taxonomy_name,
                )
                prompts_for_caching[provider.instance_name] = prompt_template

                if self.cache_enabled:
                    cache_key = self._get_cache_key(
                        request.problem_description,
                        provider.instance_name,
                        prompt_template,
                    )
                    if cache_key in self.cache:
                        logger.info(f"Cache hit for {provider.instance_name}")
                        results_with_providers.append(
                            (provider.instance_name, self.cache.get(cache_key))
                        )
                        continue

                tasks.append(
                    asyncio.create_task(
                        self._classify_with_telemetry(
                            provider=provider,
                            request=request,
                            taxonomy_df=taxonomy_df,
                            final_prompt=final_prompt,
                            request_span=request_span,
                            request_id=request_id,
                        ),
                        name=provider.instance_name,
                    )
                )

            new_results = await asyncio.gather(*tasks, return_exceptions=True)
            task_names = [task.get_name() for task in tasks]
            new_results_with_providers = [
                (task_name, result)
                for task_name, result in zip(task_names, new_results)
            ]

            verbose = os.getenv("DEBUG_LOG") in {"1", "true", "TRUE", "True"}
            logger.info("[ClassificationService] Results summary:")
            for name, res in new_results_with_providers:
                if name == "gpt-5" or verbose:
                    if isinstance(res, Exception):
                        logger.error(f" - {name}: Exception -> {repr(res)}")
                    else:
                        status = "ok"
                        if isinstance(res, dict) and res.get("error"):
                            status = f"error: {res.get('error')}"
                        latency = None
                        telemetry_payload = (
                            res.get("telemetry") if isinstance(res, dict) else None
                        )
                        if isinstance(telemetry_payload, dict):
                            latency = telemetry_payload.get("latency_ms")
                        latency_str = (
                            f", latency_ms={latency:.2f}"
                            if isinstance(latency, (int, float))
                            else ""
                        )
                        logger.info(f" - {name}: {status}{latency_str}")
            results_with_providers.extend(new_results_with_providers)

            if self.cache_enabled:
                logger.info("Caching new results...")
                for task_name, result in new_results_with_providers:
                    if not isinstance(result, Exception):
                        prompt_template = prompts_for_caching.get(task_name, "")
                        cache_key = self._get_cache_key(
                            request.problem_description, task_name, prompt_template
                        )
                        self.cache[cache_key] = result

            decision_mode = request.decision_mode or DECISION_MODE
            if decision_mode == "first":
                response = await self._get_first_result(
                    results_with_providers,
                    request.include_debug_details,
                    skip_followups=request.skip_followups,
                )
            else:
                response = await self._get_voted_results(
                    results_with_providers,
                    request.include_debug_details,
                    request_span=request_span,
                    request_id=request_id,
                    taxonomy_name=request.taxonomy_name,
                    skip_followups=request.skip_followups,
                    skip_semantic_merge=request.skip_semantic_merge,
                )

            # Convert labels to LIST format if requested
            # Automatically use LIST format if taxonomy_name is "list"
            should_convert_to_list = (
                request.taxonomy_format == "list" or request.taxonomy_name == "list"
            )
            if should_convert_to_list and response:
                response = self._convert_labels_to_list_format(
                    response,
                    taxonomy_name=request.taxonomy_name,
                )

            return response
        except Exception as exc:
            error = exc
            raise
        finally:
            output_payload = response.model_dump() if response else None
            finalize_request_trace(
                request_span,
                output_payload=output_payload,
                error=error,
            )
            await flush_telemetry_async()

    def _load_taxonomy(self, taxonomy_file: str) -> Optional[list]:
        """Load taxonomy CSV as a list of dicts with basic cleaning.

        Args:
          taxonomy_file: Path to a CSV file containing taxonomy rows.

        Returns:
          A cleaned list of dict rows or None if loading fails.
        """
        try:
            rows = read_csv_as_list_of_dicts(taxonomy_file)
            rows = dedupe_and_clean_rows(rows)
            return rows
        except FileNotFoundError:
            logger.warning(f"Taxonomy file not found: {taxonomy_file}")
            return None
        except Exception as e:
            logger.error(f"Error loading taxonomy file {taxonomy_file}: {e}")
            return None

    def _is_list_code_format(self, label: str) -> bool:
        """Check if a label appears to already be in LIST code format.

        LIST codes look like "FA-01-00-00-00 > Some Title" or just "FA-01-00-00-00".
        """
        import re

        # LIST code pattern: 2 letters, dash, 2 digits, repeated 4 more times
        pattern = r"^[A-Z]{2}-\d{2}-\d{2}-\d{2}-\d{2}"
        return bool(re.match(pattern, label.strip()))

    def _convert_labels_to_list_format(
        self,
        response: ClassificationResponse,
        taxonomy_name: str,
    ) -> ClassificationResponse:
        """Convert classification labels to LIST taxonomy format.

        Args:
            response: The classification response with labels to convert.
            taxonomy_name: The taxonomy that was used for classification.

        Returns:
            A new ClassificationResponse with labels converted to LIST codes.
        """
        if not response.labels:
            return response

        converted_labels = []
        for label in response.labels:
            original_label = label.label

            # Extract LIST code from already-formatted label if present
            list_code = None
            if self._is_list_code_format(original_label):
                # Extract code from "CODE > Title" format or just "CODE"
                list_code = original_label.split(" > ")[0].strip()
                converted_labels.append(
                    Label(
                        label=original_label,
                        confidence=label.confidence,
                        id=list_code,
                    )
                )
                continue

            if taxonomy_name == "list":
                # Labels are LIST taxonomy titles, convert to codes
                list_code = lookup_list_code_from_title(original_label)
                if list_code:
                    converted_labels.append(
                        Label(
                            label=format_list_label(list_code),
                            confidence=label.confidence,
                            id=list_code,
                        )
                    )
                else:
                    # Keep original if no mapping found (shouldn't happen often)
                    logger.warning(f"No LIST code found for title: {original_label}")
                    converted_labels.append(label)
            else:
                # Labels are OSB taxonomy, convert via mapping
                result = convert_osb_label_to_list(original_label)
                if result:
                    list_code, list_title = result
                    converted_labels.append(
                        Label(
                            label=format_list_label(list_code),
                            confidence=label.confidence,
                            id=list_code,
                        )
                    )
                else:
                    # No mapping exists - check if label might be a LIST title directly
                    # (This can happen if a classifier returned LIST titles instead of OSB)
                    list_code = lookup_list_code_from_title(original_label)
                    if list_code:
                        converted_labels.append(
                            Label(
                                label=format_list_label(list_code),
                                confidence=label.confidence,
                                id=list_code,
                            )
                        )
                    else:
                        # True unmapped label
                        logger.info(f"No LIST mapping for OSB label: {original_label}")
                        # Keep the original label but mark it as unmapped
                        converted_labels.append(
                            Label(
                                label=f"(unmapped) {original_label}",
                                confidence=label.confidence,
                            )
                        )

        # Create a new response with converted labels
        return ClassificationResponse(
            labels=converted_labels,
            follow_up_questions=response.follow_up_questions,
            likely_no_legal_problem=response.likely_no_legal_problem,
            raw_provider_results=response.raw_provider_results,
            weighted_label_scores=response.weighted_label_scores,
            weighted_question_scores=response.weighted_question_scores,
        )

    async def _classify_with_telemetry(
        self,
        *,
        provider: ClassifierProvider,
        request: ClassificationRequest,
        taxonomy_df: Any,
        final_prompt: str,
        request_span,
        request_id: str,
    ) -> Dict[str, Any]:
        """Execute a provider classify call while emitting Langfuse generation telemetry."""
        generation = start_provider_generation(
            request_span,
            name=provider.instance_name,
            model=getattr(provider, "model_name", provider.provider_type),
            input_payload={
                "prompt": self._truncate_text(final_prompt),
                "problem_description": self._truncate_text(request.problem_description),
                "taxonomy": request.taxonomy_name,
                "request_id": request_id,
                # Surface configured reasoning effort for telemetry to aid debugging
                "reasoning_effort": GPT_5_REASONING_EFFORT,
                "conversation_id": request.conversation_id,
            },
            metadata={
                "provider": provider.provider_type,
                "request_id": request_id,
                "taxonomy": request.taxonomy_name,
                **(
                    {"conversation_id": request.conversation_id}
                    if request.conversation_id
                    else {}
                ),
            },
        )

        start_time = time.perf_counter()
        try:
            # Build kwargs for classify: include reasoning_effort only when configured
            classify_kwargs: Dict[str, Any] = {"custom_prompt": final_prompt}
            if GPT_5_REASONING_EFFORT is not None:
                classify_kwargs["reasoning_effort"] = GPT_5_REASONING_EFFORT

            # Pass followup_answers if provided for multi-turn refinement
            if request.followup_answers:
                classify_kwargs["followup_answers"] = request.followup_answers

            # Pass taxonomy_format to SPOT provider so it can return LIST codes directly
            # Automatically use LIST format if taxonomy_name is "list"
            if provider.instance_name == "spot":
                taxonomy_format = (
                    "list"
                    if request.taxonomy_name == "list"
                    else request.taxonomy_format
                )
                classify_kwargs["taxonomy_format"] = taxonomy_format

            try:
                result = await asyncio.wait_for(
                    provider.classify(
                        request.problem_description,
                        taxonomy_df,
                        **classify_kwargs,
                    ),
                    timeout=CLASSIFIER_TIMEOUT_SECONDS,
                )
            except asyncio.TimeoutError as exc:
                raise TimeoutError(
                    f"Provider '{provider.instance_name}' timed out after "
                    f"{CLASSIFIER_TIMEOUT_SECONDS:.1f}s"
                ) from exc
            latency_ms = (time.perf_counter() - start_time) * 1000.0
            self._attach_latency_metadata(result, latency_ms)
            finalize_provider_generation(
                generation,
                output_payload=self._summarize_provider_result(result),
                metadata={
                    "provider": provider.instance_name,
                    "request_id": request_id,
                    "taxonomy": request.taxonomy_name,
                    "latency_ms": latency_ms,
                },
                usage_details=self._extract_usage_details_from_result(result),
            )
            return result
        except Exception as exc:
            latency_ms = (time.perf_counter() - start_time) * 1000.0
            finalize_provider_generation(
                generation,
                metadata={
                    "provider": provider.instance_name,
                    "request_id": request_id,
                    "taxonomy": request.taxonomy_name,
                    "latency_ms": latency_ms,
                },
                error=exc,
            )
            raise

    def _attach_latency_metadata(
        self, provider_result: Union[Dict[str, Any], Exception], latency_ms: float
    ) -> None:
        if not isinstance(provider_result, dict):
            return
        telemetry = provider_result.get("telemetry")
        if not isinstance(telemetry, dict):
            telemetry = {}
            provider_result["telemetry"] = telemetry
        telemetry["latency_ms"] = latency_ms

    def _extract_usage_details_from_result(
        self, provider_result: Union[Dict[str, Any], Exception]
    ) -> Optional[Dict[str, int]]:
        if not isinstance(provider_result, dict):
            return None
        telemetry = provider_result.get("telemetry")
        if not isinstance(telemetry, dict):
            return None
        usage_details = telemetry.get("usage_details")
        if not isinstance(usage_details, dict):
            return None
        sanitized: Dict[str, int] = {}
        for key, value in usage_details.items():
            if isinstance(value, (int, float)):
                sanitized[key] = int(value)
        return sanitized or None

    def _summarize_provider_result(
        self, provider_result: Union[Dict[str, Any], Exception]
    ) -> Optional[Dict[str, Any]]:
        if not isinstance(provider_result, dict):
            return None

        summary: Dict[str, Any] = {
            "likely_no_legal_problem": provider_result.get("likely_no_legal_problem")
        }

        labels_summary: List[str] = []
        for entry in provider_result.get("labels", []):
            if isinstance(entry, Label):
                labels_summary.append(entry.label)
            elif isinstance(entry, dict):
                label_value = entry.get("label") or entry.get("category")
                if label_value:
                    labels_summary.append(str(label_value))
        if labels_summary:
            summary["labels"] = labels_summary[:5]

        question_summary: List[str] = []
        for question in provider_result.get("questions", []):
            if isinstance(question, FollowUpQuestion):
                question_summary.append(question.question)
            elif isinstance(question, dict) and question.get("question"):
                question_summary.append(str(question["question"]))
            elif isinstance(question, str):
                question_summary.append(question)
        if question_summary:
            summary["questions"] = question_summary[:3]

        if provider_result.get("error"):
            summary["error"] = provider_result.get("error")

        telemetry = provider_result.get("telemetry")
        if isinstance(telemetry, dict):
            latency_value = telemetry.get("latency_ms")
            if isinstance(latency_value, (int, float)):
                summary["latency_ms"] = latency_value
        return summary

    @staticmethod
    def _truncate_text(value: str, limit: int = 1000) -> str:
        if not value:
            return ""
        if len(value) <= limit:
            return value
        trimmed = value[:limit]
        overflow = len(value) - limit
        return f"{trimmed}...(+{overflow} chars)"
