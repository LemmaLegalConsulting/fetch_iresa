from app.providers.base import LLMClassifierProvider, load_prompt
from app.models.api_models import Label
from openai import AsyncAzureOpenAI, AsyncOpenAI
import os
from typing import Any, Dict, List, Literal, Optional

if os.getenv("ENV") != "production":
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except ImportError:
        pass
import json
from datetime import datetime
from app.utils.backoff import run_with_backoff_async
from app.utils.logging import get_logger
from app.utils.json_helpers import parse_json_from_llm_response
from app.providers.utils import build_basic_telemetry, build_messages
from time import perf_counter
from app.core.config import (
    OPENAI_SUPPORTS_REASONING_OBJECT,
    OPENAI_SUPPORTS_REASONING_EFFORT,
    OPENAI_SDK_VERSION,
)

logger = get_logger(__name__)
GPT_5_FAMILY_MODELS = {"gpt-5", "gpt-5-mini", "gpt-5.2"}


class OpenAIProvider(LLMClassifierProvider):
    """Classifier provider backed by OpenAI Chat Completions."""

    def __init__(self, model_name: str = "gpt-4.1-mini"):
        """Initialize the OpenAI provider.

        Args:
          model_name: The OpenAI model to use for classification.
        """
        super().__init__("openai", model_name=model_name)

    def _get_client(self):
        """Create and return an AsyncOpenAI client using env credentials."""
        return AsyncOpenAI(api_key=os.environ.get("OPENAI_API_KEY"), max_retries=0)

    async def classify(
        self,
        problem_description: str,
        taxonomy: Any,
        custom_prompt: Optional[str] = None,
        reasoning_effort: Optional[
            Literal["none", "low", "medium", "high", "xhigh"]
        ] = None,
        followup_answers: Optional[List[Any]] = None,
    ) -> Dict[str, List[Any]]:
        """Classify a description using an OpenAI JSON-formatted response.

        Builds a system prompt from the taxonomy and parses the JSON response
        into "labels" and optional "questions".

        When followup_answers is provided, builds a multi-turn conversation:
        1. System message with taxonomy
        2. User message with initial problem description
        3. For each Q&A pair: assistant asks question, user provides answer
        4. Final user message requesting refined classification

        Args:
          problem_description: The input text to classify.
          taxonomy: The taxonomy DataFrame for prompt context.
          custom_prompt: Optional fully-formed system prompt to use.
          reasoning_effort: Optional reasoning effort level for gpt-5 models.
          followup_answers: Optional list of FollowUpAnswer objects for multi-turn refinement.

        Returns:
          Dict containing keys "labels" and "questions". List elements may be
          Pydantic `Label` dicts or plain dicts.
        """
        if custom_prompt:
            prompt = custom_prompt
        else:
            prompt, _ = load_prompt(self.provider_type, taxonomy)

        verbose = os.getenv("DEBUG_LOG") in {
            "1",
            "true",
            "TRUE",
            "True",
        } or self.model_name in GPT_5_FAMILY_MODELS
        if verbose:
            logger.info(
                f"[OpenAIProvider:{self.model_name}] classify start @ {datetime.utcnow().isoformat()}Z"
            )
            logger.info(
                f"[OpenAIProvider:{self.model_name}] prompt_len={len(prompt)} desc_len={len(problem_description)}"
            )

        preview: Optional[str] = None
        last_response: Optional[Any] = None
        # Use Responses API for gpt-5; Chat Completions for the rest
        content: Optional[str] = None
        if self.model_name in GPT_5_FAMILY_MODELS:
            if verbose:
                logger.info(f"[OpenAIProvider:{self.model_name}] using Responses API")
            try:
                # Use pre-computed flags from app.core.config to determine
                # whether to send a flat `reasoning_effort` or a nested
                # `reasoning` object. This is cheaper than inspecting the
                # signature on every request and provides consistent behavior
                # across the process lifetime.
                supports_reasoning_effort = OPENAI_SUPPORTS_REASONING_EFFORT
                supports_reasoning_object = OPENAI_SUPPORTS_REASONING_OBJECT
                if reasoning_effort is None:
                    logger.info(
                        f"[OpenAIProvider:{self.model_name}] reasoning_effort not provided; using provider default; sdk_version={OPENAI_SDK_VERSION}"
                    )
                else:
                    logger.info(
                        f"[OpenAIProvider:{self.model_name}] configured reasoning_effort={reasoning_effort}; supports_effort={supports_reasoning_effort} supports_object={supports_reasoning_object} sdk_version={OPENAI_SDK_VERSION}"
                    )

                # Build base params. We'll conditionally attach either the
                # flat or nested reasoning parameter depending on SDK support.
                responses_params: Dict[str, Any] = {
                    "model": self.model_name,
                    "input": f"SYSTEM:\n{prompt}\n\nUSER:\n{problem_description}",
                    # Avoid temperature/response_format for compatibility; enforce JSON via instructions
                }

                if reasoning_effort is not None:
                    if supports_reasoning_effort:
                        responses_params["reasoning_effort"] = reasoning_effort
                        used_reasoning_form = "flat(reasoning_effort)"
                    elif supports_reasoning_object:
                        responses_params["reasoning"] = {"effort": reasoning_effort}
                        used_reasoning_form = "object(reasoning)"
                    else:
                        used_reasoning_form = "omitted"
                    logger.info(
                        f"[OpenAIProvider:{self.model_name}] using reasoning form: {used_reasoning_form}"
                    )
                # Time the Responses API call specifically so we can surface it in telemetry
                responses_start = perf_counter()
                resp = await run_with_backoff_async(
                    self.client.responses.create, **responses_params
                )
                responses_latency_ms = (perf_counter() - responses_start) * 1000.0
                last_response = resp
                # Extract text output safely
                text_chunks: List[str] = []
                if hasattr(resp, "output_text") and resp.output_text:
                    content = resp.output_text
                elif hasattr(resp, "output") and resp.output:
                    try:
                        for item in resp.output:
                            if hasattr(item, "content"):
                                for part in item.content:
                                    if getattr(part, "type", "") in (
                                        "output_text",
                                        "text",
                                    ) and hasattr(part, "text"):
                                        text_chunks.append(part.text)
                        content = "".join(text_chunks)
                    except Exception:
                        content = None
                if content is None:
                    raise RuntimeError("No textual content returned from Responses API")
            except TypeError as te:
                # Older SDKs may not accept the 'reasoning_effort' kwarg on
                # responses.create. If that happens, try calling the Responses
                # API again without that parameter so we can stay on the
                # Responses path (which is generally faster) instead of
                # falling back to chat completions. Only if that also fails
                # do we fall back to chat.completions.
                if verbose:
                    logger.warning(
                        f"[OpenAIProvider:{self.model_name}] responses.create TypeError: {te}; retrying without reasoning_effort"
                    )
                # Retry without reasoning_effort (if it was present)
                try:
                    fallback_params = dict(responses_params)
                    fallback_params.pop("reasoning_effort", None)
                    # Retry the Responses API call without the reasoning kwarg and measure it
                    responses_start = perf_counter()
                    resp = await run_with_backoff_async(
                        self.client.responses.create, **fallback_params
                    )
                    responses_latency_ms = (perf_counter() - responses_start) * 1000.0
                    last_response = resp
                    # proceed as normal
                except Exception as inner_exc:
                    if verbose:
                        logger.warning(
                            f"[OpenAIProvider:{self.model_name}] retrying responses.create without reasoning_effort failed: {inner_exc}; falling back to chat.completions"
                        )
                    # Fallback to Chat Completions; time this call too for comparison
                    chat_start = perf_counter()
                    response = await run_with_backoff_async(
                        self.client.chat.completions.create,
                        model=self.model_name,
                        messages=build_messages(
                            prompt, problem_description, followup_answers
                        ),
                    )
                    chat_latency_ms = (perf_counter() - chat_start) * 1000.0
                    content = response.choices[0].message.content
                    last_response = response
            except AttributeError as ae:
                if verbose:
                    logger.warning(
                        f"[OpenAIProvider:{self.model_name}] responses API not available: {ae}; falling back to chat.completions"
                    )
                chat_start = perf_counter()
                response = await run_with_backoff_async(
                    self.client.chat.completions.create,
                    model=self.model_name,
                    messages=build_messages(
                        prompt, problem_description, followup_answers
                    ),
                )
                chat_latency_ms = (perf_counter() - chat_start) * 1000.0
                content = response.choices[0].message.content
                last_response = response
        else:
            # Some models (e.g., gpt-5) may not support logprobs in Chat Completions
            supports_logprobs = True
            params: Dict[str, Any] = {
                "model": self.model_name,
                "messages": build_messages(
                    prompt, problem_description, followup_answers
                ),
                "temperature": 0.0,
                "response_format": {"type": "json_object"},
            }
            if supports_logprobs:
                params.update({"logprobs": True, "top_logprobs": 1})
            # Defensive: never pass response_format/logprobs to gpt-5
            if self.model_name in GPT_5_FAMILY_MODELS:
                params.pop("response_format", None)
                params.pop("logprobs", None)
                params.pop("top_logprobs", None)

            if verbose:
                safe_params = {k: v for k, v in params.items() if k != "messages"}
                logger.info(
                    f"[OpenAIProvider:{self.model_name}] request params: {safe_params}"
                )
            response = await run_with_backoff_async(
                self.client.chat.completions.create, **params
            )
            content = response.choices[0].message.content
            last_response = response

        if verbose and content is not None:
            preview = content[:400] + ("..." if content and len(content) > 400 else "")
            logger.info(
                f"[OpenAIProvider:{self.model_name}] raw content preview: {preview}"
            )

        logger.info(
            f"[OpenAIProvider:{self.model_name}] Raw content from API: {content}"
        )

        # Use robust JSON extraction and parsing helper that handles:
        # - Code fences with various formats
        # - Trailing commas and other common LLM JSON errors
        # - Safe recovery by extracting first valid JSON object/array
        try:
            parsed_response = parse_json_from_llm_response(content or "{}")
            # Ensure parsed_response is a dict; if it's a list, wrap it or treat as error
            if not isinstance(parsed_response, dict):
                logger.warning(
                    f"[OpenAIProvider:{self.model_name}] Expected dict but got {type(parsed_response).__name__} from parse_json_from_llm_response"
                )
                parsed_response = {}
        except (json.JSONDecodeError, ValueError) as e:
            telemetry_payload = build_basic_telemetry(
                provider=self.provider_type,
                instance_name=self.instance_name,
                model_name=self.model_name,
                response=last_response,
                raw_content=content,
            )
            error_message = f"JSON decoding error from OpenAI: {e} - Content: {content}"
            logger.error(error_message)
            return {
                "labels": [],
                "questions": [],
                "error": error_message,
                "telemetry": telemetry_payload,
            }

        labels: List[Dict[str, Any]] = []
        questions: List[Dict[str, Any]] = []
        likely_no_legal_problem = parsed_response.get("likely_no_legal_problem", False)

        if not likely_no_legal_problem:
            # Categories/labels can appear together with follow-up questions; collect both.
            if parsed_response.get("categories") is not None:
                labels_from_response = parsed_response.get("categories", [])
                for cat in labels_from_response:
                    labels.append(Label(label=cat, confidence=1.0).model_dump())

            if parsed_response.get("labels") is not None:
                # Support alternative key 'labels'
                for item in parsed_response.get("labels", []):
                    if isinstance(item, str):
                        labels.append(Label(label=item, confidence=1.0).model_dump())
                    elif isinstance(item, dict) and item.get("label"):
                        labels.append(
                            Label(
                                label=item.get("label"),
                                confidence=item.get("confidence") or 1.0,
                            ).model_dump()
                        )

            if parsed_response.get("followup_questions") is not None:
                # Primary key for follow-up questions
                questions.extend(
                    [
                        q
                        for q in parsed_response.get("followup_questions", [])
                        if isinstance(q, dict) and q.get("question")
                    ]
                )

            if parsed_response.get("questions") is not None:
                # Support alternative key 'questions'
                questions.extend(
                    [
                        q if isinstance(q, dict) else {"question": str(q)}
                        for q in parsed_response.get("questions", [])
                        if (isinstance(q, dict) and q.get("question"))
                        or isinstance(q, str)
                    ]
                )

        result: Dict[str, Any] = {
            "labels": labels,
            "questions": questions,
            "likely_no_legal_problem": likely_no_legal_problem,
        }
        telemetry_payload = build_basic_telemetry(
            provider=self.provider_type,
            instance_name=self.instance_name,
            model_name=self.model_name,
            response=last_response,
            raw_content=content,
        )
        # Attach micro-timings to telemetry if present so deployed telemetry
        # shows Responses vs Chat fallback durations for diagnosis.
        if "responses_latency_ms" in locals():
            telemetry_payload["responses_latency_ms"] = locals().get(
                "responses_latency_ms"
            )
        if "chat_latency_ms" in locals():
            telemetry_payload["chat_latency_ms"] = locals().get("chat_latency_ms")
        result["telemetry"] = telemetry_payload
        if verbose:
            logger.info(
                f"[OpenAIProvider:{self.model_name}] parsed: labels={len(labels)} questions={len(questions)}"
            )
            debug_payload: Dict[str, Any] = {
                "model": self.model_name,
                "parsed_keys": (
                    list(parsed_response.keys())
                    if isinstance(parsed_response, dict)
                    else []
                ),
            }
            if preview is not None:
                debug_payload["raw_preview"] = preview
            result["debug"] = [debug_payload]
        return result


class GPT41NanoProvider(OpenAIProvider):
    """Preset provider for the `gpt-4.1-nano` model."""

    def __init__(self):
        """Use the `gpt-4.1-nano` model."""
        super().__init__(model_name="gpt-4.1-nano")


class GPT5Provider(OpenAIProvider):
    """Preset provider for the `gpt-5` model (if available)."""

    def __init__(self):
        """Use the `gpt-5` model."""
        super().__init__(model_name="gpt-5")


class GPT52Provider(OpenAIProvider):
    """Preset provider for Azure `gpt-5.2` deployment."""

    def __init__(self):
        """Use the Azure-configured `gpt-5.2` deployment name."""
        super().__init__(model_name=os.getenv("OPENAI_GPT_5_2_MODEL", "gpt-5.2"))

    def _get_client(self):
        """Create and return an AsyncAzureOpenAI or AsyncOpenAI client using env credentials."""
        azure_endpoint = os.environ.get("OPENAI_BASE_URL_GPT_5_2")
        azure_api_key = os.environ.get("OPENAI_GPT_5_2_API_KEY")
        azure_api_version = os.environ.get("OPENAI_GPT_5_2_API_VERSION")
        
        if azure_endpoint and azure_api_key and azure_api_version:
            return AsyncAzureOpenAI(
                api_key=azure_api_key,
                api_version=azure_api_version,
                azure_endpoint=azure_endpoint,
                max_retries=0,
            )
        return AsyncOpenAI(api_key=os.environ.get("OPENAI_API_KEY"), max_retries=0)
