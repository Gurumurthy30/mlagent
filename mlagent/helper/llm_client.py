"""Talks to the Ollama model. Every agent asks questions through this class.

Two ways to ask:
  ask_text        returns the model's answer as plain text
  ask_structured  returns the answer as a validated pydantic object. If the JSON is invalid,
                  the model is told what was wrong and asked once more.
Every call is written to logs/llm_calls.jsonl.
"""

import copy
import os
import re
import time
from datetime import datetime
from typing import Any, TypeVar
from urllib.parse import urlparse

import httpx
import ollama
from langchain_ollama import ChatOllama
from pydantic import BaseModel, ValidationError

from mlagent.configs.config import LlmConfig as ModelConfig
from mlagent.configs.settings import (
    LLM_CALL_TIMEOUT_SECONDS,
    LLM_REPAIR_RETRIES,
    RETRY_WAIT_SECONDS,
    RETRYABLE_STATUS_CODES,
)
from mlagent.helper.events import EventBus, EventType
from mlagent.helper.workspace import Workspace, append_json_line

TEAM_CHARTER = """\
You are an expert member of an autonomous machine learning engineering team.
Core epistemic rules:
1. EVIDENCE-FIRST: Every decision, hypothesis, and feature proposal must cite concrete quantitative findings, fold scores, or error segments from the team digest. Never invent numbers or metrics.
2. HYPOTHESIS-DRIVEN: Never execute generic checklists. Every code execution and model training run tests a specific, falsifiable hypothesis.
3. ZERO SILENT LEAKAGE: All stateful transformations (imputations, frequency maps, ordinal encodings, scalers, target encodings, SVD) must fit strictly inside training folds.
4. READ-ONLY PROJECT FOLDER: Never change, move, or delete the user's raw files. Write all scripts and artifacts inside the designated workspace folder."""

AnswerModel = TypeVar("AnswerModel", bound=BaseModel)

REPAIR_TEXT_CHARS = 1500


class LlmUnavailableError(Exception):
    """Raised when Ollama cannot be used: not running, model missing, or no valid answer."""


def is_cloud_direct(base_url: str) -> bool:
    return urlparse(base_url).hostname == "ollama.com"


def extract_json_text(response_text: str) -> str:
    """Extract JSON from a code fence or surrounding text."""
    # 1. Check for fenced code block containing an object
    fenced_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", response_text, re.DOTALL)
    if fenced_match:
        return fenced_match.group(1).strip()

    # 2. Check for generic code fence
    fenced_generic = re.search(r"```(?:json)?\s*(.*?)```", response_text, re.DOTALL)
    candidate_text = fenced_generic.group(1) if fenced_generic else response_text

    # 3. Extract outermost curly braces
    start_brace = candidate_text.find("{")
    end_brace = candidate_text.rfind("}")
    if start_brace != -1 and end_brace != -1 and start_brace < end_brace:
        return candidate_text[start_brace : end_brace + 1].strip()
    return candidate_text.strip()


def format_compact_schema(model_cls: type[BaseModel]) -> str:
    """Render a compact, token-efficient JSON-like schema representation for Gemma 32B."""
    lines: list[str] = []
    for name, field in model_cls.model_fields.items():
        ann = str(field.annotation)
        type_str = (
            ann.replace("typing.", "")
            .replace("mlagent.graph.state.", "")
            .replace("<class '", "")
            .replace("'>", "")
            .replace("Literal", "enum")
            .replace("NoneType", "null")
        )
        desc = f"  # {field.description}" if field.description else ""
        lines.append(f'  "{name}": {type_str}{desc}')
    return "{\n" + ",\n".join(lines) + "\n}"


def make_all_fields_required(schema: dict) -> dict:
    """Recursively ensure every object in the JSON schema marks all its properties as required."""
    schema = copy.deepcopy(schema)

    def _recurse(node: Any) -> None:
        if isinstance(node, dict):
            if "properties" in node and isinstance(node["properties"], dict):
                node["required"] = list(node["properties"].keys())
            for val in node.values():
                _recurse(val)
        elif isinstance(node, list):
            for item in node:
                _recurse(item)

    _recurse(schema)
    return schema


def _extract_model_name(item: Any) -> str:
    """Extract model name string from an ollama Model object or dictionary."""
    if hasattr(item, "model") and item.model:
        return str(item.model)
    if hasattr(item, "name") and item.name:
        return str(item.name)
    if isinstance(item, dict):
        return str(item.get("model") or item.get("name") or "")
    return str(item)


def _matches_model(installed_name: str, wanted_name: str) -> bool:
    """Case-insensitive model matcher supporting tags, latest, and base prefix."""
    inst = installed_name.strip().lower()
    want = wanted_name.strip().lower()
    if inst == want:
        return True
    want_latest = want if ":" in want else f"{want}:latest"
    inst_latest = inst if ":" in inst else f"{inst}:latest"
    if inst_latest == want_latest:
        return True
    if ":" not in want and inst.startswith(f"{want}:"):
        return True
    return False


class LlmClient:
    def __init__(
        self, model_config: ModelConfig, workspace: Workspace, event_bus: EventBus
    ) -> None:
        self.model_config = model_config
        self.workspace = workspace
        self.event_bus = event_bus
        timeout = getattr(model_config, "request_timeout_seconds", LLM_CALL_TIMEOUT_SECONDS)
        self.chat_model = ChatOllama(
            model=model_config.name,
            base_url=model_config.base_url,
            temperature=model_config.temperature,
            num_ctx=model_config.context,
            client_kwargs={"timeout": timeout},
        )

    def check_ready(self) -> None:
        """Make sure Ollama is running and the configured model is available."""
        model_name = self.model_config.name
        base_url = self.model_config.base_url
        if is_cloud_direct(base_url) and not os.getenv("OLLAMA_API_KEY"):
            raise LlmUnavailableError(
                "Cloud mode needs OLLAMA_API_KEY from https://ollama.com/settings/keys, "
                "and a new terminal is needed after setx."
            )
        try:
            installed_models = ollama.Client(host=base_url).list().models
        except (ConnectionError, httpx.HTTPError, ollama.ResponseError, Exception) as error:
            if is_cloud_direct(base_url):
                raise LlmUnavailableError(
                    f"Cannot reach Ollama Cloud at {base_url}: {error}"
                ) from error
            raise LlmUnavailableError(
                f"Cannot reach Ollama at {base_url}. Start Ollama and run again. ({error})"
            ) from error

        installed_names = [_extract_model_name(m) for m in installed_models if _extract_model_name(m)]
        is_matched = any(_matches_model(name, model_name) for name in installed_names)
        if not is_matched:
            if is_cloud_direct(base_url):
                raise LlmUnavailableError(
                    f"Model '{model_name}' is not available on Ollama Cloud. The name must "
                    "come from https://ollama.com/api/tags, for example gemma4:31b without "
                    "a -cloud suffix."
                )
            avail_str = ", ".join(sorted(installed_names)) if installed_names else "none"
            raise LlmUnavailableError(
                f"Model '{model_name}' is not pulled in Ollama (installed models: [{avail_str}]). "
                f"Use --model <name> to select an installed model, or run: ollama pull {model_name}"
            )

    def ask_text(
        self,
        agent_name: str,
        prompt: str,
        system_message: str | None = None,
        temperature: float | None = None,
    ) -> str:
        return self._ask(
            agent_name,
            prompt,
            json_schema=None,
            system_message=system_message,
            temperature=temperature,
        )

    def ask_structured(
        self,
        agent_name: str,
        prompt: str,
        answer_model: type[AnswerModel] | None = None,
        system_message: str | None = None,
        *,
        output_schema: type[AnswerModel] | None = None,
        system_prompt: str | None = None,
        temperature: float | None = None,
    ) -> AnswerModel:
        model_target = answer_model or output_schema
        if model_target is None:
            raise ValueError("Must specify answer_model or output_schema")
        effective_system = system_prompt or system_message or TEAM_CHARTER

        raw_schema = model_target.model_json_schema()
        json_schema = make_all_fields_required(raw_schema)
        compact_schema_text = format_compact_schema(model_target)
        schema_note = (
            "\n\n<output_format>\n"
            "Return only one JSON object matching this schema:\n"
            f"{compact_schema_text}\n\n"
            "Return only the JSON object. Do not wrap the JSON in markdown code blocks or write any text before or after the JSON.\n"
            "</output_format>"
        )
        base_prompt = prompt + schema_note
        current_prompt = base_prompt
        problem = ""
        retries = getattr(self.model_config, "retries", LLM_REPAIR_RETRIES)
        total_tries = retries + 1
        for _ in range(total_tries):
            response_text = self._ask(
                agent_name,
                current_prompt,
                json_schema,
                system_message=effective_system,
                temperature=temperature,
            )
            try:
                return model_target.model_validate_json(extract_json_text(response_text))
            except ValidationError as validation_error:
                problem = str(validation_error)
                current_prompt = (
                    f"{base_prompt}\n\n"
                    "<repair>\n"
                    "Your previous answer could not be parsed or validated against the target schema.\n"
                    f"Previous answer:\n{response_text[:REPAIR_TEXT_CHARS]}\n\n"
                    f"Validation error:\n{problem[:REPAIR_TEXT_CHARS]}\n\n"
                    "Return only the corrected JSON object. Do not wrap the JSON in markdown code blocks or write any text before or after the JSON.\n"
                    "</repair>"
                )
                self.event_bus.emit(
                    EventType.WARNING, agent_name, "the model's JSON was not valid, asking again"
                )
        raise LlmUnavailableError(
            f"{agent_name} did not get valid JSON after {total_tries} tries. "
            f"Last problem: {problem}"
        )

    def _call_model(
        self,
        prompt: str,
        json_schema: dict | None,
        system_message: str | None = None,
        temperature: float | None = None,
    ) -> str:
        """The only method that talks to Ollama. Tests replace it with scripted answers."""
        send_schema = json_schema is not None and not is_cloud_direct(self.model_config.base_url)
        keyword_arguments = {"format": json_schema} if send_schema else {}
        chat = self.chat_model.bind(temperature=temperature) if temperature is not None else self.chat_model

        # Gemma 32B has no system role. Prepend role instructions directly into user prompt.
        if system_message:
            full_prompt = f"<role>\n{system_message.strip()}\n</role>\n\n{prompt}"
        else:
            full_prompt = prompt

        answer = chat.invoke(full_prompt, **keyword_arguments)
        return answer.content

    def _ask(
        self,
        agent_name: str,
        prompt: str,
        json_schema: dict | None,
        system_message: str | None = None,
        temperature: float | None = None,
    ) -> str:
        self.event_bus.emit(
            EventType.INFO,
            agent_name,
            f"asking {self.model_config.name} ({len(prompt)} characters)",
        )
        started_at = time.monotonic()
        response_text = ""
        error_text = ""
        try:
            max_attempts = len(RETRY_WAIT_SECONDS) + 1
            for attempt in range(max_attempts):
                try:
                    response_text = self._call_model(
                        prompt, json_schema, system_message, temperature=temperature
                    )
                    error_text = ""
                    break
                except ollama.ResponseError as error:
                    error_text = str(error)
                    if error.status_code in RETRYABLE_STATUS_CODES and attempt < len(
                        RETRY_WAIT_SECONDS
                    ):
                        time.sleep(RETRY_WAIT_SECONDS[attempt])
                        continue
                    if error.status_code == 401:
                        raise LlmUnavailableError(
                            "Ollama said 401: the API key is missing or wrong. "
                            "Set OLLAMA_API_KEY and open a new terminal."
                        ) from error
                    if error.status_code == 429:
                        raise LlmUnavailableError(
                            "Ollama said 429: a rate or usage limit was reached. "
                            "Wait, or check https://ollama.com/settings/usage."
                        ) from error
                    raise LlmUnavailableError(f"Ollama call failed: {error}") from error
                except (ConnectionError, httpx.HTTPError) as error:
                    error_text = str(error)
                    raise LlmUnavailableError(f"Ollama call failed: {error}") from error
        finally:
            seconds = round(time.monotonic() - started_at, 1)
            append_json_line(
                self.workspace.llm_calls_log_path,
                {
                    "time": datetime.now().isoformat(timespec="seconds"),
                    "agent": agent_name,
                    "seconds": seconds,
                    "prompt": prompt,
                    "response": response_text,
                    "error": error_text,
                },
            )
        self.event_bus.emit(EventType.INFO, agent_name, f"answer received in {seconds} seconds")
        return response_text