"""Talks to the Ollama model. Every agent asks questions through this class.

Two ways to ask:
  ask_text        returns the model's answer as plain text
  ask_structured  returns the answer as a validated pydantic object. If the JSON is invalid,
                  the model is told what was wrong and asked once more.
Every call is written to logs/llm_calls.jsonl.
"""

import json
import os
import re
import time
from datetime import datetime
from typing import TypeVar
from urllib.parse import urlparse

import httpx
import ollama
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama
from pydantic import BaseModel, ValidationError

from mlagent.config import ModelConfig
from mlagent.events import EventBus, EventType
from mlagent.memory import Workspace, append_json_line
from mlagent.settings import (
    LLM_CALL_TIMEOUT_SECONDS,
    LLM_REPAIR_RETRIES,
    RETRY_WAIT_SECONDS,
    RETRYABLE_STATUS_CODES,
)

TEAM_CHARTER = """\
You are an expert member of an autonomous ML engineering team.
Core team rules:
- The project folder is read-only. Never change, move, or delete the user's raw files.
- Write every output file inside the workspace folder.
- Every score or metric reported must come directly from a script's executed output. Never invent numbers.
- Start simple with solid baselines. Add complexity only when empirical validation improves."""

AnswerModel = TypeVar("AnswerModel", bound=BaseModel)

REPAIR_TEXT_CHARS = 1500


class LlmUnavailableError(Exception):
    """Raised when Ollama cannot be used: not running, model missing, or no valid answer."""


def is_cloud_direct(base_url: str) -> bool:
    return urlparse(base_url).hostname == "ollama.com"


def extract_json_text(response_text: str) -> str:
    """Extract JSON from a code fence or surrounding text."""
    fenced_match = re.search(r"```(?:json)?\s*(.*?)```", response_text, re.DOTALL)
    candidate_text = fenced_match.group(1) if fenced_match else response_text
    start_brace = candidate_text.find("{")
    end_brace = candidate_text.rfind("}")
    if start_brace != -1 and end_brace != -1 and start_brace < end_brace:
        return candidate_text[start_brace : end_brace + 1].strip()
    return candidate_text.strip()


def build_repair_prompt(original_prompt: str, previous_answer: str, problem: str) -> str:
    return (
        f"{original_prompt}\n\n"
        "Your previous answer could not be used.\n"
        f"Previous answer:\n{previous_answer[:REPAIR_TEXT_CHARS]}\n\n"
        f"Problem:\n{problem[:REPAIR_TEXT_CHARS]}\n\n"
        "Answer again with only the corrected JSON."
    )


class LlmClient:
    def __init__(
        self, model_config: ModelConfig, workspace: Workspace, event_bus: EventBus
    ) -> None:
        self.model_config = model_config
        self.workspace = workspace
        self.event_bus = event_bus
        self.chat_model = ChatOllama(
            model=model_config.name,
            base_url=model_config.base_url,
            temperature=model_config.temperature,
            num_ctx=model_config.context_length,
            client_kwargs={"timeout": LLM_CALL_TIMEOUT_SECONDS},
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

        installed_names = {installed.model for installed in installed_models}
        wanted_name = model_name if ":" in model_name else f"{model_name}:latest"
        if wanted_name not in installed_names:
            if is_cloud_direct(base_url):
                raise LlmUnavailableError(
                    f"Model '{model_name}' is not available on Ollama Cloud. The name must "
                    "come from https://ollama.com/api/tags, for example gemma4:31b without "
                    "a -cloud suffix."
                )
            raise LlmUnavailableError(
                f"Model '{model_name}' is not pulled in Ollama. Run: ollama pull {model_name}"
            )

    def ask_text(self, agent_name: str, prompt: str, system_message: str | None = None) -> str:
        return self._ask(agent_name, prompt, json_schema=None, system_message=system_message)

    def ask_structured(
        self,
        agent_name: str,
        prompt: str,
        answer_model: type[AnswerModel],
        system_message: str | None = TEAM_CHARTER,
    ) -> AnswerModel:
        json_schema = answer_model.model_json_schema()
        schema_note = (
            "\n\nAnswer with only one JSON object and no other text. It must follow this JSON "
            f"schema:\n{json.dumps(json_schema)}"
        )
        base_prompt = prompt + schema_note
        current_prompt = base_prompt
        problem = ""
        total_tries = LLM_REPAIR_RETRIES + 1
        for _ in range(total_tries):
            response_text = self._ask(
                agent_name, current_prompt, json_schema, system_message=system_message
            )
            try:
                return answer_model.model_validate_json(extract_json_text(response_text))
            except ValidationError as validation_error:
                problem = str(validation_error)
                current_prompt = build_repair_prompt(base_prompt, response_text, problem)
                self.event_bus.emit(
                    EventType.WARNING, agent_name, "the model's JSON was not valid, asking again"
                )
        raise LlmUnavailableError(
            f"{agent_name} did not get valid JSON after {total_tries} tries. "
            f"Last problem: {problem}"
        )

    def _call_model(
        self, prompt: str, json_schema: dict | None, system_message: str | None = None
    ) -> str:
        """The only method that talks to Ollama. Tests replace it with scripted answers."""
        send_schema = json_schema is not None and not is_cloud_direct(self.model_config.base_url)
        keyword_arguments = {"format": json_schema} if send_schema else {}
        if system_message:
            messages = [SystemMessage(content=system_message), HumanMessage(content=prompt)]
            answer = self.chat_model.invoke(messages, **keyword_arguments)
        else:
            answer = self.chat_model.invoke(prompt, **keyword_arguments)
        return answer.content

    def _ask(
        self,
        agent_name: str,
        prompt: str,
        json_schema: dict | None,
        system_message: str | None = None,
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
                    response_text = self._call_model(prompt, json_schema, system_message)
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
