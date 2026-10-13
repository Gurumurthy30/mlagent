"""Talks to the local Ollama model. Every agent asks its questions through this one class.

Two ways to ask:
  ask_text        returns the model's answer as plain text
  ask_structured  returns the answer as a validated pydantic object. If the JSON is invalid,
                  the model is told what was wrong and asked once more.
Every call is written to logs/llm_calls.jsonl.
"""

import re
import time
from datetime import datetime
from typing import TypeVar

import httpx
import ollama
from langchain_ollama import ChatOllama
from pydantic import BaseModel, ValidationError

from mlagent.config import ModelConfig
from mlagent.events import EventBus, EventType
from mlagent.memory import Workspace, append_json_line
from mlagent.settings import LLM_CALL_TIMEOUT_SECONDS, LLM_REPAIR_RETRIES

AnswerModel = TypeVar("AnswerModel", bound=BaseModel)

REPAIR_TEXT_CHARS = 1500


class LlmUnavailableError(Exception):
    """Raised when Ollama cannot be used: not running, model missing, or no valid answer."""


def extract_json_text(response_text: str) -> str:
    """Remove a ```json fence when the model added one around its answer."""
    fenced_match = re.search(r"```(?:json)?\s*(.*?)```", response_text, re.DOTALL)
    json_text = fenced_match.group(1) if fenced_match else response_text
    return json_text.strip()


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
        """Make sure Ollama is running and the configured model is pulled."""
        model_name = self.model_config.name
        base_url = self.model_config.base_url
        try:
            installed_models = ollama.Client(host=base_url).list().models
        except (ConnectionError, httpx.HTTPError, ollama.ResponseError) as error:
            raise LlmUnavailableError(
                f"Cannot reach Ollama at {base_url}. Start Ollama and run again. ({error})"
            ) from error

        installed_names = {installed.model for installed in installed_models}
        wanted_name = model_name if ":" in model_name else f"{model_name}:latest"
        if wanted_name not in installed_names:
            raise LlmUnavailableError(
                f"Model '{model_name}' is not pulled in Ollama. Run: ollama pull {model_name}"
            )

    def ask_text(self, agent_name: str, prompt: str) -> str:
        return self._ask(agent_name, prompt, json_schema=None)

    def ask_structured(
        self, agent_name: str, prompt: str, answer_model: type[AnswerModel]
    ) -> AnswerModel:
        json_schema = answer_model.model_json_schema()
        current_prompt = prompt
        problem = ""
        total_tries = LLM_REPAIR_RETRIES + 1
        for _ in range(total_tries):
            response_text = self._ask(agent_name, current_prompt, json_schema)
            try:
                return answer_model.model_validate_json(extract_json_text(response_text))
            except ValidationError as validation_error:
                problem = str(validation_error)
                current_prompt = build_repair_prompt(prompt, response_text, problem)
                self.event_bus.emit(
                    EventType.WARNING, agent_name, "the model's JSON was not valid, asking again"
                )
        raise LlmUnavailableError(
            f"{agent_name} did not get valid JSON after {total_tries} tries. "
            f"Last problem: {problem}"
        )

    def _call_model(self, prompt: str, json_schema: dict | None) -> str:
        """The only method that talks to Ollama. Tests replace it with scripted answers."""
        keyword_arguments = {"format": json_schema} if json_schema else {}
        answer = self.chat_model.invoke(prompt, **keyword_arguments)
        return answer.content

    def _ask(self, agent_name: str, prompt: str, json_schema: dict | None) -> str:
        self.event_bus.emit(
            EventType.INFO,
            agent_name,
            f"asking {self.model_config.name} ({len(prompt)} characters)",
        )
        started_at = time.monotonic()
        response_text = ""
        error_text = ""
        try:
            response_text = self._call_model(prompt, json_schema)
        except (ConnectionError, httpx.HTTPError, ollama.ResponseError) as error:
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
