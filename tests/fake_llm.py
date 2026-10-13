"""A scripted stand-in for the LLM, so tests never need Ollama.

It replaces only the single method that talks to Ollama. Parsing, the repair retry and
the call logging in LlmClient still run for real.
"""

import json

from pydantic import BaseModel

from mlagent.config import ModelConfig
from mlagent.events import EventBus
from mlagent.llm_client import LlmClient
from mlagent.memory import Workspace


class FakeLlmClient(LlmClient):
    """Gives back the scripted answers one by one.

    An answer can be a string, a dict, a pydantic object, or a function that receives
    the prompt and returns one of those.
    """

    def __init__(
        self,
        scripted_answers: list,
        workspace: Workspace,
        event_bus: EventBus | None = None,
    ) -> None:
        super().__init__(ModelConfig(), workspace, event_bus or EventBus())
        self.scripted_answers = list(scripted_answers)
        self.received_prompts: list[str] = []
        self.received_json_schemas: list[dict | None] = []

    def check_ready(self) -> None:
        return None

    def _call_model(self, prompt: str, json_schema: dict | None) -> str:
        self.received_prompts.append(prompt)
        self.received_json_schemas.append(json_schema)
        if not self.scripted_answers:
            raise AssertionError("FakeLlmClient has no scripted answer left.")
        next_answer = self.scripted_answers.pop(0)
        if callable(next_answer):
            next_answer = next_answer(prompt)
        if isinstance(next_answer, BaseModel):
            return next_answer.model_dump_json()
        if isinstance(next_answer, dict):
            return json.dumps(next_answer)
        return next_answer
