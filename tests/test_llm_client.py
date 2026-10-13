import json

import pytest
from fake_llm import FakeLlmClient
from fake_ollama_server import FakeOllamaServer
from pydantic import BaseModel

from mlagent.config import ModelConfig
from mlagent.events import EventBus, EventType
from mlagent.llm_client import (
    LlmClient,
    LlmUnavailableError,
    build_repair_prompt,
    extract_json_text,
)


class ReviewAnswer(BaseModel):
    focus: str
    score: float


def read_llm_call_log(workspace):
    lines = workspace.llm_calls_log_path.read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines]


def test_extract_json_text_removes_a_code_fence():
    assert extract_json_text('```json\n{"a": 1}\n```') == '{"a": 1}'
    assert extract_json_text('Here it is:\n```\n{"a": 1}\n```\nDone') == '{"a": 1}'
    assert extract_json_text('  {"a": 1}  ') == '{"a": 1}'


def test_repair_prompt_contains_the_original_prompt_the_answer_and_the_problem():
    repair_prompt = build_repair_prompt("ORIGINAL", "BAD ANSWER", "missing field score")

    assert "ORIGINAL" in repair_prompt
    assert "BAD ANSWER" in repair_prompt
    assert "missing field score" in repair_prompt


def test_ask_text_returns_the_answer_and_logs_the_call(workspace):
    llm_client = FakeLlmClient(["plain answer"], workspace)

    answer = llm_client.ask_text("Data Analyst", "question?")

    assert answer == "plain answer"
    logged_call = read_llm_call_log(workspace)[0]
    assert logged_call["agent"] == "Data Analyst"
    assert logged_call["prompt"] == "question?"
    assert logged_call["response"] == "plain answer"
    assert logged_call["error"] == ""
    assert llm_client.received_json_schemas == [None]


def test_ask_structured_returns_a_validated_object_and_sends_the_schema(workspace):
    llm_client = FakeLlmClient([{"focus": "model", "score": 0.8}], workspace)

    answer = llm_client.ask_structured("Reviewer", "decide", ReviewAnswer)

    assert answer == ReviewAnswer(focus="model", score=0.8)
    assert llm_client.received_json_schemas[0]["properties"].keys() == {"focus", "score"}


def test_ask_structured_accepts_json_inside_a_code_fence(workspace):
    llm_client = FakeLlmClient(['```json\n{"focus": "tune", "score": 1}\n```'], workspace)

    answer = llm_client.ask_structured("Reviewer", "decide", ReviewAnswer)

    assert answer.focus == "tune"


def test_invalid_json_is_repaired_with_one_more_question(workspace):
    event_bus = EventBus()
    received_events = []
    event_bus.subscribe(received_events.append)
    llm_client = FakeLlmClient(
        ["this is not json", {"focus": "features", "score": 0.5}], workspace, event_bus
    )

    answer = llm_client.ask_structured("Reviewer", "decide", ReviewAnswer)

    assert answer.focus == "features"
    assert len(llm_client.received_prompts) == 2
    assert "this is not json" in llm_client.received_prompts[1]
    assert "Problem" in llm_client.received_prompts[1]
    assert any(event.event_type == EventType.WARNING for event in received_events)


def test_a_wrong_field_type_is_also_repaired(workspace):
    llm_client = FakeLlmClient(
        [{"focus": "model", "score": "not a number"}, {"focus": "model", "score": 0.9}],
        workspace,
    )

    answer = llm_client.ask_structured("Reviewer", "decide", ReviewAnswer)

    assert answer.score == 0.9


def test_two_invalid_answers_raise_llm_unavailable(workspace):
    llm_client = FakeLlmClient(["bad", "still bad"], workspace)

    with pytest.raises(LlmUnavailableError, match="did not get valid JSON after 2 tries"):
        llm_client.ask_structured("Reviewer", "decide", ReviewAnswer)

    assert len(read_llm_call_log(workspace)) == 2


def test_scripted_answer_can_be_a_function_of_the_prompt(workspace):
    llm_client = FakeLlmClient([lambda prompt: f"echo: {prompt}"], workspace)

    assert llm_client.ask_text("Coder", "hi") == "echo: hi"


@pytest.fixture
def fake_ollama_server():
    server = FakeOllamaServer(installed_model_names=["gemma4:31b", "llama3:latest"])
    server.start()
    yield server
    server.stop()


def build_real_client(base_url, workspace, model_name="gemma4:31b"):
    model_config = ModelConfig(name=model_name, base_url=base_url)
    return LlmClient(model_config, workspace, EventBus())


def test_check_ready_passes_when_the_model_is_pulled(fake_ollama_server, workspace):
    build_real_client(fake_ollama_server.base_url, workspace).check_ready()


def test_check_ready_accepts_a_model_name_without_a_tag(fake_ollama_server, workspace):
    build_real_client(fake_ollama_server.base_url, workspace, "llama3").check_ready()


def test_check_ready_names_a_model_that_is_not_pulled(fake_ollama_server, workspace):
    llm_client = build_real_client(fake_ollama_server.base_url, workspace, "missing-model:7b")

    with pytest.raises(LlmUnavailableError, match="ollama pull missing-model:7b"):
        llm_client.check_ready()


def test_check_ready_says_when_ollama_is_not_running(workspace):
    llm_client = build_real_client("http://127.0.0.1:1", workspace)

    with pytest.raises(LlmUnavailableError, match="Cannot reach Ollama"):
        llm_client.check_ready()


def test_real_client_sends_context_length_temperature_and_prompt(fake_ollama_server, workspace):
    fake_ollama_server.chat_reply_text = "the reply"
    llm_client = build_real_client(fake_ollama_server.base_url, workspace)

    answer = llm_client.ask_text("Modeler", "train a model")

    sent_request = fake_ollama_server.chat_requests[0]
    assert answer == "the reply"
    assert sent_request["model"] == "gemma4:31b"
    assert sent_request["options"]["num_ctx"] == 16384
    assert sent_request["options"]["temperature"] == 0.2
    assert sent_request["messages"][0]["content"] == "train a model"
    assert not sent_request.get("format")


def test_real_client_sends_the_json_schema_as_format(fake_ollama_server, workspace):
    fake_ollama_server.chat_reply_text = '{"focus": "model", "score": 0.7}'
    llm_client = build_real_client(fake_ollama_server.base_url, workspace)

    answer = llm_client.ask_structured("Reviewer", "decide", ReviewAnswer)

    sent_request = fake_ollama_server.chat_requests[0]
    assert answer == ReviewAnswer(focus="model", score=0.7)
    assert set(sent_request["format"]["properties"]) == {"focus", "score"}


def test_a_model_error_from_ollama_becomes_llm_unavailable_and_is_logged(
    fake_ollama_server, workspace
):
    fake_ollama_server.chat_status_code = 404
    llm_client = build_real_client(fake_ollama_server.base_url, workspace)

    with pytest.raises(LlmUnavailableError, match="Ollama call failed"):
        llm_client.ask_text("Modeler", "train")

    assert "model not found" in read_llm_call_log(workspace)[0]["error"]


def test_ollama_stopping_during_a_run_becomes_llm_unavailable(fake_ollama_server, workspace):
    llm_client = build_real_client(fake_ollama_server.base_url, workspace)
    fake_ollama_server.stop()

    with pytest.raises(LlmUnavailableError):
        llm_client.ask_text("Modeler", "train")
