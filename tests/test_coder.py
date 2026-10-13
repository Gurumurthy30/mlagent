import json
import textwrap

from mlagent.agents.coder import (
    collect_lookup_names,
    extract_code_blocks,
    last_error_line,
    run_coder,
)
from mlagent.events import EventType
from mlagent.memory import create_new_ledger
from mlagent.state import ExperimentRun

SUCCESS_SCRIPT = """
print('training done')
print('<RESULT>{"score": 0.9}</RESULT>')
"""


def python_answer(script_text: str) -> str:
    return "Here is the script.\n```python\n" + textwrap.dedent(script_text).strip() + "\n```"


def failing_answer(error_message: str) -> str:
    return python_answer(f"raise ValueError({error_message!r})")


def run_test_task(agent_context, work_folder, ledger=None):
    return run_coder(
        task_text="Train a baseline model on the training table.",
        result_tag="RESULT",
        result_keys_text='- "score": float, the validation score',
        working_folder=work_folder,
        script_name="train.py",
        caller_key="modeler",
        ledger=ledger or create_new_ledger("predict churn", "macro F1"),
        agent_context=agent_context,
    )


def test_first_attempt_success_returns_the_result_and_saves_the_script(
    make_agent_context, workspace
):
    agent_context = make_agent_context([python_answer(SUCCESS_SCRIPT)])
    work_folder = workspace.experiment_folder("experiment_001")

    result = run_test_task(agent_context, work_folder)

    assert result.succeeded
    assert result.result_data == {"score": 0.9}
    assert result.attempts_used == 1
    assert result.error_text == ""
    assert "training done" in result.stdout
    assert result.script_path == work_folder / "train.py"
    assert "<RESULT>" in result.script_path.read_text(encoding="utf-8")


def test_the_prompt_tells_the_coder_its_task_limits_and_result_format(
    make_agent_context, workspace
):
    agent_context = make_agent_context([python_answer(SUCCESS_SCRIPT)])

    run_test_task(agent_context, workspace.experiment_folder("experiment_001"))

    prompt = agent_context.llm_client.received_prompts[0]
    assert "TASK FROM THE Modeler" in prompt
    assert "Train a baseline model on the training table." in prompt
    assert "ATTEMPT 1 OF 4." in prompt
    assert "This is your first attempt." in prompt
    assert "Time limit for one script run: 30 minutes" in prompt
    assert "<RESULT>JSON</RESULT>" in prompt
    assert '- "score": float, the validation score' in prompt
    assert "the random seed 42" in prompt
    assert "You have 3 lookups left" in prompt
    assert "(nothing in memory yet)" in prompt
    assert "{" not in prompt.replace("{}", "")


def test_the_plan_and_recent_crashes_reach_the_coder_through_its_digest(
    make_agent_context, workspace
):
    ledger = create_new_ledger("predict churn", "macro F1")
    ledger.plan.primary_metric = "macro_f1"
    ledger.plan.validation_scheme = "stratified_kfold"
    ledger.runs.append(ExperimentRun(experiment_id="experiment_001", error="CUDA out of memory"))
    agent_context = make_agent_context([python_answer(SUCCESS_SCRIPT)])

    run_test_task(agent_context, workspace.experiment_folder("experiment_002"), ledger)

    prompt = agent_context.llm_client.received_prompts[0]
    assert "primary metric: macro_f1" in prompt
    assert "stratified_kfold" in prompt
    assert "CUDA out of memory" in prompt


def test_a_failed_script_is_fixed_on_the_next_attempt_with_the_error_shown(
    make_agent_context, workspace
):
    agent_context = make_agent_context(
        [python_answer("print(1 / 0)"), python_answer(SUCCESS_SCRIPT)]
    )

    result = run_test_task(agent_context, workspace.experiment_folder("experiment_001"))

    second_prompt = agent_context.llm_client.received_prompts[1]
    assert result.succeeded
    assert result.attempts_used == 2
    assert "ATTEMPT 2 OF 4." in second_prompt
    assert "Attempt 1 failed." in second_prompt
    assert "ZeroDivisionError" in second_prompt


def test_a_script_that_prints_no_result_line_is_sent_back(make_agent_context, workspace):
    agent_context = make_agent_context(
        [python_answer("print('finished but forgot the result')"), python_answer(SUCCESS_SCRIPT)]
    )

    result = run_test_task(agent_context, workspace.experiment_folder("experiment_001"))

    second_prompt = agent_context.llm_client.received_prompts[1]
    assert result.succeeded
    assert "did not print a valid <RESULT>JSON</RESULT> line" in second_prompt
    assert "finished but forgot the result" in second_prompt


def test_the_coder_gives_up_after_the_last_attempt(make_agent_context, workspace):
    agent_context = make_agent_context(
        [failing_answer(f"error number {number}") for number in range(1, 5)]
    )

    result = run_test_task(agent_context, workspace.experiment_folder("experiment_001"))

    assert not result.succeeded
    assert result.attempts_used == 4
    assert result.result_data == {}
    assert "error number 4" in result.error_text
    assert len(agent_context.llm_client.received_prompts) == 4


def test_the_same_error_twice_in_a_row_stops_early(make_agent_context, workspace):
    agent_context = make_agent_context(
        [failing_answer("same problem"), failing_answer("same problem")]
    )

    result = run_test_task(agent_context, workspace.experiment_folder("experiment_001"))

    assert not result.succeeded
    assert result.attempts_used == 2
    assert "same problem" in result.error_text


def test_an_answer_without_code_counts_as_an_attempt(make_agent_context, workspace):
    agent_context = make_agent_context(["I cannot do that.", python_answer(SUCCESS_SCRIPT)])

    result = run_test_task(agent_context, workspace.experiment_folder("experiment_001"))

    assert result.succeeded
    assert result.attempts_used == 2
    assert "0 python blocks" in agent_context.llm_client.received_prompts[1]


def test_two_python_blocks_are_refused(make_agent_context, workspace):
    two_scripts = python_answer("print(1)") + "\n" + python_answer("print(2)")
    agent_context = make_agent_context([two_scripts, python_answer(SUCCESS_SCRIPT)])

    result = run_test_task(agent_context, workspace.experiment_folder("experiment_001"))

    assert result.succeeded
    assert "2 python blocks" in agent_context.llm_client.received_prompts[1]


def test_a_shell_block_runs_before_the_script_and_both_are_logged(make_agent_context, workspace):
    answer = "```bash\necho preparing\n```\n" + python_answer(SUCCESS_SCRIPT)
    agent_context = make_agent_context([answer])

    result = run_test_task(agent_context, workspace.experiment_folder("experiment_001"))

    log_lines = workspace.commands_log_path.read_text(encoding="utf-8").splitlines()
    log_records = [json.loads(line) for line in log_lines]
    assert result.succeeded
    assert result.attempts_used == 1
    assert [record["kind"] for record in log_records] == ["shell", "python"]


def test_a_failing_shell_block_stops_before_the_script_is_run(make_agent_context, workspace):
    work_folder = workspace.experiment_folder("experiment_001")
    first_answer = "```bash\nexit 5\n```\n" + python_answer(SUCCESS_SCRIPT)
    agent_context = make_agent_context([first_answer, python_answer(SUCCESS_SCRIPT)])

    result = run_test_task(agent_context, work_folder)

    assert result.succeeded
    assert result.attempts_used == 2
    assert (
        "A shell command failed, so the script was not run."
        in (agent_context.llm_client.received_prompts[1])
    )


def test_a_blocked_shell_command_is_reported_back_and_not_run(
    make_agent_context, workspace, tmp_path
):
    protected_file = tmp_path / "keep_me.txt"
    protected_file.write_text("important", encoding="utf-8")
    first_answer = f"```bash\nrm -rf {protected_file}\n```\n" + python_answer(SUCCESS_SCRIPT)
    agent_context = make_agent_context([first_answer, python_answer(SUCCESS_SCRIPT)])

    run_test_task(agent_context, workspace.experiment_folder("experiment_001"))

    assert protected_file.exists()
    assert "Blocked" in agent_context.llm_client.received_prompts[1]


def test_a_lookup_answer_does_not_use_an_attempt(make_agent_context, workspace):
    agent_context = make_agent_context(
        ["```lookup\njson.dumps\n```", python_answer(SUCCESS_SCRIPT)]
    )

    result = run_test_task(agent_context, workspace.experiment_folder("experiment_001"))

    second_prompt = agent_context.llm_client.received_prompts[1]
    assert result.succeeded
    assert result.attempts_used == 1
    assert "ATTEMPT 1 OF 4." in second_prompt
    assert "Lookup 'json.dumps':" in second_prompt
    assert "You have 2 lookups left" in second_prompt


def test_an_invalid_lookup_name_is_refused_without_running_anything(make_agent_context, workspace):
    agent_context = make_agent_context(
        ["```lookup\nos; echo hacked\n```", python_answer(SUCCESS_SCRIPT)]
    )

    run_test_task(agent_context, workspace.experiment_folder("experiment_001"))

    assert "not a valid dotted name" in agent_context.llm_client.received_prompts[1]
    commands_logged = workspace.commands_log_path.read_text(encoding="utf-8")
    assert "hacked" not in commands_logged


def test_lookups_are_limited_and_extra_lookups_count_as_an_attempt(make_agent_context, workspace):
    lookup_answers = [
        f"```lookup\n{name}\n```"
        for name in ["json.dumps", "json.loads", "os.getcwd", "os.listdir"]
    ]
    agent_context = make_agent_context([*lookup_answers, python_answer(SUCCESS_SCRIPT)])

    result = run_test_task(agent_context, workspace.experiment_folder("experiment_001"))

    fifth_prompt = agent_context.llm_client.received_prompts[4]
    assert result.succeeded
    assert result.attempts_used == 2
    assert "No lookups are left." in fifth_prompt
    assert "ATTEMPT 2 OF 4." in fifth_prompt


def test_a_timeout_tells_the_model_to_make_the_script_faster(make_agent_context, workspace):
    slow_script = "import time\ntime.sleep(60)"
    agent_context = make_agent_context(
        [python_answer(slow_script), python_answer(SUCCESS_SCRIPT)], timeout_seconds=1
    )

    result = run_test_task(agent_context, workspace.experiment_folder("experiment_001"))

    second_prompt = agent_context.llm_client.received_prompts[1]
    assert result.succeeded
    assert "Timed out after 1 seconds" in second_prompt
    assert "Make it faster" in second_prompt


def test_events_show_the_start_and_the_end_of_the_task(make_agent_context, workspace):
    agent_context = make_agent_context([python_answer(SUCCESS_SCRIPT)])
    received_events = []
    agent_context.event_bus.subscribe(received_events.append)

    run_test_task(agent_context, workspace.experiment_folder("experiment_001"))

    coder_events = [event for event in received_events if event.agent_name == "Coder"]
    assert coder_events[0].event_type == EventType.AGENT_STARTED
    assert "task from Modeler" in coder_events[0].message
    assert coder_events[-1].event_type == EventType.AGENT_FINISHED
    assert "done after 1 attempt" in coder_events[-1].message


def test_extract_code_blocks_reads_labels_and_treats_no_label_as_python():
    answer_text = "Intro\n```Python\nprint(1)\n```\nthen\n```\nprint(2)\n```\n```ps1\nGet-Date\n```"

    code_blocks = extract_code_blocks(answer_text)

    assert [block.language for block in code_blocks] == ["python", "", "ps1"]
    assert code_blocks[0].code.strip() == "print(1)"


def test_collect_lookup_names_reads_one_name_per_line():
    code_blocks = extract_code_blocks("```lookup\njson.dumps\n\n  os.getcwd  \n```")

    assert collect_lookup_names(code_blocks) == ["json.dumps", "os.getcwd"]


def test_last_error_line_ignores_empty_lines():
    assert last_error_line("Traceback...\n  File x\nValueError: bad\n\n") == "ValueError: bad"
    assert last_error_line("") == ""
