import json
import textwrap
import time

import psutil
import pytest

from mlagent.code_runner import (
    CodeRunner,
    extract_tagged_json,
    find_blocked_text,
)
from mlagent.settings import MAX_COMMAND_OUTPUT_CHARS, MAX_ERROR_TEXT_CHARS


@pytest.fixture
def work_folder(workspace):
    return workspace.experiment_folder("experiment_001")


def write_script(work_folder, source_code, name="script.py"):
    script_path = work_folder / name
    script_path.write_text(textwrap.dedent(source_code), encoding="utf-8")
    return script_path


def process_is_running(process_id):
    try:
        return psutil.Process(process_id).status() != psutil.STATUS_ZOMBIE
    except psutil.NoSuchProcess:
        return False


def test_successful_script_returns_its_output(workspace, work_folder):
    script_path = write_script(work_folder, "print('hello from script')")

    result = CodeRunner(workspace, timeout_seconds=30).run_python_script(script_path, work_folder)

    assert result.succeeded
    assert result.exit_code == 0
    assert result.stdout.strip() == "hello from script"
    assert result.error_text == ""


def test_failing_script_reports_the_error_at_the_end_of_the_traceback(workspace, work_folder):
    script_path = write_script(work_folder, "print(1 / 0)")

    result = CodeRunner(workspace, timeout_seconds=30).run_python_script(script_path, work_folder)

    assert not result.succeeded
    assert result.exit_code == 1
    assert "ZeroDivisionError" in result.error_text


def test_error_text_is_cut_to_its_limit_but_keeps_the_end(workspace, work_folder):
    script_path = write_script(
        work_folder,
        """
        import sys
        sys.stderr.write("A" * 10000 + "FINAL_ERROR_LINE")
        sys.exit(2)
        """,
    )

    result = CodeRunner(workspace, timeout_seconds=30).run_python_script(script_path, work_folder)

    assert len(result.error_text) == MAX_ERROR_TEXT_CHARS
    assert result.error_text.endswith("FINAL_ERROR_LINE")


def test_script_runs_inside_the_given_working_folder(workspace, work_folder):
    script_path = write_script(work_folder, "import os\nprint(os.getcwd())")

    result = CodeRunner(workspace, timeout_seconds=30).run_python_script(script_path, work_folder)

    assert result.stdout.strip() == str(work_folder)


def test_script_cannot_wait_for_keyboard_input(workspace, work_folder):
    script_path = write_script(work_folder, "input('type something: ')")

    result = CodeRunner(workspace, timeout_seconds=30).run_python_script(script_path, work_folder)

    assert not result.succeeded
    assert "EOFError" in result.error_text


def test_unicode_output_is_read_correctly(workspace, work_folder):
    script_path = write_script(work_folder, "print('caf\\u00e9 \\u2713')")

    result = CodeRunner(workspace, timeout_seconds=30).run_python_script(script_path, work_folder)

    assert result.stdout.strip() == "caf\u00e9 \u2713"


def test_very_long_output_is_cut_but_the_end_is_kept(workspace, work_folder):
    script_path = write_script(work_folder, "print('x' * 100000)\nprint('THE_END')")

    result = CodeRunner(workspace, timeout_seconds=30).run_python_script(script_path, work_folder)

    assert len(result.stdout) <= MAX_COMMAND_OUTPUT_CHARS
    assert result.stdout.strip().endswith("THE_END")


def test_timeout_stops_the_script_and_keeps_the_partial_output(workspace, work_folder):
    script_path = write_script(
        work_folder,
        """
        import time
        print('before sleeping', flush=True)
        time.sleep(120)
        """,
    )

    started_at = time.monotonic()
    result = CodeRunner(workspace, timeout_seconds=2).run_python_script(script_path, work_folder)

    assert result.timed_out
    assert not result.succeeded
    assert "Timed out after 2 seconds" in result.error_text
    assert "before sleeping" in result.stdout
    assert time.monotonic() - started_at < 20


def test_timeout_also_kills_processes_started_by_the_script(workspace, work_folder):
    process_id_file = work_folder / "child_process_id.txt"
    script_path = write_script(
        work_folder,
        f"""
        import subprocess, sys, time
        child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(120)"])
        open({str(process_id_file)!r}, "w").write(str(child.pid))
        time.sleep(120)
        """,
    )

    result = CodeRunner(workspace, timeout_seconds=3).run_python_script(script_path, work_folder)

    child_process_id = int(process_id_file.read_text())
    for _ in range(50):
        if not process_is_running(child_process_id):
            break
        time.sleep(0.1)
    assert result.timed_out
    assert not process_is_running(child_process_id)


def test_shell_command_returns_its_output(workspace, work_folder):
    result = CodeRunner(workspace, timeout_seconds=30).run_shell_command("echo hello", work_folder)

    assert result.succeeded
    assert result.stdout.strip() == "hello"


def test_failing_shell_command_reports_a_nonzero_exit_code(workspace, work_folder):
    result = CodeRunner(workspace, timeout_seconds=30).run_shell_command("exit 3", work_folder)

    assert not result.succeeded
    assert result.exit_code == 3


def test_blocked_shell_command_is_never_executed(workspace, work_folder, tmp_path):
    protected_file = tmp_path / "keep_me.txt"
    protected_file.write_text("important", encoding="utf-8")

    result = CodeRunner(workspace, timeout_seconds=30).run_shell_command(
        f"rm -rf {protected_file}", work_folder
    )

    assert protected_file.exists()
    assert result.exit_code == -1
    assert "Blocked" in result.error_text


def test_find_blocked_text_ignores_case_and_allows_safe_commands():
    assert find_blocked_text("SHUTDOWN now") == "shutdown"
    assert find_blocked_text("Remove-Item -Recurse C:\\data") == "remove-item -recurse"
    assert find_blocked_text("pip install lightgbm") is None


def test_missing_shell_program_returns_an_error_result(workspace, work_folder, monkeypatch):
    monkeypatch.setattr(
        "mlagent.code_runner.build_shell_command", lambda text: ["no_such_tool_xyz"]
    )

    result = CodeRunner(workspace, timeout_seconds=30).run_shell_command("echo hi", work_folder)

    assert result.exit_code == -1
    assert result.error_text != ""


def test_every_run_is_written_to_the_command_log(workspace, work_folder):
    code_runner = CodeRunner(workspace, timeout_seconds=30)
    script_path = write_script(work_folder, "print('logged')")

    code_runner.run_python_script(script_path, work_folder)
    code_runner.run_shell_command("echo second", work_folder)
    code_runner.run_shell_command("shutdown now", work_folder)

    lines = workspace.commands_log_path.read_text(encoding="utf-8").splitlines()
    log_records = [json.loads(line) for line in lines]
    assert [record["kind"] for record in log_records] == ["python", "shell", "shell"]
    assert log_records[0]["exit_code"] == 0
    assert "logged" in log_records[0]["stdout_end"]
    assert log_records[2]["exit_code"] == -1


def test_extract_tagged_json_returns_the_last_block():
    output_text = 'noise <RESULT>{"score": 1}</RESULT> more <RESULT>{"score": 2}</RESULT> end'

    assert extract_tagged_json(output_text, "RESULT") == {"score": 2}


def test_extract_tagged_json_reads_json_spread_over_several_lines():
    output_text = '<RESULT>\n{\n  "scores": [0.1, 0.2]\n}\n</RESULT>'

    assert extract_tagged_json(output_text, "RESULT") == {"scores": [0.1, 0.2]}


def test_extract_tagged_json_returns_none_when_it_cannot_use_the_block():
    assert extract_tagged_json("no tags here", "RESULT") is None
    assert extract_tagged_json("<RESULT>not json</RESULT>", "RESULT") is None
    assert extract_tagged_json("<RESULT>[1, 2]</RESULT>", "RESULT") is None
    assert extract_tagged_json('<OTHER>{"a": 1}</OTHER>', "RESULT") is None


def test_scripts_do_not_leave_bytecode_files_behind(workspace, work_folder):
    helper_path = write_script(work_folder, "VALUE = 1", name="helper_module.py")
    main_path = write_script(
        work_folder,
        "import sys\nimport helper_module\nprint(sys.dont_write_bytecode)",
        name="main.py",
    )

    result = CodeRunner(workspace, timeout_seconds=30).run_python_script(main_path, work_folder)

    assert helper_path.exists()
    assert result.stdout.strip() == "True"
    assert not (work_folder / "__pycache__").exists()
