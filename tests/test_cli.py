import pytest
from fake_llm import FakeLlmClient
from full_run_answers import answers_before_the_modeler, answers_from_the_modeler
from test_modeling_pipeline_on_sample_project import write_sample_project

from mlagent import cli
from mlagent.cli import (
    METRICS_QUESTION,
    RESUME_QUESTION,
    TASK_QUESTION,
    decide_exit_code,
    main,
    start_command,
)
from mlagent.environment_probe import EnvironmentInfo, get_shell_name
from mlagent.llm_client import LlmUnavailableError
from mlagent.memory import Workspace, create_new_ledger, load_ledger, save_ledger
from mlagent.state import Evaluation, ExperimentRun, FinalResult


class FakeUi:
    """Stands in for the console: scripted answers in, recorded messages out."""

    def __init__(self, text_answers=(), choice_answers=()):
        self.text_answers = list(text_answers)
        self.choice_answers = list(choice_answers)
        self.questions: list[str] = []
        self.messages: list[str] = []
        self.events: list = []

    def handle_event(self, event):
        self.events.append(event)

    def ask_text(self, question):
        self.questions.append(question)
        if not self.text_answers:
            raise EOFError
        return self.text_answers.pop(0)

    def ask_choice(self, question, choices):
        self.questions.append(question)
        return self.choice_answers.pop(0)

    def show_message(self, text, style=""):
        self.messages.append(text)

    @property
    def all_messages(self) -> str:
        return "\n".join(self.messages)


class NotReadyClient(FakeLlmClient):
    def check_ready(self):
        raise LlmUnavailableError("Cannot reach Ollama at http://localhost:11434.")


def make_factory(scripted_answers=(), client_class=FakeLlmClient):
    def create_client(model_config, workspace, event_bus):
        return client_class(scripted_answers, workspace, event_bus)

    return create_client


def fake_probe():
    return EnvironmentInfo(
        operating_system="TestOS 1",
        shell_name=get_shell_name(),
        python_version="3.12.0",
        cpu_count=4,
        ram_total_gb=16.0,
        ram_free_gb=8.0,
    )


def finished_ledger(**changes):
    ledger = create_new_ledger("Predict churn.", "macro F1")
    ledger.status = "done"
    ledger.phase = "done"
    ledger.stop_reason = "budget"
    for name, value in changes.items():
        setattr(ledger, name, value)
    return ledger


@pytest.fixture
def pipeline_calls(monkeypatch):
    """Replaces the pipeline with a stub that records the ledger it receives."""
    received_ledgers = []

    def stub_pipeline(ledger, agent_context):
        received_ledgers.append(ledger)
        return finished_ledger(user_task=ledger.user_task)

    monkeypatch.setattr(cli, "run_pipeline", stub_pipeline)
    return received_ledgers


def run_start(project_folder, ui, **options):
    options.setdefault("create_llm_client", make_factory())
    options.setdefault("probe", fake_probe)
    return start_command(project_folder, ui, **options)


def test_a_new_run_asks_for_the_task_and_the_metrics_and_saves_them(tmp_path, pipeline_calls):
    ui = FakeUi(text_answers=["Predict which customers churn.", "  macro F1 mainly, also AUC  "])

    exit_code = run_start(tmp_path, ui)

    workspace = Workspace(tmp_path)
    assert exit_code == 0
    assert ui.questions == [TASK_QUESTION, METRICS_QUESTION]
    assert pipeline_calls[0].user_task == "Predict which customers churn."
    assert pipeline_calls[0].user_metrics_text == "macro F1 mainly, also AUC"
    assert load_ledger(workspace).user_task == "Predict which customers churn."
    assert workspace.config_path.exists()
    assert workspace.environment_path.exists()
    assert f"Created {workspace.config_path}" in ui.all_messages
    assert "OS: TestOS 1" in ui.all_messages


def test_an_empty_metrics_answer_is_allowed_and_an_empty_task_is_asked_again(
    tmp_path, pipeline_calls
):
    ui = FakeUi(text_answers=["", "   ", "Cluster these customers.", ""])

    run_start(tmp_path, ui)

    assert ui.questions.count(TASK_QUESTION) == 3
    assert pipeline_calls[0].user_task == "Cluster these customers."
    assert pipeline_calls[0].user_metrics_text == ""
    assert "Please type something." in ui.all_messages


def test_closed_input_stops_with_a_message(tmp_path, pipeline_calls):
    ui = FakeUi(text_answers=[])

    exit_code = run_start(tmp_path, ui)

    assert exit_code == 2
    assert "No task was given" in ui.all_messages
    assert pipeline_calls == []


def test_ollama_that_is_not_ready_stops_before_asking_anything(tmp_path, pipeline_calls):
    ui = FakeUi(text_answers=["task", ""])

    exit_code = run_start(tmp_path, ui, create_llm_client=make_factory(client_class=NotReadyClient))

    assert exit_code == 2
    assert "Cannot reach Ollama" in ui.all_messages
    assert ui.questions == []
    assert pipeline_calls == []


def test_a_broken_config_file_stops_with_its_message(tmp_path, pipeline_calls):
    workspace = Workspace(tmp_path)
    workspace.create_folders()
    workspace.config_path.write_text("limits:\n  max_experiment_loop: 3\n", encoding="utf-8")
    ui = FakeUi(text_answers=["task", ""])

    exit_code = run_start(tmp_path, ui)

    assert exit_code == 2
    assert "Cannot use" in ui.all_messages
    assert pipeline_calls == []


def test_choosing_fresh_keeps_the_old_workspace_and_the_edited_config(tmp_path, pipeline_calls):
    workspace = Workspace(tmp_path)
    workspace.create_folders()
    workspace.config_path.write_text("limits:\n  max_experiment_loops: 7\n", encoding="utf-8")
    save_ledger(workspace, create_new_ledger("The old task.", "old metric"))
    ui = FakeUi(text_answers=["The new task.", ""], choice_answers=["fresh"])

    exit_code = run_start(tmp_path, ui)

    archives = [
        path for path in tmp_path.iterdir() if path.name.startswith("mlagent_workspace_old_")
    ]
    assert exit_code == 0
    assert ui.questions[0] == RESUME_QUESTION
    assert len(archives) == 1
    assert load_ledger(Workspace(archives[0].parent)).user_task == "The new task."
    assert "max_experiment_loops: 7" in workspace.config_path.read_text(encoding="utf-8")
    assert "The old workspace was kept as" in ui.all_messages
    assert pipeline_calls[0].user_task == "The new task."


def test_choosing_resume_does_not_ask_for_the_task_again(tmp_path, pipeline_calls):
    workspace = Workspace(tmp_path)
    workspace.create_folders()
    saved_ledger = create_new_ledger("The saved task.", "saved metric")
    saved_ledger.phase = "ml_lead"
    save_ledger(workspace, saved_ledger)
    ui = FakeUi(choice_answers=["resume"])

    exit_code = run_start(tmp_path, ui)

    assert exit_code == 0
    assert ui.questions == [RESUME_QUESTION]
    assert pipeline_calls[0].user_task == "The saved task."
    assert pipeline_calls[0].phase == "ml_lead"
    assert "Resuming the saved run at: ml_lead." in ui.all_messages


def test_resuming_a_failed_run_clears_the_failure_and_keeps_the_phase(tmp_path, pipeline_calls):
    workspace = Workspace(tmp_path)
    workspace.create_folders()
    failed_ledger = create_new_ledger("task", "metric")
    failed_ledger.phase = "data_analyst"
    failed_ledger.status = "failed"
    failed_ledger.stop_reason = "llm_unavailable"
    failed_ledger.error_message = "Ollama stopped"
    save_ledger(workspace, failed_ledger)
    ui = FakeUi(choice_answers=["resume"])

    run_start(tmp_path, ui)

    resumed_ledger = pipeline_calls[0]
    assert resumed_ledger.status == "running"
    assert resumed_ledger.stop_reason is None
    assert resumed_ledger.error_message == ""
    assert resumed_ledger.phase == "data_analyst"


def test_resuming_a_finished_run_shows_its_result_without_running_again(tmp_path, pipeline_calls):
    workspace = Workspace(tmp_path)
    workspace.create_folders()
    finished = finished_ledger(final=FinalResult(report_path="/work/report.md"))
    save_ledger(workspace, finished)
    ui = FakeUi(choice_answers=["resume"])

    exit_code = run_start(tmp_path, ui)

    assert exit_code == 0
    assert pipeline_calls == []
    assert "This run already finished." in ui.all_messages
    assert "Report: /work/report.md" in ui.all_messages


def test_a_damaged_saved_ledger_asks_the_user_to_start_fresh(tmp_path, pipeline_calls):
    workspace = Workspace(tmp_path)
    workspace.create_folders()
    workspace.ledger_path.write_text("damaged", encoding="utf-8")
    ui = FakeUi(choice_answers=["resume"])

    exit_code = run_start(tmp_path, ui)

    assert exit_code == 2
    assert "Start again and choose fresh" in ui.all_messages
    assert pipeline_calls == []


def test_ctrl_c_stops_politely_and_says_how_to_resume(tmp_path, monkeypatch):
    def interrupted_pipeline(ledger, agent_context):
        raise KeyboardInterrupt

    monkeypatch.setattr(cli, "run_pipeline", interrupted_pipeline)
    ui = FakeUi(text_answers=["task", ""])

    exit_code = run_start(tmp_path, ui)

    assert exit_code == 130
    assert "choose resume to continue" in ui.all_messages


def test_an_unexpected_error_points_to_the_logs(tmp_path, monkeypatch):
    def broken_pipeline(ledger, agent_context):
        raise ValueError("a bug")

    monkeypatch.setattr(cli, "run_pipeline", broken_pipeline)
    ui = FakeUi(text_answers=["task", ""])

    exit_code = run_start(tmp_path, ui)

    assert exit_code == 1
    assert f"See {Workspace(tmp_path).logs_folder}" in ui.all_messages


def test_the_summary_names_the_best_result_the_predictions_and_the_report(tmp_path, monkeypatch):
    def pipeline_with_result(ledger, agent_context):
        result = finished_ledger(user_task=ledger.user_task)
        result.plan.primary_metric = "macro_f1"
        result.runs = [
            ExperimentRun(
                experiment_id="experiment_003",
                model_family="lightgbm",
                score_mean=0.8123,
                validation_scores=[0.81],
            )
        ]
        result.evaluations = [Evaluation(experiment_id="experiment_003", verdict="approve")]
        result.final = FinalResult(
            predictions_path="/work/report/predictions.csv",
            check_messages=["All delivery checks passed."],
            report_path="/work/report/report.md",
        )
        return result

    monkeypatch.setattr(cli, "run_pipeline", pipeline_with_result)
    ui = FakeUi(text_answers=["task", ""])

    exit_code = run_start(tmp_path, ui)

    assert exit_code == 0
    assert "Finished. Stop reason: budget." in ui.all_messages
    assert "Best result: experiment_003 (lightgbm), macro_f1 0.8123" in ui.all_messages
    assert "Predictions: /work/report/predictions.csv" in ui.all_messages
    assert "  All delivery checks passed." in ui.all_messages
    assert "Report: /work/report/report.md" in ui.all_messages


@pytest.mark.parametrize(
    ("status", "stop_reason", "expected_code"),
    [
        ("done", "budget", 0),
        ("done", "target", 0),
        ("done", "llm_unavailable", 1),
        ("done", "agent_failed", 1),
        ("failed", "agent_failed", 1),
    ],
)
def test_the_exit_code_reflects_how_the_run_ended(status, stop_reason, expected_code):
    ledger = finished_ledger(status=status, stop_reason=stop_reason)

    assert decide_exit_code(ledger) == expected_code


def test_main_without_a_command_prints_the_help(capsys):
    assert main([]) == 2
    assert "start" in capsys.readouterr().out


def test_main_prints_the_version(capsys):
    with pytest.raises(SystemExit) as exit_info:
        main(["--version"])

    assert exit_info.value.code == 0
    assert capsys.readouterr().out.startswith("mlagent 0.")


def test_main_start_runs_the_start_command_in_the_current_folder(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    received_folders = []

    def fake_start_command(project_folder, ui):
        received_folders.append(project_folder)
        return 0

    monkeypatch.setattr(cli, "start_command", fake_start_command)

    assert main(["start"]) == 0
    assert received_folders == [tmp_path]


def write_short_config(workspace):
    workspace.create_folders()
    workspace.config_path.write_text(
        "limits:\n  max_experiment_loops: 2\n  max_eda_loops: 1\n", encoding="utf-8"
    )


def test_a_whole_run_through_the_command_with_real_scripts(tmp_path):
    workspace = Workspace(tmp_path)
    write_short_config(workspace)
    write_sample_project(tmp_path)
    answers = [*answers_before_the_modeler(workspace), *answers_from_the_modeler(workspace)]
    ui = FakeUi(text_answers=["Predict which customers churn.", "macro F1 mainly, also AUC"])

    exit_code = run_start(tmp_path, ui, create_llm_client=make_factory(answers))

    final_ledger = load_ledger(workspace)
    assert exit_code == 0
    assert final_ledger.status == "done"
    assert final_ledger.user_metrics_text == "macro F1 mainly, also AUC"
    assert workspace.report_path.exists()
    assert "Finished. Stop reason: budget." in ui.all_messages
    assert f"Report: {workspace.report_path}" in ui.all_messages
    assert any(event.message.startswith("round 1 of 2") for event in ui.events)
    run_log_text = workspace.run_log_path.read_text(encoding="utf-8")
    assert "Data Engineer" in run_log_text and "Submitter" in run_log_text


def test_an_interrupted_run_resumes_where_it_stopped(tmp_path):
    workspace = Workspace(tmp_path)
    write_short_config(workspace)
    write_sample_project(tmp_path)

    def interrupt_at_the_modeler(prompt):
        raise KeyboardInterrupt

    first_answers = [*answers_before_the_modeler(workspace), interrupt_at_the_modeler]
    first_ui = FakeUi(text_answers=["Predict which customers churn.", "macro F1"])

    first_exit_code = run_start(tmp_path, first_ui, create_llm_client=make_factory(first_answers))

    saved_ledger = load_ledger(workspace)
    assert first_exit_code == 130
    assert saved_ledger.status == "running"
    assert saved_ledger.phase == "experiment_loop"
    assert saved_ledger.plan.splits_path != ""
    assert saved_ledger.current_feature_version == 1
    assert saved_ledger.runs == []

    second_answers = answers_from_the_modeler(workspace)
    second_ui = FakeUi(choice_answers=["resume"])

    second_exit_code = run_start(
        tmp_path, second_ui, create_llm_client=make_factory(second_answers)
    )

    final_ledger = load_ledger(workspace)
    assert second_exit_code == 0
    assert second_ui.questions == [RESUME_QUESTION]
    assert "Resuming the saved run at: experiment_loop." in second_ui.all_messages
    assert final_ledger.status == "done"
    assert final_ledger.user_task == "Predict which customers churn."
    assert [run.experiment_id for run in final_ledger.runs] == ["experiment_001", "experiment_002"]
    assert final_ledger.current_feature_version == 1
    assert workspace.report_path.exists()
