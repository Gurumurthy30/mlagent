import pandas as pd
import pytest

from mlagent.deliverable_checks import check_predictions_file, find_sample_submission
from mlagent.state import FileInfo, ProjectBrief


@pytest.fixture
def project_folder(tmp_path):
    project = tmp_path / "project"
    project.mkdir()
    pd.DataFrame({"id": [1, 2, 3], "churn": [0, 0, 0]}).to_csv(
        project / "sample_submission.csv", index=False
    )
    return project


@pytest.fixture
def brief():
    return ProjectBrief(
        files=[
            FileInfo(path="train.csv", role="train"),
            FileInfo(path="sample_submission.csv", role="sample_submission"),
        ]
    )


def write_predictions(tmp_path, data, name="predictions.csv"):
    predictions_path = tmp_path / name
    pd.DataFrame(data).to_csv(predictions_path, index=False)
    return predictions_path


def test_a_file_that_matches_the_sample_submission_passes(tmp_path, project_folder, brief):
    predictions_path = write_predictions(tmp_path, {"id": [3, 1, 2], "churn": [1, 0, 1]})

    assert check_predictions_file(predictions_path, brief, project_folder) == []


@pytest.mark.parametrize(
    ("data", "expected_problem"),
    [
        ({"id": [1, 2, 3], "label": [0, 1, 0]}, "columns differ"),
        ({"id": [1, 2], "churn": [0, 1]}, "row count differs"),
        ({"id": [1, 2, 9], "churn": [0, 1, 0]}, "1 missing, 1 extra"),
        ({"id": [1, 2, 3], "churn": [0, None, 1]}, "Missing values in the columns: ['churn']"),
        ({"id": [1, 2, 3], "churn": ["yes", "no", "yes"]}, "should be numeric"),
    ],
)
def test_each_kind_of_mismatch_is_named(tmp_path, project_folder, brief, data, expected_problem):
    predictions_path = write_predictions(tmp_path, data)

    problems = check_predictions_file(predictions_path, brief, project_folder)

    assert any(expected_problem in problem for problem in problems)


def test_without_a_sample_submission_only_basic_checks_run(tmp_path, project_folder):
    empty_brief = ProjectBrief()
    good_path = write_predictions(tmp_path, {"anything": [1, 2], "goes": [3, 4]})
    missing_value_path = write_predictions(tmp_path, {"a": [1, None]}, name="other.csv")

    assert check_predictions_file(good_path, empty_brief, project_folder) == []
    assert (
        "Missing values"
        in check_predictions_file(missing_value_path, empty_brief, project_folder)[0]
    )


def test_missing_empty_and_unreadable_files_are_reported(tmp_path, project_folder, brief):
    missing_path = tmp_path / "missing.csv"
    empty_path = tmp_path / "empty.csv"
    empty_path.write_text("", encoding="utf-8")
    header_only_path = tmp_path / "header_only.csv"
    header_only_path.write_text("id,churn\n", encoding="utf-8")

    assert "does not exist" in check_predictions_file(missing_path, brief, project_folder)[0]
    assert "is empty" in check_predictions_file(empty_path, brief, project_folder)[0]
    assert "no rows" in check_predictions_file(header_only_path, brief, project_folder)[0]


def test_files_that_are_not_csv_only_need_to_exist_and_have_content(
    tmp_path, project_folder, brief
):
    json_path = tmp_path / "predictions.json"
    json_path.write_text('{"a": 1}', encoding="utf-8")

    assert check_predictions_file(json_path, brief, project_folder) == []


def test_a_predictions_folder_must_contain_something(tmp_path, project_folder, brief):
    folder = tmp_path / "predictions"
    folder.mkdir()
    assert "is empty" in check_predictions_file(folder, brief, project_folder)[0]

    (folder / "mask_001.png").write_bytes(b"png")
    assert check_predictions_file(folder, brief, project_folder) == []


def test_only_an_existing_csv_with_the_right_role_counts_as_a_sample_submission(
    tmp_path, project_folder
):
    wrong_role = ProjectBrief(files=[FileInfo(path="sample_submission.csv", role="train")])
    not_csv = ProjectBrief(files=[FileInfo(path="sample.json", role="sample_submission")])
    not_there = ProjectBrief(files=[FileInfo(path="nope.csv", role="sample_submission")])

    assert find_sample_submission(wrong_role, project_folder) is None
    assert find_sample_submission(not_csv, project_folder) is None
    assert find_sample_submission(not_there, project_folder) is None
