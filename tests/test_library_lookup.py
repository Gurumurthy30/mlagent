import pytest

from mlagent.code_runner import CodeRunner
from mlagent.library_lookup import describe_python_object, find_python_object
from mlagent.settings import LIBRARY_HELP_CHARS


def test_function_description_has_signature_and_documentation():
    description = describe_python_object("json.dumps")

    assert description.startswith("json.dumps(")
    assert "Serialize" in description


def test_nested_attribute_of_a_class_is_found():
    description = describe_python_object("collections.OrderedDict.move_to_end")

    assert "move_to_end" in description


def test_module_description_lists_public_names():
    description = describe_python_object("json")

    assert "Public names:" in description
    assert "dumps" in description


def test_description_is_cut_to_its_limit():
    assert len(describe_python_object("os")) <= LIBRARY_HELP_CHARS


def test_unknown_names_raise_lookup_error():
    with pytest.raises(LookupError, match="Not found"):
        find_python_object("no_such_library_xyz.thing")
    with pytest.raises(LookupError, match="Not found"):
        find_python_object("json.no_such_function")


def test_lookup_runs_as_a_module_through_the_code_runner(workspace):
    code_runner = CodeRunner(workspace, timeout_seconds=30)

    result = code_runner.run_python_module("mlagent.library_lookup", ["json.loads"], workspace.root)

    assert result.succeeded
    assert result.stdout.startswith("json.loads(")


def test_failed_lookup_reports_the_error(workspace):
    code_runner = CodeRunner(workspace, timeout_seconds=30)

    result = code_runner.run_python_module(
        "mlagent.library_lookup", ["no_such_lib_xyz"], workspace.root
    )

    assert not result.succeeded
    assert "Not found: no_such_lib_xyz" in result.error_text
