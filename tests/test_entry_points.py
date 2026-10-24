from __future__ import annotations

import subprocess
import sys
from pathlib import Path
import json


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_official_src_module_entry_point_shows_help() -> None:
    result = subprocess.run(
        ["uv", "run", "python", "-m", "src", "--help"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0
    assert "--functions_definition" in result.stdout
    assert "--input" in result.stdout
    assert "--output" in result.stdout


def test_compatibility_main_entry_point_shows_help() -> None:
    result = subprocess.run(
        [sys.executable, "main.py", "--help"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0
    assert "--functions_definition" in result.stdout


def test_valid_invocation_writes_expected_json(tmp_path: Path) -> None:
    output_path = tmp_path / "function_calling_results.json"

    result = subprocess.run(
        [
            "uv",
            "run",
            "python",
            "-m",
            "src",
            "--output",
            str(output_path),
        ],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0
    assert output_path.exists()

    output = json.loads(output_path.read_text())

    assert isinstance(output, list)
    assert len(output) == 11

    for entry in output:
        assert set(entry) == {"prompt", "name", "parameters"}
        assert isinstance(entry["prompt"], str)
        assert isinstance(entry["name"], str)
        assert isinstance(entry["parameters"], dict)

    definitions = json.loads(
        (PROJECT_ROOT / "data/input/functions_definition.json").read_text()
    )

    definitions_by_name = {
        definition["name"]: definition
        for definition in definitions
    }

    prompts = json.loads(
        (PROJECT_ROOT / "data/input/function_calling_tests.json").read_text()
    )

    assert [entry["prompt"] for entry in output] == [
        item["prompt"] for item in prompts
    ]

    for entry in output:
        name = entry["name"]

        # Reject TODO, unknown, or hardcoded invalid names.
        assert name in definitions_by_name

        definition = definitions_by_name[name]
        expected_parameters = definition["parameters"]
        actual_parameters = entry["parameters"]

        # Every required parameter exists, and no extra parameter is produced.
        assert set(actual_parameters) == set(expected_parameters)

        for parameter_name, parameter_definition in expected_parameters.items():
            value = actual_parameters[parameter_name]
            expected_type = parameter_definition["type"]

            if expected_type == "string":
                assert isinstance(value, str)

            elif expected_type == "number":
                # bool is a subclass of int in Python, so exclude it.
                assert isinstance(value, (int, float))
                assert not isinstance(value, bool)

            elif expected_type == "boolean":
                assert isinstance(value, bool)

            else:
                raise AssertionError(
                    f"Unsupported test type: {expected_type}"
                )


def test_changed_definition_schema_is_used(tmp_path: Path) -> None:
    definitions_path = tmp_path / "definitions.json"
    input_path = tmp_path / "prompts.json"
    output_path = tmp_path / "results.json"

    definitions_path.write_text(
        json.dumps(
            [
                {
                    "name": "fn_add_numbers",
                    "description": (
                        "Add two numbers together and return their sum."
                    ),
                    "parameters": {
                        "value": {"type": "number"},
                    },
                    "returns": {"type": "number"},
                }
            ]
        ),
        encoding="utf-8",
    )

    input_path.write_text(
        json.dumps(
            [
                {"prompt": "What is the sum of 2 and 3?"},
            ]
        ),
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            "uv",
            "run",
            "python",
            "-m",
            "src",
            "--functions_definition",
            str(definitions_path),
            "--input",
            str(input_path),
            "--output",
            str(output_path),
        ],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0
    assert output_path.exists()

    output = json.loads(output_path.read_text(encoding="utf-8"))

    assert len(output) == 1
    assert output[0]["prompt"] == "What is the sum of 2 and 3?"
    assert output[0]["name"] == "fn_add_numbers"
    assert set(output[0]["parameters"]) == {"value"}
    assert isinstance(output[0]["parameters"]["value"], (int, float))
    assert not isinstance(output[0]["parameters"]["value"], bool)


def test_missing_input_is_handled(tmp_path: Path) -> None:
    missing_input = tmp_path / "missing.json"
    output_path = tmp_path / "function_calling_results.json"

    result = subprocess.run(
        [
            "uv",
            "run",
            "python",
            "-m",
            "src",
            "--input",
            str(missing_input),
            "--output",
            str(output_path),
        ],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode != 0
    assert "Input" in result.stderr
    assert str(missing_input) in result.stderr
    assert not output_path.exists()


def test_input_incorrect_format_is_handled(tmp_path: Path) -> None:
    incorrect_format_input = tmp_path / "incorrect_format.json"
    incorrect_format_input.write_text("{ invalid json", encoding="utf-8")

    output_path = tmp_path / "function_calling_results.json"

    result = subprocess.run(
        [
            "uv",
            "run",
            "python",
            "-m",
            "src",
            "--input",
            str(incorrect_format_input),
            "--output",
            str(output_path),
        ],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode != 0
    assert "Input" in result.stderr
    assert "Expecting property name" in result.stderr
    assert str(incorrect_format_input) in result.stderr
    assert not output_path.exists()
