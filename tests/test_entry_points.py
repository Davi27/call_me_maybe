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
    assert result.returncode == 0
    assert "File not found" in result.stdout
    assert str(missing_input) in result.stdout
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
    assert result.returncode == 0
    assert "contains the incorrect format" in result.stdout
    assert str(incorrect_format_input) in result.stdout
    assert not output_path.exists()
