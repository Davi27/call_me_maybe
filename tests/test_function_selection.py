from typing import cast

from main import build_function_selection_context
from main import build_argument_object_context
from main import build_parameter_generation_context
from main import function_keyword_score
from main import select_function_name
from main import is_string_prefix_of_candidate
from main import extract_string_candidates
from main import extract_number_candidates
from main import map_ordered_number_arguments
from model_adapter import ModelAdapter


def test_function_selection_context_contains_request_and_candidates() -> None:
    functions = [
        {
            "name": "alpha",
            "description": "Do the first operation.",
            "parameters": {"value": {"type": "number"}},
        },
        {
            "name": "beta",
            "description": "Do the second operation.",
            "parameters": {"text": {"type": "string"}},
        },
    ]

    context = build_function_selection_context(
        "Please do the second operation with hello.",
        functions,
    )

    assert "Please do the second operation with hello." in context
    assert "alpha" in context
    assert "Do the first operation." in context
    assert "beta" in context
    assert "Do the second operation." in context
    assert context.endswith("Name:\nfn_")


def test_argument_object_context_contains_complete_schema() -> None:
    function = {
        "name": "some_operation",
        "description": "Perform an operation with two values.",
        "parameters": {
            "first_value": {"type": "number"},
            "second_value": {"type": "string"},
        },
    }

    context = build_argument_object_context(
        "Use 42 and hello.",
        function,
    )

    assert "Use 42 and hello." in context
    assert "some_operation" in context
    assert "Perform an operation with two values." in context
    assert "first_value" in context
    assert "number" in context
    assert "second_value" in context
    assert "string" in context
    assert context.endswith("JSON object:\n")


def test_parameter_generation_context_contains_partial_object() -> None:
    function = {
        "name": "some_operation",
        "description": "Perform an operation with two values.",
        "parameters": {
            "first_value": {"type": "number"},
            "second_value": {"type": "string"},
        },
    }

    context = build_parameter_generation_context(
        "Use 42 and hello.",
        function,
        {"first_value": 42},
        "second_value",
        "string",
    )

    assert "Use 42 and hello." in context
    assert "some_operation" in context
    assert "first_value" in context
    assert "second_value" in context
    assert '"first_value": 42' in context
    assert "Current parameter: second_value (string)" in context
    assert context.endswith("JSON value:\n")


def test_function_keyword_score_prefers_matching_candidate() -> None:
    request = "Please greet the person."

    greeting_function = {
        "name": "fn_greet",
        "description": "Generate a message for a person by name.",
        "parameters": {},
    }

    square_root_function = {
        "name": "fn_get_square_root",
        "description": "Calculate the square root of a number.",
        "parameters": {},
    }

    greeting_score = function_keyword_score(
        request,
        greeting_function,
    )
    square_root_score = function_keyword_score(
        request,
        square_root_function,
    )

    assert greeting_score > square_root_score


def test_function_keyword_score_ignores_common_words() -> None:
    request = "What is the sum of 2 and 3?"

    add_function = {
        "name": "fn_add_numbers",
        "description": "Add two numbers together and return their sum.",
        "parameters": {},
    }

    square_root_function = {
        "name": "fn_get_square_root",
        "description": "Calculate the square root of a number.",
        "parameters": {},
    }

    add_score = function_keyword_score(
        request,
        add_function,
    )
    square_root_score = function_keyword_score(
        request,
        square_root_function,
    )

    assert add_score > square_root_score


class RecordingAdapter:
    def encode_prompt(self, prompt: str) -> list[int]:
        return [1]


def test_select_function_name_generates_from_unique_keyword_candidate(
    monkeypatch,
) -> None:
    functions = [
        {
            "name": "fn_greet",
            "description": "Generate a greeting message for a person by name.",
            "parameters": {},
        },
        {
            "name": "fn_get_square_root",
            "description": "Calculate the square root of a number.",
            "parameters": {},
        },
    ]
    captured = {}

    def fake_generate_function_name(
        adapter,
        context_ids,
        initial_partial_name,
        allowed_names,
        max_steps,
    ):
        captured["allowed_names"] = allowed_names
        return allowed_names[0]

    monkeypatch.setattr(
        "function_selection.generate_function_name",
        fake_generate_function_name,
    )

    result = select_function_name(
        cast(ModelAdapter, RecordingAdapter()),
        "Greet shrek",
        functions,
        10,
    )

    assert result == "fn_greet"
    assert captured["allowed_names"] == ["fn_greet"]


def test_string_prefix_must_match_one_candidate() -> None:
    candidates = ["shrek", "john"]

    assert is_string_prefix_of_candidate('"s', candidates)
    assert is_string_prefix_of_candidate('"shr', candidates)
    assert is_string_prefix_of_candidate('"shrek"', candidates)

    assert not is_string_prefix_of_candidate('"x', candidates)
    assert not is_string_prefix_of_candidate('"shrekx"', candidates)


def test_extract_string_candidates_from_greeting() -> None:
    result = extract_string_candidates("Greet shrek")

    assert result == ["shrek"]


def test_extract_string_candidates_from_replacement_request() -> None:
    result = extract_string_candidates(
        'Replace all numbers in "Hello 34" with NUMBERS'
    )

    assert result == ["Hello 34", "NUMBERS"]


def test_empty_completed_string_is_not_a_candidate() -> None:
    assert not is_string_prefix_of_candidate(
        '""',
        ["shrek", "john"],
    )


def test_completed_string_must_equal_candidate() -> None:
    assert not is_string_prefix_of_candidate(
        '"s"',
        ["shrek", "john"],
    )

    assert is_string_prefix_of_candidate(
        '"shrek"',
        ["shrek", "john"],
    )


def test_extract_number_candidates() -> None:
    result = extract_number_candidates(
        "What is the sum of 2 and 3?"
    )

    assert result == ["2", "3"]


def test_map_ordered_number_arguments() -> None:
    result = map_ordered_number_arguments(
        ["a", "b"],
        ["2", "3"],
    )

    assert result == {
        "a": 2,
        "b": 3,
    }


def test_map_ordered_number_arguments_returns_none_when_counts_differ() -> None:
    result = map_ordered_number_arguments(
        ["a"],
        ["1000", "2500"],
    )

    assert result is None
