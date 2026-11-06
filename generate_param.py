import json
from typing import Any, cast

from candidates import build_candidate_records
from definitions import extract_parameter_types
from errors import NoLegalContinuationError, StepLimitExceededError
from generate_number import (
    extract_number_candidates,
    generate_number_argument,
    generate_integer_argument,
    map_ordered_integer_arguments,
    map_ordered_number_arguments,
)
from model_adapter import ModelAdapter
from prefix_status import (
    json_boolean_prefix_status,
    json_string_prefix_status,
)
from substitute_accuracy import (
    extract_string_candidates,
    is_string_prefix_of_candidate,
    longest_string_candidate,
    regex_string_candidate,
    replacement_string_candidate,
)


def generate_string_argument(
    adapter: ModelAdapter,
    initial_context_ids: list[int],
    max_steps: int,
    candidates: list[str],
) -> str:
    """Generate a string as a function argument.

    Args:
        adapter: Model adapter used to score candidate tokens.
        initial_context_ids: Token IDs representing the initial context.
        max_steps: Maximum number of generation iterations.
        candidates: Request-derived string candidates.

    Returns:
        The generated string.

    Raises:
        NoLegalContinuationError: If no token can continue a valid string.
        StepLimitExceededError: If the string is incomplete after max_steps.
    """
    context_ids = list(initial_context_ids)
    text = ""

    for _ in range(max_steps):
        logits = adapter.score_ids(context_ids)
        records = build_candidate_records(
            range(len(logits)),
            logits,
            adapter.decode_id,
        )
        legal_records = [
            record
            for record in records
            if record["piece"]
            and json_string_prefix_status(
                text + record["piece"]
            ) != "invalid"
            and is_string_prefix_of_candidate(
                text + record["piece"],
                candidates,
            )
        ]
        if not legal_records:
            raise NoLegalContinuationError(
                "No legal continuation for this partial string"
            )

        selected_record = max(
            legal_records,
            key=lambda record: record["score"],
        )
        context_ids.append(selected_record["id"])
        text += selected_record["piece"]

        if json_string_prefix_status(text) == "complete":
            return cast(str, json.loads(text))

    raise StepLimitExceededError(
        "No string was created in the steps length"
    )


def generate_boolean_argument(
    adapter: ModelAdapter,
    initial_context_ids: list[int],
    max_steps: int,
) -> bool:
    """Generate a boolean as a function argument.

    Args:
        adapter: Model adapter used to score candidate tokens.
        initial_context_ids: Token IDs representing the initial context.
        max_steps: Maximum number of generation iterations.

    Returns:
        The generated boolean.

    Raises:
        NoLegalContinuationError: If no token can continue a valid boolean.
        StepLimitExceededError: If the boolean is incomplete after max_steps.
    """
    context_ids = list(initial_context_ids)
    text = ""

    for _ in range(max_steps):
        logits = adapter.score_ids(context_ids)
        records = build_candidate_records(
            range(len(logits)),
            logits,
            adapter.decode_id,
        )
        legal_records = [
            record
            for record in records
            if record["piece"]
            and json_boolean_prefix_status(
                text + record["piece"]
            ) != "invalid"
        ]
        if not legal_records:
            raise NoLegalContinuationError(
                "No legal continuation for this partial boolean"
            )

        selected_record = max(
            legal_records,
            key=lambda record: record["score"],
        )
        context_ids.append(selected_record["id"])
        text += selected_record["piece"]

        if json_boolean_prefix_status(text) == "complete":
            return cast(bool, json.loads(text))

    raise StepLimitExceededError(
        "No boolean was created in the steps length"
    )


def generate_argument_object(
    adapter: ModelAdapter,
    request: str,
    func_def: dict,
    max_steps: int,
) -> dict:
    """Generate an argument dictionary for a validated function definition.

    Args:
        adapter: Model adapter used to generate constrained values.
        request: The natural-language request.
        func_def: The selected function definition.
        max_steps: Maximum number of generation iterations per value.

    Returns:
        A dictionary mapping parameter names to generated values.

    Raises:
        ValueError: If the function definition or a value type is invalid.
        NoLegalContinuationError: If a value has no valid continuation.
        StepLimitExceededError: If a value is incomplete after max_steps.
    """
    parameter_types = extract_parameter_types(func_def)
    arguments: dict[str, Any] = {}
    string_candidates = extract_string_candidates(request)

    numeric_parameter_names = [
        param_name
        for param_name, kind in parameter_types.items()
        if kind in {"number", "integer"}
    ]
    numeric_candidates = extract_number_candidates(request)
    if len(numeric_parameter_names) == len(numeric_candidates):
        ordered_numeric_arguments: dict[str, Any] = {}
        can_map_numeric_arguments = True
        for parameter_name, numeric_text in zip(
            numeric_parameter_names,
            numeric_candidates,
        ):
            kind = parameter_types[parameter_name]
            if kind == "integer":
                integer_arguments = map_ordered_integer_arguments(
                    [parameter_name],
                    [numeric_text],
                )
                if integer_arguments is None:
                    can_map_numeric_arguments = False
                    break
                ordered_numeric_arguments.update(integer_arguments)
            else:
                number_arguments = map_ordered_number_arguments(
                    [parameter_name],
                    [numeric_text],
                )
                if number_arguments is None:
                    can_map_numeric_arguments = False
                    break
                ordered_numeric_arguments.update(number_arguments)
        if can_map_numeric_arguments:
            arguments.update(ordered_numeric_arguments)

    for param_name, kind in parameter_types.items():
        if param_name in arguments:
            continue

        if kind == "string" and "source" in param_name.lower():
            arguments[param_name] = longest_string_candidate(string_candidates)
            continue

        if kind == "string" and "replacement" in param_name.lower():
            replacement = replacement_string_candidate(
                request,
                string_candidates,
            )
            if replacement is not None:
                arguments[param_name] = replacement
                continue

        if kind == "string" and (
            "regex" in param_name.lower()
            or "pattern" in param_name.lower()
        ):
            regex = regex_string_candidate(request, string_candidates)
            if regex is not None:
                arguments[param_name] = regex
                continue

        ending_instruction = ""
        if kind == "number":
            ending_instruction = (
                f"Give the number for {param_name} from the request as a "
                "JSON number followed immediately by a comma.\n"
            )

        field_context = (
            f"Request: {request}\n"
            f"Function: {func_def['name']} - {func_def['description']}\n"
            f"Parameter: {param_name} ({kind})\n"
            f"{ending_instruction}JSON value: "
        )
        context_ids = adapter.encode_prompt(field_context)

        if kind == "boolean":
            value: Any = generate_boolean_argument(
                adapter,
                context_ids,
                max_steps,
            )
        elif kind == "string":
            value = generate_string_argument(
                adapter,
                context_ids,
                max_steps,
                string_candidates,
            )
        elif kind == "number":
            value = generate_number_argument(
                adapter,
                context_ids,
                max_steps,
            )
        elif kind == "integer":
            value = generate_integer_argument(
                adapter,
                context_ids,
                max_steps,
            )
        else:
            raise ValueError(f"Unsupported parameter type: {kind}")
        arguments[param_name] = value

    return arguments
