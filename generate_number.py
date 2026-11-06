import json
import re

from candidates import build_candidate_records
from errors import NoLegalContinuationError, StepLimitExceededError
from model_adapter import ModelAdapter
from prefix_status import (
    json_integer_prefix_status,
    json_number_prefix_status,
)


def generate_number_argument(
    adapter: ModelAdapter,
    initial_context_ids: list[int],
    max_steps: int,
) -> float:
    """Generate a JSON number as a function argument.

    Args:
        adapter: Model adapter used to score candidate tokens.
        initial_context_ids: Token IDs representing the initial context.
        max_steps: Maximum number of generation iterations.

    Returns:
        The generated number as a float.

    Raises:
        NoLegalContinuationError: If no token can continue a valid number.
        StepLimitExceededError: If the number is incomplete after max_steps.
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
            and json_number_prefix_status(
                text + record["piece"]
            ) != "invalid"
        ]
        if not legal_records:
            raise NoLegalContinuationError(
                "No legal continuation for this partial number"
            )

        selected_record = max(
            legal_records,
            key=lambda record: record["score"],
        )
        context_ids.append(selected_record["id"])
        text += selected_record["piece"]

        if json_number_prefix_status(text) == "ended":
            return float(json.loads(text[:-1]))

    raise StepLimitExceededError(
        "No number was created in the steps length"
    )


def generate_integer_argument(
    adapter: ModelAdapter,
    initial_context_ids: list[int],
    max_steps: int,
) -> int:
    """Generate a JSON integer as a function argument.

    Args:
        adapter: Model adapter used to score candidate tokens.
        initial_context_ids: Token IDs representing the initial context.
        max_steps: Maximum number of generation iterations.

    Returns:
        The generated number as an integer.

    Raises:
        NoLegalContinuationError: If no token can continue a valid integer.
        StepLimitExceededError: If the integer is incomplete after max_steps.
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
            and json_integer_prefix_status(
                text + record["piece"]
            ) != "invalid"
        ]
        if not legal_records:
            raise NoLegalContinuationError(
                "No legal continuation for this partial number"
            )

        selected_record = max(
            legal_records,
            key=lambda record: record["score"],
        )
        context_ids.append(selected_record["id"])
        text += selected_record["piece"]

        if json_integer_prefix_status(text) == "ended":
            return int(json.loads(text[:-1]))

    raise StepLimitExceededError(
        "No number was created in the steps length"
    )


def extract_number_candidates(request: str) -> list[str]:
    """Extract integer and decimal number literals from a request.

    Args:
        request: The natural-language request.

    Returns:
        The numeric literals found in their original textual form.
    """
    return re.findall(r"-?\d+(?:\.\d+)?", request)


def number_value(text: str) -> float:
    """Convert a numeric literal to the required float representation."""
    return float(text)


def map_ordered_number_arguments(
    parameter_names: list[str],
    number_candidates: list[str],
) -> dict[str, float] | None:
    """Map extracted numeric values to parameter names when counts match.

    Args:
        parameter_names: Parameter names to populate.
        number_candidates: Numeric literals extracted from the request.

    Returns:
        A parameter-to-float mapping, or None when the counts differ.
    """
    if len(parameter_names) != len(number_candidates):
        return None

    return {
        parameter_name: number_value(number_text)
        for parameter_name, number_text in zip(
            parameter_names,
            number_candidates,
        )
    }


def map_ordered_integer_arguments(
    parameter_names: list[str],
    integer_candidates: list[str],
) -> dict[str, int] | None:
    """Map extracted integral values to integer parameter names.

    Args:
        parameter_names: Integer parameter names to populate.
        integer_candidates: Numeric literals extracted from the request.

    Returns:
        A parameter-to-integer mapping, or None when the counts differ or a
        candidate is decimal.
    """
    if len(parameter_names) != len(integer_candidates):
        return None
    if any("." in candidate for candidate in integer_candidates):
        return None

    return {
        parameter_name: int(integer_text)
        for parameter_name, integer_text in zip(
            parameter_names,
            integer_candidates,
        )
    }
