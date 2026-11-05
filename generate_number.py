import json
import re
from typing import cast

from candidates import build_candidate_records
from errors import NoLegalContinuationError, StepLimitExceededError
from model_adapter import ModelAdapter
from prefix_status import json_number_prefix_status


def generate_number_argument(
    adapter: ModelAdapter,
    initial_context_ids: list[int],
    max_steps: int,
) -> float:
    """ 
        Generate a number as a value of a function parameter

        Arguments:
            adapter - an instance of the ModelAdapter class
            initial_context_ids - the context to feed the model
            max_steps - the limit of iterations

        Return:
            the number generated converted into a float
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
    """ 
        Generate a number as a value of a function parameter

        Arguments:
            adapter - an instance of the ModelAdapter class
            initial_context_ids - the context to feed the model
            max_steps - the limit of iterations

        Return:
            the number generated converted into an integer
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
            return int(json.loads(text[:-1]))

    raise StepLimitExceededError(
        "No number was created in the steps length"
    )


def extract_number_candidates(request: str) -> list[str]:
    """ 
        Extract all digits from the input

        Arguments:
            request - the input

        Return:
            a list with all the numbers found
    """
    return re.findall(r"-?\d+(?:\.\d+)?", request)


def number_value(text: str) -> float:
    return float(text)


def map_ordered_number_arguments(
    parameter_names: list[str],
    number_candidates: list[str],
) -> dict[str, float] | None:
    """ 
        Order the numbers, if more than one is needed

        Arguments:
            parameter_names - a list of parameter names
            number_candidates - a list with extracted numbers

        Return:
            a dictionary matching the name - 
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
