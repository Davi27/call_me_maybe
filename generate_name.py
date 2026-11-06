from typing import cast

from candidates import build_candidate_records, candidate_filter
from errors import NoLegalContinuationError, StepLimitExceededError
from model_adapter import ModelAdapter


def generate_function_name(
    adapter: ModelAdapter,
    initial_context_ids: list[int],
    initial_partial_name: str,
    allowed_names: list[str],
    max_steps: int,
) -> str:
    """Generate an allowed function name one token at a time.

    Args:
        adapter: Model adapter used to score candidate tokens.
        initial_context_ids: Token IDs representing the initial context.
        initial_partial_name: Function-name prefix generated so far.
        allowed_names: Function names that may be generated.
        max_steps: Maximum number of generation iterations.

    Returns:
        The generated function name.

    Raises:
        NoLegalContinuationError: If no token can continue an allowed name.
        StepLimitExceededError: If the name is incomplete after max_steps.
    """
    context_ids = list(initial_context_ids)
    partial_name = initial_partial_name

    for _ in range(max_steps):
        logits = adapter.score_ids(context_ids)
        records = build_candidate_records(
            range(len(logits)),
            logits,
            adapter.decode_id,
        )
        legal_records = candidate_filter(
            partial_name,
            allowed_names,
            records,
        )
        if not legal_records:
            raise NoLegalContinuationError(
                "No legal continuation for this partial name"
            )

        selected_record = max(
            legal_records,
            key=lambda record: record["score"],
        )
        selected_id = selected_record["id"]
        context_ids.append(selected_id)
        partial_name += cast(str, selected_record["piece"])

        if partial_name in allowed_names:
            return partial_name

    raise StepLimitExceededError(
        "No name was created in the steps length"
    )
