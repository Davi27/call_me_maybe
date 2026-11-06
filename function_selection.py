import re

from context_builder import build_function_selection_context
from generate_name import generate_function_name
from model_adapter import ModelAdapter


FUNCTION_STOP_WORDS = {
    "a",
    "an",
    "and",
    "are",
    "do",
    "for",
    "how",
    "in",
    "is",
    "me",
    "of",
    "please",
    "the",
    "this",
    "to",
    "what",
    "with",
}


def function_keyword_score(request: str, func: dict) -> int:
    """Return the number of words shared by the request and function metadata.

    Args:
        request: The natural-language request.
        func: A function definition containing a name and description.

    Returns:
        The number of shared non-stop words.
    """
    request_words = {
        word
        for word in re.findall(r"[a-z0-9]+", request.lower())
        if word not in FUNCTION_STOP_WORDS
    }

    function_text = f"{func['name']} {func['description']}"

    function_words = {
        word
        for word in re.findall(
            r"[a-z0-9]+",
            function_text.lower().replace("_", " "),
        )
        if word not in FUNCTION_STOP_WORDS
    }
    return len(request_words & function_words)


def select_function_name(
    adapter: ModelAdapter,
    request: str,
    funcs: list[dict],
    max_steps: int,
) -> str:
    """Select and generate the function name that best matches the request.

    Args:
        adapter: Model adapter used to score candidate tokens.
        request: The natural-language request.
        funcs: Available function definitions.
        max_steps: Maximum number of generation iterations.

    Returns:
        The selected function name.

    Raises:
        NoLegalContinuationError: If no token can continue an allowed name.
        StepLimitExceededError: If the name is incomplete after max_steps.
    """
    scored_functions = [
        (
            function_keyword_score(request, func),
            func["name"],
        )
        for func in funcs
    ]
    highest_score = max(score for score, _ in scored_functions)
    best_names = [
        name
        for score, name in scored_functions
        if score == highest_score
    ]

    context = build_function_selection_context(request, funcs)
    context_ids = adapter.encode_prompt(context)

    return generate_function_name(
        adapter,
        context_ids,
        "fn_",
        best_names,
        max_steps,
    )
