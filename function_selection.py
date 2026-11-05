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
    """ 
        Look for same words in prompt - function_name | description.

        Arguments:
            request - a prompt
            func - a function definition

        Return:
            the number of matches
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
    """ 
        Stores matches of words between the prompt and 
        function name|description.

        Arguments:
            adapter - an instance to the model adapter class
            request - a prompt
            funcs - a list with all function definitions
            max_steps - the limit of times the loop can run

        Return:
            the return from generate_function_name()
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
	