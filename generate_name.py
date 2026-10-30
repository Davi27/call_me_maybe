

def generate_function_name(adapter: ModelAdapter, initial_context_ids: list[int], initial_partial_name: str, allowed_names: list[str], max_steps: int) -> str:
    context_ids = list(initial_context_ids)
    partial_name = initial_partial_name

    for _ in range(max_steps):
        logits = adapter.score_ids(context_ids)
        records = build_candidate_records(range(len(logits)), logits, adapter.decode_id)
        legal_records = candidate_filter(partial_name, allowed_names, records)
        if len(legal_records) == 0:
            raise NoLegalContinuationError("No legal continuation for this partial name")
        selected_record = max(legal_records, key=lambda record: record["score"])

        selected_id = selected_record["id"]
        context_ids.append(selected_id)
        partial_name += cast(str, selected_record["piece"])

        if partial_name in allowed_names:
            return partial_name
    raise StepLimitExceededError("No name was created in the steps length")


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
    scored_functions = [
        (
            function_keyword_score(request, func),
            func["name"],
        )
        for func in funcs
    ]

    highest_score = max(
        score for score, _ in scored_functions
    )
    best_names = [
        name
        for score, name in scored_functions
        if score == highest_score
    ]

    if highest_score > 0 and len(best_names) == 1:
        return best_names[0]

    context = build_function_selection_context(request, funcs)
    context_ids = adapter.encode_prompt(context)
    names = [func["name"] for func in funcs]

    return generate_function_name(
        adapter,
        context_ids,
        "fn_",
        names,
        max_steps,
    )
