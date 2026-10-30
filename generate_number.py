

def generate_number_argument(
    adapter: ModelAdapter,
    initial_context_ids: list[int],
    max_steps: int,
) -> int | float:
    context_ids = list(initial_context_ids)
    text = ""

    for _ in range(max_steps):
        logits = adapter.score_ids(context_ids)
        records = build_candidate_records(range(len(logits)), logits, adapter.decode_id)
        legal_records = [
            record for record in records
            if record["piece"]
            and json_number_prefix_status(text + record["piece"]) != "invalid"
        ]
        if len(legal_records) == 0:
            raise NoLegalContinuationError("No legal continuation for this partial number")
        selected_record = max(legal_records, key=lambda record: record["score"])

        selected_id = selected_record["id"]
        context_ids.append(selected_id)
        text += selected_record["piece"]
        if json_number_prefix_status(text) == "ended":
            return cast(int | float, json.loads(text[:-1]))

    raise StepLimitExceededError("No number was created in the steps length")


def extract_number_candidates(request: str) -> list[str]:
    return re.findall(
        r"-?\d+(?:\.\d+)?",
        request,
    )


def number_value(text: str) -> int | float:
    if "." in text:
        return float(text)

    return int(text)


def map_ordered_number_arguments(
    parameter_names: list[str],
    number_candidates: list[str],
) -> dict[str, int | float] | None:
    if len(parameter_names) != len(number_candidates):
        return None

    return {
        parameter_name: number_value(number_text)
        for parameter_name, number_text in zip(
            parameter_names,
            number_candidates,
        )
    }