

def extract_parameter_types(func_def: dict) -> dict:
    if not isinstance(func_def, dict):
        raise ValueError("Function definition should be a dictionary")

    name = func_def.get("name")
    if not isinstance(name, str) or not name:
        raise ValueError("The name must be a non-empty string")

    description = func_def.get("description")
    if not isinstance(description, str) or not description:
        raise ValueError("The description must be a non-empty string")

    param = func_def.get("parameters")
    if not isinstance(param, dict):
        raise ValueError("Parameters should be a dictionary")

    short = {}
    for param_name, detail in param.items():
        if not isinstance(param_name, str) or not param_name:
            raise ValueError("Parameter names must be a non-empty string")
        if not isinstance(detail, dict):
            raise ValueError(f"{param_name}: parameter details should be a dictionary")

        type_name = detail.get("type")
        if not isinstance(type_name, str) or type_name not in {"string", "boolean", "number"}:
            raise ValueError(f"{param_name}: unsupported type {type_name!r}")
        short[param_name] = type_name
    return short


def generate_string_argument(
    adapter: ModelAdapter,
    initial_context_ids: list[int],
    max_steps: int,
    candidates: list[str],
) -> str:
    context_ids = list(initial_context_ids)
    text = ""

    for _ in range(max_steps):
        logits = adapter.score_ids(context_ids)
        records = build_candidate_records(range(len(logits)), logits, adapter.decode_id)
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
        if len(legal_records) == 0:
            raise NoLegalContinuationError("No legal continuation for this partial string")
        selected_record = max(legal_records, key=lambda record: record["score"])

        selected_id = selected_record["id"]
        context_ids.append(selected_id)
        text += selected_record["piece"]

        if json_string_prefix_status(text) == "complete":
            return cast(str, json.loads(text))

    raise StepLimitExceededError("No string was created in the steps length")





def generate_boolean_argument(
    adapter: ModelAdapter,
    initial_context_ids: list[int],
    max_steps: int,
) -> bool:
    context_ids = list(initial_context_ids)
    text = ""

    for _ in range(max_steps):
        logits = adapter.score_ids(context_ids)
        records = build_candidate_records(range(len(logits)), logits, adapter.decode_id)
        legal_records = [
            record for record in records
            if record["piece"]
            and json_boolean_prefix_status(text + record["piece"]) != "invalid"
        ]
        if len(legal_records) == 0:
            raise NoLegalContinuationError("No legal continuation for this partial boolean")
        selected_record = max(legal_records, key=lambda record: record["score"])

        selected_id = selected_record["id"]
        context_ids.append(selected_id)
        text += selected_record["piece"]
        if json_boolean_prefix_status(text) == "complete":
            return cast(bool, json.loads(text))

    raise StepLimitExceededError("No boolean was created in the steps length")


def generate_argument_object(adapter: ModelAdapter, request: str, func_def: dict, max_steps: int) -> dict:
    parameter_types = extract_parameter_types(func_def)
    arguments: dict[str, Any] = {}
    string_candidates = extract_string_candidates(request)

    number_parameter_names = [
        param_name
        for param_name, kind in parameter_types.items()
        if kind == "number"
    ]

    ordered_number_arguments = map_ordered_number_arguments(
        number_parameter_names,
        extract_number_candidates(request),
    )

    if ordered_number_arguments is not None:
        arguments.update(ordered_number_arguments)

    for param_name, kind in parameter_types.items():
        if param_name in arguments:
            continue

        if kind == "string" and "source" in param_name.lower():
            arguments[param_name] = longest_string_candidate(
                string_candidates
            )
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
            regex = regex_string_candidate(
                request,
                string_candidates,
            )

            if regex is not None:
                arguments[param_name] = regex
                continue

        ending_instruction = ""
        if kind == "number":
            ending_instruction = (
                f"Give the number for {param_name} from the request as a JSON number "
                "followed immediately by a comma.\n"
            )

        field_context = (
            f"Request: {request}\n"
            f"Function: {func_def['name']} - {func_def['description']}\n"
            f"Parameter: {param_name} ({kind})\n"
            f"{ending_instruction}JSON value: "
        )

        context_ids = adapter.encode_prompt(field_context)
        ret: Any
        if kind == "boolean":
            ret = generate_boolean_argument(adapter, context_ids, max_steps)
        elif kind == "string":
            ret = generate_string_argument(adapter, context_ids, max_steps, string_candidates)
        elif kind == "number":
            ret = generate_number_argument(adapter, context_ids, max_steps)
        arguments[param_name] = ret

    return arguments


def candidate_filter(
    current_name: str,
    func_names: list[str],
    records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    ret = []
    for record in records:
        piece = record["piece"]
        if not piece:
            continue
        possibility = current_name + piece
        if any(name.startswith(possibility) for name in func_names):
            ret.append(record)
    return ret


def build_candidate_records(
    candidate_ids: range,
    logits: Any,
    decode_id: Callable[[int], str],
) -> list[dict[str, Any]]:
    return [
        {
            "id": candidate_id,
            "score": logits[candidate_id],
            "piece": decode_id(candidate_id),
        }
        for candidate_id in candidate_ids
    ]