from pathlib import Path
import re
import json
import argparse
import llm_sdk
import sys
from typing import Any, Callable, cast


class NoLegalContinuationError(ValueError):
    pass


class StepLimitExceededError(Exception):
    pass


class ModelAdapter:
    def __init__(self) -> None:
        model_factory: Any = getattr(llm_sdk, "Small_LLM_Model")
        self._sdk_model: Any = model_factory()

    def encode_prompt(self, prompt: str) -> list[int]:
        return cast(list[int], self._sdk_model.encode(prompt)[0].tolist())

    def score_ids(self, ids: list[int]) -> Any:
        return self._sdk_model.get_logits_from_input_ids(ids)

    def decode_id(self, token_id: int) -> str:
        return cast(str, self._sdk_model.decode([token_id]))


def longest_string_candidate(
    candidates: list[str],
) -> str:
    if not candidates:
        raise ValueError("No string candidates were provided")

    return max(
        candidates,
        key=len,
    )


def replacement_string_candidate(
    request: str,
    candidates: list[str],
) -> str | None:
    match = re.search(
        r"\bwith\s+(?:\"([^\"]+)\"|'([^']+)'|([A-Za-z0-9_*+-]+))",
        request,
        re.IGNORECASE,
    )

    if match is None:
        return None

    replacement = next(
        value
        for value in match.groups()
        if value is not None
    )

    if replacement in candidates:
        return replacement

    return None


def regex_string_candidate(
    request: str,
    candidates: list[str],
) -> str | None:
    lowered_request = request.lower()

    if "numbers" in lowered_request:
        return r"\d+"

    if "vowels" in lowered_request:
        return r"[aeiouAEIOU]"

    word_match = re.search(
        r"\bword\s+['\"]([^'\"]+)['\"]",
        request,
        re.IGNORECASE,
    )

    if word_match is not None:
        word = word_match.group(1)

        if word in candidates:
            return word

    return None


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


def is_string_prefix_of_candidate(
    text: str,
    candidates: list[str],
) -> bool:
    if not text.startswith('"'):
        return False

    if text == '""':
        return "" in candidates

    partial_text = text[1:]

    if partial_text.endswith('"'):
        partial_text = partial_text[:-1]
        return partial_text in candidates

    return any(
        candidate.startswith(partial_text)
        for candidate in candidates
    )


def extract_string_candidates(request: str) -> list[str]:
    quoted_values = re.findall(
        r'"([^"]*)"|\'([^\']*)\'',
        request,
    )

    candidates = []

    for double_quoted, single_quoted in quoted_values:
        value = double_quoted or single_quoted
        if value and value not in candidates:
            candidates.append(value)

    words = re.findall(
        r"[A-Za-z][A-Za-z0-9_]*",
        request,
    )

    if words:
        final_word = words[-1]
        if final_word not in candidates:
            candidates.append(final_word)

    return candidates


def build_function_selection_context(request: str, funcs: list[dict]) -> str:
    context = "Choose the function that best matches the request.\n"
    context += "Return only one available function name.\n\n"
    context = "Available Functions:\n"

    for func in funcs:
        context += func["name"] + " - " + func["description"] + "\n"

    context += "\nRequest:\n"
    context += request
    context += "\nName:\nfn_"

    return context


def build_argument_object_context(
    request: str,
    func_def: dict,
) -> str:
    context = (
        "Generate the complete argument object for the selected function.\n"
        "Return exactly one valid JSON object.\n\n"
    )
    context += f"Request:\n{request}\n\n"
    context += (
        f"Function:\n{func_def['name']} - "
        f"{func_def['description']}\n\n"
    )
    context += "Required parameters:\n"

    for param_name, details in func_def["parameters"].items():
        context += f"- {param_name}: {details['type']}\n"

    context += "\nJSON object:\n"
    return context


def build_parameter_generation_context(
    request: str,
    func_def: dict,
    generated_arguments: dict,
    param_name: str,
    kind: str,
) -> str:
    context = build_argument_object_context(request, func_def)
    context += (
        f"\nAlready generated arguments:\n"
        f"{json.dumps(generated_arguments)}\n"
    )
    context += f"Current parameter: {param_name} ({kind})\n"
    context += "Generate the JSON value for this parameter:\n"
    context += "JSON value:\n"

    return context


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


def json_boolean_prefix_status(text: str) -> str:
    if not text:
        return "incomplete"
    accept = ["false", "true"]
    for _ in text:
        if text in accept:
            return "complete"
        if not any(target.startswith(text) for target in accept):
            return "invalid"
        else:
            continue
    return "incomplete"


def json_number_prefix_status(text: str) -> str:
    i = 0
    is_float = False
    last = len(text) - 1
    nbrs = "0123456789"
    enders = " ,"
    if not text:
        return "incomplete"
    if text[i] not in "-0123456789":
        return "invalid"
    if i == last and text[i] in "-":
        return "incomplete"
    if i < last and text[i] == "0":
        if text[i + 1] not in "., ":
            return "invalid"
    elif i < last and text[i] == "-" and text[i + 1] == "0":
        if last > 1 and text[i + 2] not in "., ":
            return "invalid"

    for c in text[1:]:
        i += 1
        if c == "." and not is_float:
            if text[i - 1] not in nbrs:
                return "invalid"
            if i == last:
                return "incomplete"
            is_float = True
            continue
        elif c == "." and is_float:
            return "invalid"
        if c in nbrs:
            continue
        if c in enders:
            if i != last or text[i - 1] not in nbrs:
                return "invalid"
            return "ended"
        if c not in nbrs and c not in enders:
            return "invalid"
    return "complete"


def json_string_prefix_status(text: str) -> str:
    i = 0
    escaped = False
    unicode_left = 0
    if not text:
        return "incomplete"
    if text[0] != '"':
        return "invalid"
    for c in text[1:]:
        i += 1
        if unicode_left:
            if c not in "0123456789abcdefABCDEF":
                return "invalid"
            unicode_left -= 1
            continue

        if escaped:
            escaped = False
            if c == "u":
                unicode_left = 4
                continue
            if c not in '"\\/bfnrt':
                return "invalid"
            continue
        if c == "\\":
            escaped = True
            continue
        if c == '"':
            return "complete" if i == len(text) - 1 else "invalid"
        if ord(c) < 32:
            return "invalid"
    return "incomplete"
    """Return 'incomplete', 'complete', or 'invalid'."""


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


def main() -> None:
    parser = argparse.ArgumentParser()

    # set default arguments
    parser.add_argument("--functions_definition", default="data/input/functions_definition.json")
    parser.add_argument("--input", default="data/input/function_calling_tests.json")
    parser.add_argument("--output", default="data/output/function_calling_results.json")

    args = parser.parse_args()

    output_path = Path(args.output)
    if output_path.resolve() in {
        Path(args.input).resolve(),
        Path(args.functions_definition).resolve(),
    }:
        print("Output path must differ from input paths", file=sys.stderr)
        raise SystemExit(1)
    try:
        output_path.unlink(missing_ok=True)
    except OSError as error:
        print(f"Output {args.output}: {error}", file=sys.stderr)
        raise SystemExit(1)

    # protect from missing or wrong format input files
    try:
        with open(args.functions_definition, "r") as f:
            funcs = json.load(f)
        if not isinstance(funcs, list) or not funcs:
            raise ValueError("Function definitions must be a non-empty list")
        param = {}
        for func in funcs:
            types = extract_parameter_types(func)
            param[func["name"]] = types
    except (OSError, ValueError) as error:
        print(f"Definitions {args.functions_definition}: {error}", file=sys.stderr)
        raise SystemExit(1)

    try:
        with open(args.input, "r", encoding="utf-8") as f:
            inpt = json.load(f)
        if not isinstance(inpt, list):
            raise ValueError("Prompt input must be a JSON array")
        for i, inp in enumerate(inpt):
            if not isinstance(inp, dict):
                raise ValueError(f"Prompt {i} must be an object")
            if not isinstance(inp.get("prompt"), str):
                raise ValueError(f"Prompt {i} must contain a string prompt")
    except (OSError, ValueError) as error:
        print(f"Input {args.input}: {error}", file=sys.stderr)
        raise SystemExit(1)

    adapter = ModelAdapter()

    # legal function names list
    fns = []
    for fn in funcs:
        fns.append(fn["name"])

    # store the values in a list. Run a loop so we can convert all prompts
    results = []

    for i, inp in enumerate(inpt):
        context = build_function_selection_context(inp["prompt"], funcs)

        try:
            """ encoded_context = adapter.encode_prompt(context)
            chosen_name = generate_function_name(
                adapter, encoded_context, "fn_", fns, 10
            ) """
            chosen_name = select_function_name(
                adapter,
                inp["prompt"],
                funcs,
                10,
            )
            func_def = None
            for funct in funcs:
                if funct["name"] == chosen_name:
                    func_def = funct
                    break
            if func_def is None:
                raise ValueError(f"No definition for {chosen_name}")

            res = generate_argument_object(
                adapter, inp["prompt"], func_def, 10
            )
        except (NoLegalContinuationError, StepLimitExceededError,
                ValueError, RuntimeError) as error:
            print(
                f"Prompt {i}: {type(error).__name__}: {error}",
                file=sys.stderr,
            )
            raise SystemExit(1)

        result = {
            "prompt": inp["prompt"],
            "name": chosen_name,
            "parameters": res
        }
        results.append(result)

    try:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, allow_nan=False)
    except (OSError, TypeError, ValueError) as error:
        try:
            output_path.unlink(missing_ok=True)
        except OSError as cleanup_error:
            print(f"Could not remove partial output: {cleanup_error}", file=sys.stderr)
        print(f"Output {args.output}: {error}", file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
