import json
import argparse
import llm_sdk
import torch


class NoLegalContinuationError(ValueError):
    pass


class StepLimitExceededError(Exception):
    pass


class ModelAdapter:
    def __init__(self):
        self._sdk_model = llm_sdk.Small_LLM_Model()

    def encode_prompt(self, prompt):
        return self._sdk_model.encode(prompt)

    def score_ids(self, ids):
        id_list = ids[0].tolist()
        return self._sdk_model.get_logits_from_input_ids(id_list)

    def decode_id(self, token_id):
        return self._sdk_model.decode([token_id])


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


def build_candidate_records(candidate_ids, logits, decode_id):
    return [
        {
            "id": candidate_id,
            "score": logits[candidate_id],
            "piece": decode_id(candidate_id),
        }
        for candidate_id in candidate_ids
    ]


def candidate_filter(current_name: str, func_names: list[str], records: list[dict]) -> list[dict]:
    ret = []
    for record in records:
        piece = record["piece"]
        if not piece:
            continue
        possibility = current_name + piece
        if any(name.startswith(possibility) for name in func_names):
            ret.append(record)
    return ret


def generate_function_name(adapter: ModelAdapter, initial_context_ids: torch.Tensor, initial_partial_name: str, allowed_names: list[str], max_steps: int) -> str:
    context_ids = initial_context_ids.clone()
    partial_name = initial_partial_name

    for _ in range(max_steps):
        logits = adapter.score_ids(context_ids)
        records = build_candidate_records(range(len(logits)), logits, adapter.decode_id)
        legal_records = candidate_filter(partial_name, allowed_names, records)
        if len(legal_records) == 0:
            raise NoLegalContinuationError("No legal continuation for this partial name")
        selected_record = max(legal_records, key=lambda record: record["score"])

        selected_id = selected_record["id"]
        selected_id_tensor = torch.tensor([[selected_id]], dtype=context_ids.dtype, device=context_ids.device)
        context_ids = torch.cat((context_ids, selected_id_tensor), dim=1)
        partial_name += selected_record["piece"]

        if partial_name in allowed_names:
            return partial_name
    raise StepLimitExceededError("No name was created in the steps length")


def generate_string_argument(adapter, initial_context_ids, max_steps):
    context_ids = initial_context_ids.clone()
    text = ""

    for _ in range(max_steps):
        logits = adapter.score_ids(context_ids)
        records = build_candidate_records(range(len(logits)), logits, adapter.decode_id)
        legal_records = [
            record for record in records
            if record["piece"]
            and json_string_prefix_status(text + record["piece"]) != "invalid"
        ]
        if len(legal_records) == 0:
            raise NoLegalContinuationError("No legal continuation for this partial string")
        selected_record = max(legal_records, key=lambda record: record["score"])

        selected_id = selected_record["id"]
        selected_id_tensor = torch.tensor([[selected_id]], dtype=context_ids.dtype, device=context_ids.device)
        context_ids = torch.cat((context_ids, selected_id_tensor), dim=1)
        text += selected_record["piece"]
        if json_string_prefix_status(text) == "complete":
            return json.loads(text)

    raise StepLimitExceededError("No string was created in the steps length")


def generate_number_argument(adapter, initial_context_ids, max_steps):
    context_ids = initial_context_ids.clone()
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
        selected_id_tensor = torch.tensor([[selected_id]], dtype=context_ids.dtype, device=context_ids.device)
        context_ids = torch.cat((context_ids, selected_id_tensor), dim=1)
        text += selected_record["piece"]
        if json_number_prefix_status(text) == "ended":
            return json.loads(text[:-1])

    raise StepLimitExceededError("No number was created in the steps length")


def generate_boolean_argument(adapter, initial_context_ids, max_steps):
    context_ids = initial_context_ids.clone()
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
        selected_id_tensor = torch.tensor([[selected_id]], dtype=context_ids.dtype, device=context_ids.device)
        context_ids = torch.cat((context_ids, selected_id_tensor), dim=1)
        text += selected_record["piece"]
        if json_boolean_prefix_status(text) == "complete":
            return json.loads(text)

    raise StepLimitExceededError("No boolean was created in the steps length")


def generate_argument_object(adapter: ModelAdapter, request: str, func_def: dict, max_steps: int) -> dict:
    parameter_types = extract_parameter_types(func_def)
    arguments = {}

    for param_name, kind in parameter_types.items():
        field_context = (
            f"Request: {request}\n"
            f"Function: {func_def['name']} - {func_def['description']}\n"
            f"Parameter: {param_name} ({kind})\n"
            "JSON value: "
        )
        context_ids = adapter.encode_prompt(field_context)
        if kind == "boolean":
            ret = generate_boolean_argument(adapter, context_ids, max_steps)
        elif kind == "string":
            ret = generate_string_argument(adapter, context_ids, max_steps)
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
    except FileNotFoundError:
        print("File not found", args.functions_definition)
        return
    except ValueError as error:
        print(args.functions_definition, error)
        return

    try:
        with open(args.input, "r") as f:
            inpt = json.load(f)
    except FileNotFoundError:
        print("File not found", args.input)
        return
    except ValueError:
        print(args.input, "contains the incorrect format")
        return

    # create the context than encode it
    context = "Functions:\n"
    for func in funcs:
        context += func["name"] + " - " + func["description"] + "\n"
    context += "\nRequest:\n" + inpt[0]["prompt"]
    context += "\nName:\nfn_"

    adapter = ModelAdapter()

    encoded_context = adapter.encode_prompt(context)

    # legal function names list
    fns = []
    for fn in funcs:
        fns.append(fn["name"])


    # generate the function name
    chosen_name = generate_function_name(adapter, encoded_context, "fn_", fns, 10)
    print(chosen_name)

    # store the values in a list. Run a loop so we can convert all prompts
    results = []

    for test in inpt:
        result = {
            "prompt": test["prompt"],
            "name": "TODO",
            "parameters": {}
        }
        results.append(result)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
