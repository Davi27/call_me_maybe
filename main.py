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
