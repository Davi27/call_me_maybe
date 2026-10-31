import argparse
import json
import sys
from pathlib import Path

from context_builder import (
    build_argument_object_context,
    build_function_selection_context,
    build_parameter_generation_context,
)
from candidates import build_candidate_records, candidate_filter
from definitions import extract_parameter_types
from errors import NoLegalContinuationError, StepLimitExceededError
from function_selection import function_keyword_score, select_function_name
from generate_name import generate_function_name
from generate_number import (
    extract_number_candidates,
    generate_number_argument,
    map_ordered_number_arguments,
)
from generate_param import (
    generate_argument_object,
    generate_boolean_argument,
    generate_string_argument,
)
from model_adapter import ModelAdapter
from prefix_status import (
    json_boolean_prefix_status,
    json_number_prefix_status,
    json_string_prefix_status,
)
from substitute_accuracy import (
    extract_string_candidates,
    is_string_prefix_of_candidate,
    longest_string_candidate,
    regex_string_candidate,
    replacement_string_candidate,
)


__all__ = [
    "ModelAdapter",
    "NoLegalContinuationError",
    "StepLimitExceededError",
    "build_argument_object_context",
    "build_candidate_records",
    "build_function_selection_context",
    "build_parameter_generation_context",
    "candidate_filter",
    "extract_number_candidates",
    "extract_parameter_types",
    "extract_string_candidates",
    "function_keyword_score",
    "generate_argument_object",
    "generate_boolean_argument",
    "generate_function_name",
    "generate_number_argument",
    "generate_string_argument",
    "is_string_prefix_of_candidate",
    "json_boolean_prefix_status",
    "json_number_prefix_status",
    "json_string_prefix_status",
    "longest_string_candidate",
    "main",
    "map_ordered_number_arguments",
    "regex_string_candidate",
    "replacement_string_candidate",
    "select_function_name",
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--functions_definition",
        default="data/input/functions_definition.json",
    )
    parser.add_argument(
        "--input",
        default="data/input/function_calling_tests.json",
    )
    parser.add_argument(
        "--output",
        default="data/output/function_calling_results.json",
    )
    args = parser.parse_args()

    output_path = Path(args.output)
    input_paths = {
        Path(args.input).resolve(),
        Path(args.functions_definition).resolve(),
    }
    if output_path.resolve() in input_paths:
        print("Output path must differ from input paths", file=sys.stderr)
        raise SystemExit(1)
    output_path.parent.mkdir(exist_ok=True)

    try:
        output_path.unlink(missing_ok=True)
    except OSError as error:
        print(f"Output {args.output}: {error}", file=sys.stderr)
        raise SystemExit(1)

    try:
        with open(args.functions_definition, encoding="utf-8") as file:
            funcs = json.load(file)
        if not isinstance(funcs, list) or not funcs:
            raise ValueError("Function definitions must be a non-empty list")
        for func in funcs:
            extract_parameter_types(func)
    except (OSError, ValueError) as error:
        print(
            f"Definitions {args.functions_definition}: {error}",
            file=sys.stderr,
        )
        raise SystemExit(1)

    try:
        with open(args.input, encoding="utf-8") as file:
            inputs = json.load(file)
        if not isinstance(inputs, list):
            raise ValueError("Prompt input must be a JSON array")
        for index, item in enumerate(inputs):
            if not isinstance(item, dict):
                raise ValueError(f"Prompt {index} must be an object")
            if not isinstance(item.get("prompt"), str):
                raise ValueError(
                    f"Prompt {index} must contain a string prompt"
                )
    except (OSError, ValueError) as error:
        print(f"Input {args.input}: {error}", file=sys.stderr)
        raise SystemExit(1)

    adapter = ModelAdapter()
    results = []

    for index, item in enumerate(inputs):
        try:
            chosen_name = select_function_name(
                adapter,
                item["prompt"],
                funcs,
                10,
            )
            func_def = next(
                (func for func in funcs if func["name"] == chosen_name),
                None,
            )
            if func_def is None:
                raise ValueError(f"No definition for {chosen_name}")

            parameters = generate_argument_object(
                adapter,
                item["prompt"],
                func_def,
                10,
            )
        except (
            NoLegalContinuationError,
            StepLimitExceededError,
            ValueError,
            RuntimeError,
        ) as error:
            print(
                f"Prompt {index}: {type(error).__name__}: {error}",
                file=sys.stderr,
            )
            raise SystemExit(1)

        results.append(
            {
                "prompt": item["prompt"],
                "name": chosen_name,
                "parameters": parameters,
            }
        )

    try:
        with open(args.output, "w", encoding="utf-8") as file:
            json.dump(results, file, indent=2, allow_nan=False)
    except (OSError, TypeError, ValueError) as error:
        try:
            output_path.unlink(missing_ok=True)
        except OSError as cleanup_error:
            print(
                f"Could not remove partial output: {cleanup_error}",
                file=sys.stderr,
            )
        print(f"Output {args.output}: {error}", file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
