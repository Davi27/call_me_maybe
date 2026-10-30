*This project has been created as part of the 42 curriculum by davimar3.*

# Call Me Maybe

## Description

Call Me Maybe is a Python project that converts natural-language prompts into structured function calls. It reads a list of function definitions and prompts, selects an allowed function name, generates the required parameters, and writes one JSON result per prompt.

The project uses the supplied `llm_sdk` wrapper around the Qwen model. The generation code applies constrained decoding: at each generation step it filters the model vocabulary to tokens that preserve a valid function-name or JSON-value prefix. This prevents the generated result from leaving the allowed function names or the expected JSON value type.

The supported parameter types are:

- `string`
- `number`
- `boolean`

## Installation

Install the project dependencies with:

```bash
make install
```

This runs `uv sync` and creates or updates the project environment from `pyproject.toml` and `uv.lock`.

## Execution

Run the application with the default paths:

```bash
make run
```

The equivalent command is:

```bash
uv run python -m src
```

Optional input and output paths can be supplied:

```bash
uv run python -m src \\
  --functions_definition data/input/functions_definition.json \\
  --input data/input/function_calling_tests.json \\
  --output data/output/function_calling_results.json
```

The debugger entry point is:

```bash
make debug
```

The program removes an existing output file before processing. If an input file is missing or malformed, or if generation cannot complete, it reports the error on stderr and does not leave a successful output file behind.

## Input and output contract

### Function definitions

Default path:

```text
data/input/functions_definition.json
```

The file contains a JSON array. Each function definition has a name, description, and parameter dictionary. Each parameter specifies one supported type.

Example:

```json
[
  {
    "name": "fn_add_numbers",
    "description": "Add two numbers together and return their sum.",
    "parameters": {
      "a": {"type": "number"},
      "b": {"type": "number"}
    }
  }
]
```

The definitions are validated before model execution. Function names, descriptions, parameter names, parameter objects, and supported types must be present and correctly shaped.

### Prompt input

Default path:

```text
data/input/function_calling_tests.json
```

The file contains a JSON array of objects with a string `prompt` field:

```json
[
  {"prompt": "What is the sum of 2 and 3?"}
]
```

### Output

Default path:

```text
data/output/function_calling_results.json
```

The output is a JSON array. Every entry has exactly these keys:

```json
{
  "prompt": "What is the sum of 2 and 3?",
  "name": "fn_add_numbers",
  "parameters": {
    "a": 2,
    "b": 3
  }
}
```

The output preserves the original prompt order. The selected name must be present in the supplied definitions. The parameter object must contain the required parameter names, with values matching the declared types.

## Algorithm explanation

The application processes each prompt through the following stages.

### 1. Validate the inputs

The CLI loads both JSON input files and validates their top-level structure. Function definitions are checked with `extract_parameter_types()`. Missing files, invalid JSON, missing fields, unsupported parameter types, and invalid records are reported as input errors before generation starts.

### 2. Build the function-selection context

The context builder presents the request together with the available function names and descriptions. The model receives a prompt ending in the prefix `fn_`, which is the beginning shared by the supplied function names.

The current implementation first calculates a lightweight keyword score from the request and each function's name and description. Common stop words are ignored. The functions with the highest score form the legal candidate list. If several functions have the same score, all tied names remain available.

The model is still called for every selection. Even when the candidate list contains one name, `generate_function_name()` generates that name token by token rather than returning the string directly.

### 3. Generate the function name with constrained decoding

`generate_function_name()` performs a repeated next-token loop:

```text
context text
    | tokenize
    v
input token IDs
    | model inference
    v
logits for the next token
    | keep tokens that extend an allowed function name
    v
highest-scoring legal token
    | append token ID and decoded piece
    v
partial function name
```

At each step:

1. The adapter requests logits for the current token-ID context.
2. The vocabulary is converted into records containing token ID, score, and decoded text piece.
3. `candidate_filter()` removes pieces that would make the partial name impossible to complete into one of the allowed names.
4. The highest-scoring remaining token is selected.
5. Its ID is appended to the context and its decoded piece is appended to the partial name.
6. Generation stops when the partial name exactly matches an allowed function name.

Two explicit failures are used:

- `NoLegalContinuationError`: no token can continue the current partial name.
- `StepLimitExceededError`: the name was not completed within the configured number of steps.

### 4. Generate the parameter object

After selecting a function, the program retrieves its definition and creates the required parameter object.

The implementation handles values according to their declared types:

- Number parameters can use numbers extracted from the request when the number of values matches the number of numeric parameters. Otherwise, they use constrained numeric generation.
- String parameters use request-derived candidates for source strings, replacement strings, and regular-expression patterns when the parameter name identifies one of those roles. Other strings use constrained JSON-string generation.
- Boolean parameters use constrained generation over the legal prefixes of `true` and `false`.

The parameter-generation functions maintain the partial JSON value and reject tokens that would make the value invalid for the required type.

For strings, the decoder checks both JSON string syntax and whether the current text remains a prefix of an allowed candidate. For numbers, it checks signs, digits, decimal points, and termination. For booleans, only prefixes of `true` or `false` remain legal.

### 5. Write the result

After every prompt has produced a valid result, the program writes the complete result list with `json.dump(..., allow_nan=False)`. If writing fails, it removes the partial output file and reports the error.

## Design

The implementation is divided into small modules:

- `main.py`: command-line parsing, input/output handling, and orchestration.
- `model_adapter.py`: project-owned boundary around the supplied `llm_sdk` model.
- `context_builder.py`: prompts used for function and parameter generation.
- `function_selection.py`: candidate scoring and model-backed name generation.
- `generate_name.py`: constrained function-name generation.
- `generate_param.py`: constrained string, boolean, and parameter-object generation.
- `generate_number.py`: constrained numbers and numeric request extraction.
- `prefix_status.py`: JSON prefix validation for strings, numbers, and booleans.
- `substitute_accuracy.py`: request-derived string and regular-expression candidates.
- `candidates.py`: token records and legal-prefix filtering.
- `definitions.py`: function-definition validation.
- `errors.py`: explicit generation failure types.
- `src/__main__.py`: official `python -m src` entry point.

The `ModelAdapter` isolates the rest of the project from the SDK. Project code calls the adapter's public operations for encoding, logits, and decoding instead of importing model implementation details directly.

## Performance

The application performs model inference repeatedly because constrained decoding requests logits for each generated token. The main cost is loading the model and running inference; scanning the vocabulary and filtering legal token pieces is additional per-token work.

On the development machine used for verification, the default 11-prompt run completed successfully in approximately 49.5 seconds. This measurement includes model loading and depends on hardware, Python environment, model-cache state, and input size. It is not a fixed guarantee for other machines.

The implementation uses a maximum generation-step limit so an impossible or unfinished generation cannot loop indefinitely. Errors are reported instead of producing an invalid result.

## Challenges and design decisions

### Keeping generated output valid

A small language model may produce an invalid function name or malformed JSON if it is only prompted to follow a schema. The project therefore filters logits before selecting each token. Invalid prefixes are removed before they can enter the generated result.

### Preserving the function schema

The function definition is used as the source of truth for the selected function and its required parameters. The final output is checked by the test suite for exact parameter keys and declared value types.

### Separating model inference from project logic

The supplied SDK handles model operations, but the project needs ordinary Python control flow for candidate filtering, prefix validation, input validation, and output writing. The adapter boundary keeps those responsibilities separate.

### Handling request-specific values

Some prompts contain values that should be copied from the request, such as numbers, source strings, replacement strings, or regex patterns. The implementation extracts these values before falling back to constrained generation. This keeps values such as `2`, `3`, `hello`, and `NUMBERS` aligned with the prompt while still validating the final types.

### Handling failures explicitly

Generation can fail when no legal token remains or when the step limit is reached. These cases use named exceptions so the CLI can report which prompt failed and avoid leaving a misleading successful output file.

## Testing

Run the complete test suite with:

```bash
uv run --with pytest python -m pytest -q
```

The tests cover:

- candidate records and legal-prefix filtering;
- safe failure when no legal function-name continuation exists;
- step-limit failure;
- function-selection context construction;
- string, number, and boolean prefix validation;
- request-derived string and number candidates;
- function-definition validation;
- the official `python -m src` entry point;
- the compatibility `main.py` entry point;
- output count, order, keys, function-name membership, parameter names, and parameter types;
- changed function-definition schemas;
- missing and malformed input files.

The verified suite currently reports 33 passing tests.

Run the style and type checks with:

```bash
make lint
```

The lint target runs Flake8 and mypy with the project-specific exclusions and required flags.

Remove generated caches with:

```bash
make clean
```

## Real example

With the default input files, a prompt such as:

```text
What is the sum of 2 and 3?
```

produces:

```json
{
  "prompt": "What is the sum of 2 and 3?",
  "name": "fn_add_numbers",
  "parameters": {
    "a": 2,
    "b": 3
  }
}
```

A replacement prompt such as:

```text
Replace all numbers in "Hello 34 I'm 233 years old" with NUMBERS
```

produces a schema-compliant result with the source string, the `\\d+` pattern, and the replacement string:

```json
{
  "prompt": "Replace all numbers in \"Hello 34 I'm 233 years old\" with NUMBERS",
  "name": "fn_substitute_string_with_regex",
  "parameters": {
    "source_string": "Hello 34 I'm 233 years old",
    "regex": "\\d+",
    "replacement": "NUMBERS"
  }
}
```

## Known limitations

- The current implementation supports the parameter types `string`, `number`, and `boolean` only.
- Request-derived string handling recognizes roles from parameter names such as `source`, `replacement`, `regex`, and `pattern`.
- Numeric extraction uses the order of numbers in the request when the number of numeric parameters matches the number of extracted values.
- Generation quality depends on the supplied model and the descriptions in the function definitions.
- The current candidate-scoring step narrows the legal function-name list before constrained token generation. The model still generates the final name, but the quality of the initial candidate narrowing affects which names are available to it.
- Performance depends strongly on model loading and hardware.

## Resources

- [Python documentation](https://docs.python.org/3/): language reference and standard-library documentation.
- [The `json` module](https://docs.python.org/3/library/json.html): JSON parsing and serialization used by the CLI.
- [uv documentation](https://docs.astral.sh/uv/): project dependency and environment management.
- The 42 `call_me_maybe` subject: project requirements, constrained-decoding objective, Makefile rules, README requirements, and evaluation constraints.

## AI use disclosure

AI assistance was used to explain project concepts, inspect and debug implementation issues, suggest and review bounded code changes, verify commands and tests, and help structure this documentation. AI-generated suggestions were checked against the repository, the project subject, executable tests, and observed runtime behavior. The final implementation and documentation are the student's responsibility.
