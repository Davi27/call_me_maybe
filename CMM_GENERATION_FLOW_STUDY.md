# CMM Generation Flow — Study Note

## Big picture

For each prompt, CMM follows this path:

```text
main()
  -> validate files
  -> create ModelAdapter
  -> select function name
  -> generate parameters
  -> build result object
  -> write output JSON
```

The important separation is:

```text
Python controls structure and schema.
The model supplies constrained token choices.
```

---

## 1. `main()` starts the program

File: `main.py`

`main()` is the orchestrator. It does not generate tokens itself.

It:

1. Parses command-line arguments.
2. Creates the output directory if needed.
3. Removes an old output file.
4. Loads and validates function definitions.
5. Loads and validates prompts.
6. Creates the model adapter.
7. Processes every prompt.
8. Writes the final results file.

Default paths:

```text
data/input/functions_definition.json
data/input/function_calling_tests.json
data/output/function_calling_results.json
```

---

## 2. Prepare the output path

```python
output_path = Path(args.output)
output_path.parent.mkdir(exist_ok=True)
output_path.unlink(missing_ok=True)
```

Purpose:

- `Path(args.output)` represents the output file path.
- `.parent` represents its directory, such as `data/output`.
- `mkdir(exist_ok=True)` creates the directory if it does not exist.
- `unlink(missing_ok=True)` removes an old output file, but does nothing if there is no old file.

This prevents an old result from being confused with the current run.

---

## 3. Validate function definitions

`main()` loads the function definitions and calls:

```python
extract_parameter_types(func)
```

File: `definitions.py`

It checks:

- the function is a dictionary;
- the name is a non-empty string;
- the description is a non-empty string;
- the parameters value is a dictionary;
- each parameter has a valid name;
- each type is supported.

Supported types:

```text
string
number
boolean
```

Example result:

```python
{
    "a": "number",
    "b": "number",
}
```

---

## 4. Validate prompt input

Each input item must be a dictionary containing a string called `prompt`.

Example:

```json
{
  "prompt": "What is the sum of 2 and 3?"
}
```

Invalid files stop the program before model execution begins.

---

## 5. Create `ModelAdapter`

File: `model_adapter.py`

```python
adapter = ModelAdapter()
```

`ModelAdapter` is the project-owned boundary around `llm_sdk`.

It exposes three operations:

```python
encode_prompt(prompt)
score_ids(ids)
decode_id(token_id)
```

Purpose of each method:

```text
encode_prompt()
text -> token IDs

score_ids()
token IDs -> logits for the next token

decode_id()
token ID -> text piece
```

The rest of the project uses the adapter instead of calling the SDK model directly.

---

# Function-name generation

## 6. `select_function_name()`

File: `function_selection.py`

`main()` calls:

```python
chosen_name = select_function_name(
    adapter,
    item["prompt"],
    funcs,
    10,
)
```

The `10` is the maximum number of name-generation steps.

`select_function_name()` performs two tasks:

1. Narrows the legal function-name list.
2. Sends the request through constrained name generation.

---

## 7. Score the functions

Function:

```python
function_keyword_score(request, func)
```

It:

1. Extracts words from the request.
2. Removes common stop words.
3. Extracts words from the function name and description.
4. Counts shared words.

Example:

```text
Request:  Greet shrek
Function: fn_greet - Generate a greeting message
Shared word: greet
Score: 1
```

The names with the highest score become `best_names`.

Example:

```python
best_names = ["fn_greet"]
```

If there is a tie, several names remain:

```python
best_names = [
    "fn_first_operation",
    "fn_second_operation",
]
```

Important current design note: this keyword-scoring step is a heuristic candidate-narrowing step. The subject requires the function to be chosen using the LLM, so this is an area that needs review.

---

## 8. Build the function-selection context

Function:

```python
build_function_selection_context(request, funcs)
```

File: `context_builder.py`

It creates text containing:

- available function names;
- descriptions;
- the original request;
- the beginning of the function name: `fn_`.

Conceptual context:

```text
Available Functions:
fn_add_numbers - Add two numbers together.
fn_greet - Generate a greeting message.

Request:
Greet shrek

Name:
fn_
```

Then:

```python
context_ids = adapter.encode_prompt(context)
```

The text becomes token IDs for the model.

---

## 9. `generate_function_name()`

File: `generate_name.py`

Call:

```python
generate_function_name(
    adapter,
    context_ids,
    "fn_",
    best_names,
    max_steps,
)
```

Initial state:

```python
partial_name = "fn_"
```

The function then repeats a constrained token loop.

---

## 10. Get model logits

```python
logits = adapter.score_ids(context_ids)
```

The model returns a score for every vocabulary token.

The score answers:

> How suitable is this token as the next token for the current context?

---

## 11. Build candidate records

Function:

```python
build_candidate_records(...)
```

File: `candidates.py`

It combines the important information for every token:

```python
{
    "id": token_id,
    "score": token_score,
    "piece": decoded_text,
}
```

Keeping these values together lets the program compare scores while knowing exactly which text each token would add.

---

## 12. Filter illegal name continuations

Function:

```python
candidate_filter(
    partial_name,
    allowed_names,
    records,
)
```

The filter keeps only tokens that preserve at least one allowed name prefix.

Example:

```text
Current name: fn_
Token piece:  g
Result:       fn_g
```

If an allowed name begins with `fn_g`, the token is legal.

But:

```text
Current name: fn_
Token piece:  square
Result:       fn_square
```

is rejected if no allowed name starts with `fn_square`.

This prevents the model from leaving the set of allowed function names.

---

## 13. Select and append the best legal token

```python
selected_record = max(
    legal_records,
    key=lambda record: record["score"],
)
```

The highest-scoring legal token is selected.

Then two things are updated:

```python
context_ids.append(selected_id)
partial_name += selected_piece
```

The token is added to:

1. the next model context;
2. the generated function name.

The loop repeats:

```text
get logits
  -> build records
  -> filter legal tokens
  -> select highest legal score
  -> append token
  -> repeat
```

---

## 14. Finish or fail

The function returns when:

```python
partial_name in allowed_names
```

Possible failures:

```python
NoLegalContinuationError
```

No token can continue the current partial name.

```python
StepLimitExceededError
```

The name was not completed within the allowed number of steps.

---

# Parameter generation

## 15. Find the selected function definition

Back in `main()`:

```python
func_def = next(
    (func for func in funcs if func["name"] == chosen_name),
    None,
)
```

The selected name is matched with its complete definition.

Example:

```python
{
    "name": "fn_add_numbers",
    "description": "Add two numbers together.",
    "parameters": {
        "a": {"type": "number"},
        "b": {"type": "number"}
    }
}
```

---

## 16. `generate_argument_object()`

File: `generate_param.py`

`main()` calls:

```python
parameters = generate_argument_object(
    adapter,
    item["prompt"],
    func_def,
    10,
)
```

The function starts with:

```python
arguments = {}
```

Python owns the parameter object and its keys. The model does not create the complete object.

---

## 17. Extract request candidates

String candidates:

```python
extract_string_candidates(request)
```

File: `substitute_accuracy.py`

Number candidates:

```python
extract_number_candidates(request)
```

File: `generate_number.py`

Example:

```text
What is the sum of 2 and 3?
```

produces:

```python
["2", "3"]
```

---

## 18. Map numbers directly when possible

If the number of numeric parameters equals the number of numbers in the request:

```python
map_ordered_number_arguments(
    parameter_names,
    number_candidates,
)
```

Example:

```python
parameter_names = ["a", "b"]
number_candidates = ["2", "3"]
```

Result:

```python
{
    "a": 2,
    "b": 3,
}
```

These values are inserted directly into the Python dictionary.

---

## 19. Handle special string parameters

Some parameter names identify a role.

Source strings:

```python
longest_string_candidate()
```

Replacement strings:

```python
replacement_string_candidate()
```

Regular-expression strings:

```python
regex_string_candidate()
```

These helpers extract values from the request and insert them directly into `arguments`.

These are Python extraction paths, not model-generation paths.

---

## 20. Build a context for a remaining value

For a parameter not already filled, the program builds a context containing:

- the original request;
- the selected function;
- the parameter name;
- the parameter type.

Then:

```python
context_ids = adapter.encode_prompt(field_context)
```

The current type decides which generator is called.

---

## 21. Dispatch by parameter type

```python
if kind == "boolean":
    generate_boolean_argument(...)
elif kind == "string":
    generate_string_argument(...)
elif kind == "number":
    generate_number_argument(...)
```

Each generator follows the same general process:

```text
get logits
  -> build token records
  -> remove invalid prefixes
  -> select highest legal token
  -> append token
  -> check completion
```

---

## 22. String generation

Function:

```python
generate_string_argument()
```

It checks two constraints:

1. The text must remain valid JSON-string syntax.
2. The text must remain a prefix of an allowed candidate.

Functions used:

```python
json_string_prefix_status()
is_string_prefix_of_candidate()
```

Example candidate:

```text
hello
```

Legal partial values:

```text
"
"h
"he
"hel
"hello
"hello"
```

When complete, the scalar text is converted with:

```python
json.loads(text)
```

The result is a Python string.

---

## 23. Boolean generation

Function:

```python
generate_boolean_argument()
```

Legal values are restricted to prefixes of:

```text
true
false
```

Examples:

```text
t       -> incomplete
tr      -> incomplete
true    -> complete
x       -> invalid
```

When complete, the value is converted into a Python `bool`.

---

## 24. Number generation

Function:

```python
generate_number_argument()
```

File: `generate_number.py`

`json_number_prefix_status()` checks valid number states, including:

- optional minus sign;
- digits;
- optional decimal point;
- decimal digits;
- valid termination.

Examples:

```text
-
-1
-1.
-1.5
-1.5,
```

When the number is complete, it is parsed into an `int` or `float`.

---

## 25. Add each value to the Python object

After a type-specific generator returns:

```python
arguments[param_name] = value
```

Example:

```python
arguments["name"] = "shrek"
arguments["enabled"] = True
arguments["amount"] = 42
```

When all required parameters are handled:

```python
return arguments
```

---

# Final result

## 26. Build one output entry

Back in `main()`:

```python
results.append(
    {
        "prompt": item["prompt"],
        "name": chosen_name,
        "parameters": parameters,
    }
)
```

Python creates the complete output structure:

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

---

## 27. Write the output file

After all prompts are processed:

```python
json.dump(results, file, indent=2, allow_nan=False)
```

The result is written to:

```text
data/output/function_calling_results.json
```

If writing fails, the program deletes the partial output file so it cannot be mistaken for a valid result.

---

# Main concepts to remember

## Model adapter

Connects project code to the SDK:

```text
text <-> token IDs <-> model logits
```

## Candidate records

Keep token ID, score, and decoded text together.

## Constrained decoding

The model may choose only tokens that preserve a valid result.

## Prefix validation

The program checks whether a partial output can still become a valid final value.

## Python-owned structure

Python creates:

- output entries;
- parameter keys;
- the parameter dictionary;
- the output file.

## Model-generated content

The model participates in constrained token selection for:

- function-name generation;
- string values that use the generator;
- number values that use the generator;
- boolean values.

## Important current limitation

`function_keyword_score()` narrows the candidate names before constrained generation. The subject says the function should be chosen using the LLM rather than heuristics, so this part requires further review.
