import json


def build_function_selection_context(request: str, funcs: list[dict]) -> str:
    context = "Choose the function that best matches the request.\n"
    context += "Return only one available function name.\n\n"
    context += "Available Functions:\n"

    for func in funcs:
        context += f"{func['name']} - {func['description']}\n"

    context += f"\nRequest:\n{request}\nName:\nfn_"
    return context


def build_argument_object_context(
    request: str,
    func_def: dict,
) -> str:
    context = (
        "Generate the complete argument object for the selected function.\n"
        "Return exactly one valid JSON object.\n\n"
        f"Request:\n{request}\n\n"
        f"Function:\n{func_def['name']} - {func_def['description']}\n\n"
        "Required parameters:\n"
    )

    for param_name, details in func_def["parameters"].items():
        context += f"- {param_name}: {details['type']}\n"

    return context + "\nJSON object:\n"


def build_parameter_generation_context(
    request: str,
    func_def: dict,
    generated_arguments: dict,
    param_name: str,
    kind: str,
) -> str:
    context = build_argument_object_context(request, func_def)
    context += f"\nAlready generated arguments:\n{json.dumps(generated_arguments)}\n"
    context += f"Current parameter: {param_name} ({kind})\n"
    context += "Generate the JSON value for this parameter:\nJSON value:\n"
    return context
