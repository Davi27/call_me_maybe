def build_function_selection_context(request: str, funcs: list[dict]) -> str:
    """Build the prompt used to generate a function name.

    Args:
        request: The natural-language request.
        funcs: Available function definitions.

    Returns:
        The complete function-selection context.
    """
    context = "Choose the function that best matches the request.\n"
    context += "Return only one available function name.\n\n"
    context += "Available Functions:\n"

    for func in funcs:
        context += f"{func['name']} - {func['description']}\n"

    context += f"\nRequest:\n{request}\nName:\nfn_"
    return context
