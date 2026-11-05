def build_function_selection_context(request: str, funcs: list[dict]) -> str:
    """ 
        Create the context to feed the model for the generation of the 
        name of the function

        Arguments:
            request - the prompt
            funcs - a list with all function definitions

        Return:
            the full context
    """
    context = "Choose the function that best matches the request.\n"
    context += "Return only one available function name.\n\n"
    context += "Available Functions:\n"

    for func in funcs:
        context += f"{func['name']} - {func['description']}\n"

    context += f"\nRequest:\n{request}\nName:\nfn_"
    return context
