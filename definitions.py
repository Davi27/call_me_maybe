def extract_parameter_types(func_def: dict) -> dict[str, str]:
    """Validate one function definition and return its parameter types.

    Args:
        func_def: A single function definition.

    Returns:
        A dictionary mapping parameter names to supported type names.

    Raises:
        ValueError: If the definition or one of its parameters is invalid.
    """
    if not isinstance(func_def, dict):
        raise ValueError("Function definition should be a dictionary")

    name = func_def.get("name")
    if not isinstance(name, str) or not name:
        raise ValueError("The name must be a non-empty string")

    description = func_def.get("description")
    if not isinstance(description, str) or not description:
        raise ValueError("The description must be a non-empty string")

    parameters = func_def.get("parameters")
    if not isinstance(parameters, dict):
        raise ValueError("Parameters should be a dictionary")

    result: dict[str, str] = {}
    for parameter_name, detail in parameters.items():
        if not isinstance(parameter_name, str) or not parameter_name:
            raise ValueError("Parameter names must be a non-empty string")
        if not isinstance(detail, dict):
            raise ValueError(
                f"{parameter_name}: parameter details should be a dictionary"
            )

        type_name = detail.get("type")
        if type_name not in {"string", "boolean", "number", "integer"}:
            raise ValueError(
                f"{parameter_name}: unsupported type {type_name!r}"
            )
        result[parameter_name] = type_name

    return result
