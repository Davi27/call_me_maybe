from main import extract_parameter_types


def test_extract_parameter_types() -> None:
    arg = {
        "name": "fn_send",
        "description": "Send a message",
        "parameters": {"recipient": {"type": "string"}},
        "returns": {"type": "boolean"},
    }
    ret = extract_parameter_types(arg)

    assert ret == {"recipient": "string"}