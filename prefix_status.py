def json_boolean_prefix_status(text: str) -> str:
    if not text:
        return "incomplete"

    accepted = ["false", "true"]
    if text in accepted:
        return "complete"
    if any(target.startswith(text) for target in accepted):
        return "incomplete"
    return "invalid"


def json_number_prefix_status(text: str) -> str:
    if not text:
        return "incomplete"

    index = 0
    is_float = False
    last = len(text) - 1
    digits = "0123456789"
    enders = " ,"

    if text[index] not in "-0123456789":
        return "invalid"
    if index == last and text[index] == "-":
        return "incomplete"
    if index < last and text[index] == "0":
        if text[index + 1] not in "., ":
            return "invalid"
    elif index < last and text[index] == "-" and text[index + 1] == "0":
        if last > 1 and text[index + 2] not in "., ":
            return "invalid"

    for character in text[1:]:
        index += 1
        if character == "." and not is_float:
            if text[index - 1] not in digits:
                return "invalid"
            if index == last:
                return "incomplete"
            is_float = True
            continue
        if character == "." and is_float:
            return "invalid"
        if character in digits:
            continue
        if character in enders:
            if index != last or text[index - 1] not in digits:
                return "invalid"
            return "ended"
        return "invalid"

    return "complete"


def json_string_prefix_status(text: str) -> str:
    if not text:
        return "incomplete"
    if text[0] != '"':
        return "invalid"

    escaped = False
    unicode_left = 0
    for index, character in enumerate(text[1:], start=1):
        if unicode_left:
            if character not in "0123456789abcdefABCDEF":
                return "invalid"
            unicode_left -= 1
            continue

        if escaped:
            escaped = False
            if character == "u":
                unicode_left = 4
            elif character not in '"\\/bfnrt':
                return "invalid"
            continue

        if character == "\\":
            escaped = True
            continue
        if character == '"':
            return "complete" if index == len(text) - 1 else "invalid"
        if ord(character) < 32:
            return "invalid"

    return "incomplete"
