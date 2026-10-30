def json_boolean_prefix_status(text: str) -> str:
    if not text:
        return "incomplete"
    accept = ["false", "true"]
    for _ in text:
        if text in accept:
            return "complete"
        if not any(target.startswith(text) for target in accept):
            return "invalid"
        else:
            continue
    return "incomplete"


def json_number_prefix_status(text: str) -> str:
    i = 0
    is_float = False
    last = len(text) - 1
    nbrs = "0123456789"
    enders = " ,"
    if not text:
        return "incomplete"
    if text[i] not in "-0123456789":
        return "invalid"
    if i == last and text[i] in "-":
        return "incomplete"
    if i < last and text[i] == "0":
        if text[i + 1] not in "., ":
            return "invalid"
    elif i < last and text[i] == "-" and text[i + 1] == "0":
        if last > 1 and text[i + 2] not in "., ":
            return "invalid"

    for c in text[1:]:
        i += 1
        if c == "." and not is_float:
            if text[i - 1] not in nbrs:
                return "invalid"
            if i == last:
                return "incomplete"
            is_float = True
            continue
        elif c == "." and is_float:
            return "invalid"
        if c in nbrs:
            continue
        if c in enders:
            if i != last or text[i - 1] not in nbrs:
                return "invalid"
            return "ended"
        if c not in nbrs and c not in enders:
            return "invalid"
    return "complete"


def json_string_prefix_status(text: str) -> str:
    i = 0
    escaped = False
    unicode_left = 0
    if not text:
        return "incomplete"
    if text[0] != '"':
        return "invalid"
    for c in text[1:]:
        i += 1
        if unicode_left:
            if c not in "0123456789abcdefABCDEF":
                return "invalid"
            unicode_left -= 1
            continue

        if escaped:
            escaped = False
            if c == "u":
                unicode_left = 4
                continue
            if c not in '"\\/bfnrt':
                return "invalid"
            continue
        if c == "\\":
            escaped = True
            continue
        if c == '"':
            return "complete" if i == len(text) - 1 else "invalid"
        if ord(c) < 32:
            return "invalid"
    return "incomplete"
    """Return 'incomplete', 'complete', or 'invalid'."""

