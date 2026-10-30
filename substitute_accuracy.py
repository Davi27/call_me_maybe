
def longest_string_candidate(
    candidates: list[str],
) -> str:
    if not candidates:
        raise ValueError("No string candidates were provided")

    return max(
        candidates,
        key=len,
    )


def replacement_string_candidate(
    request: str,
    candidates: list[str],
) -> str | None:
    match = re.search(
        r"\bwith\s+(?:\"([^\"]+)\"|'([^']+)'|([A-Za-z0-9_*+-]+))",
        request,
        re.IGNORECASE,
    )

    if match is None:
        return None

    replacement = next(
        value
        for value in match.groups()
        if value is not None
    )

    if replacement in candidates:
        return replacement

    return None


def regex_string_candidate(
    request: str,
    candidates: list[str],
) -> str | None:
    lowered_request = request.lower()

    if "numbers" in lowered_request:
        return r"\d+"

    if "vowels" in lowered_request:
        return r"[aeiouAEIOU]"

    word_match = re.search(
        r"\bword\s+['\"]([^'\"]+)['\"]",
        request,
        re.IGNORECASE,
    )

    if word_match is not None:
        word = word_match.group(1)

        if word in candidates:
            return word

    return None


def is_string_prefix_of_candidate(
    text: str,
    candidates: list[str],
) -> bool:
    if not text.startswith('"'):
        return False

    if text == '""':
        return "" in candidates

    partial_text = text[1:]

    if partial_text.endswith('"'):
        partial_text = partial_text[:-1]
        return partial_text in candidates

    return any(
        candidate.startswith(partial_text)
        for candidate in candidates
    )


def extract_string_candidates(request: str) -> list[str]:
    quoted_values = re.findall(
        r'"([^"]*)"|\'([^\']*)\'',
        request,
    )

    candidates = []

    for double_quoted, single_quoted in quoted_values:
        value = double_quoted or single_quoted
        if value and value not in candidates:
            candidates.append(value)

    words = re.findall(
        r"[A-Za-z][A-Za-z0-9_]*",
        request,
    )

    if words:
        final_word = words[-1]
        if final_word not in candidates:
            candidates.append(final_word)

    return candidates