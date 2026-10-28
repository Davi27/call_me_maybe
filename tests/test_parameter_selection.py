from main import longest_string_candidate
from main import replacement_string_candidate
from main import regex_string_candidate


def test_longest_string_candidate_is_selected_as_source() -> None:
    candidates = [
        "cat",
        "dog",
        "The cat sat on the mat with another cat",
    ]

    result = longest_string_candidate(candidates)

    assert result == "The cat sat on the mat with another cat"


def test_replacement_string_candidate_follows_with() -> None:
    candidates = [
        "cat",
        "dog",
        "The cat sat on the mat with another cat",
    ]

    result = replacement_string_candidate(
        "Substitute the word 'cat' with 'dog' in "
        "'The cat sat on the mat with another cat'",
        candidates,
    )

    assert result == "dog"


def test_unquoted_replacement_string_candidate_follows_with() -> None:
    candidates = [
        "Hello 34 I'm 233 years old",
        "NUMBERS",
    ]

    result = replacement_string_candidate(
        'Replace all numbers in "Hello 34 I\'m 233 years old" with NUMBERS',
        candidates,
    )

    assert result == "NUMBERS"


def test_regex_candidate_for_numbers() -> None:
    result = regex_string_candidate(
        'Replace all numbers in "Hello 34" with NUMBERS',
        ["Hello 34", "NUMBERS"],
    )

    assert result == r"\d+"


def test_regex_candidate_for_vowels() -> None:
    result = regex_string_candidate(
        "Replace all vowels in 'Programming is fun' with asterisks",
        ["Programming is fun", "asterisks"],
    )

    assert result == r"[aeiouAEIOU]"


def test_regex_candidate_for_quoted_word() -> None:
    result = regex_string_candidate(
        "Substitute the word 'cat' with 'dog' in 'The cat sat'",
        ["cat", "dog", "The cat sat"],
    )

    assert result == "cat"