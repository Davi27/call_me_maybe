from typing import Any, Callable


def candidate_filter(
    current_name: str,
    func_names: list[str],
    records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """
        Keep token records that preserve at least one allowed name prefix.

        Arguments:
            current_name - the name generated so far
            func_names - the name of all functions
            records - the records of each token

        Return:
            a list with the records of the legal tokens
    """
    return [
        record
        for record in records
        if record["piece"]
        and any(
            name.startswith(current_name + record["piece"])
            for name in func_names
        )
    ]


def build_candidate_records(
    candidate_ids: range,
    logits: Any,
    decode_id: Callable[[int], str],
) -> list[dict[str, Any]]:
    """
        Keep each token id, score, and decoded piece together

        Arguments:
            candidate_ids - the number of tokens
            logits - the score of each token
            decode_id - the translation of the decoded token 

        Return:
            a list with the records of all tokens
    """
    return [
        {
            "id": candidate_id,
            "score": logits[candidate_id],
            "piece": decode_id(candidate_id),
        }
        for candidate_id in candidate_ids
    ]
