import torch
import pytest
from main import NoLegalContinuationError
from main import StepLimitExceededError
from main import build_candidate_records
from main import generate_function_name
from main import candidate_filter

class FakeAdapter:
    def __init__(self):
        self.seen_contexts = []

    def score_ids(self, ids):
        self.seen_contexts.append(ids[0].tolist())
        return [0, 5, 4, 99]

    def decode_id(self, token_id):
        return {0: "", 1: "a", 2: "b", 3: "X"}[token_id]


def test_candidate_records_keep_id_score_and_piece_together() -> None:
    logits = [0.0] * 13
    logits[10] = 2.5
    logits[12] = -1.0
    pieces = {10: "+", 12: "-"}

    records = build_candidate_records([10, 12], logits, pieces.__getitem__)

    assert records == [
        {"id": 10, "score": 2.5, "piece": "+"},
        {"id": 12, "score": -1.0, "piece": "-"},
    ]


def test_filter_right_behaviour() -> None:
    current_name = "CA"
    func_names = ["CALL", "CANCEL"]
    records = [
        {"id": 10, "score": 2.5, "piece": "L"},
        {"id": 12, "score": 9.0, "piece": "X"},
    ]
    scores = [record["score"] for record in records]

    ret = candidate_filter(current_name, func_names, records)

    assert max(scores) > ret[0]["score"]
    assert ret == [records[0]]


def test_selection_of_legal_candidate() -> None:

    adapter = FakeAdapter()
    initial_context_ids = torch.tensor([[7]], dtype=torch.long)

    result = generate_function_name(adapter, initial_context_ids, "fn_", ["fn_ab"], 2)

    assert result == "fn_ab"
    assert adapter.seen_contexts == [[7], [7, 1]]


def test_no_legal_continuation() -> None:
    adapter = FakeAdapter()
    context = torch.tensor([[7]], dtype=torch.long)

    with pytest.raises(NoLegalContinuationError):
        generate_function_name(adapter, context, "fn_", ["fn_z"], 2)


def test_step_limit_before_complete_name() -> None:
    adapter = FakeAdapter()
    context = torch.tensor([[7]], dtype=torch.long)

    with pytest.raises(StepLimitExceededError):
        generate_function_name(adapter, context, "fn_", ["fn_ab"], 1)