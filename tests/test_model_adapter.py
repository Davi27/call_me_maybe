
def test_prompt_to_logit() -> None:
    from main import ModelAdapter
    adapter = ModelAdapter()
    ids = adapter.encode_prompt("This is a test!")
    logits = adapter.score_ids(ids)

    assert ids
    assert all(isinstance(token_id, int) for token_id in ids)
    assert isinstance(ids, list)
    assert isinstance(logits, list)
    assert len(logits) == 151936
