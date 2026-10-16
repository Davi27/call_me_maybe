import torch

def test_prompt_to_logit() -> None:
    from main import ModelAdapter
    adapter = ModelAdapter()
    ids = adapter.encode_prompt("This is a test!")
    logits = adapter.score_ids(ids)

    assert isinstance(ids, torch.Tensor)
    assert ids.shape[0] == 1
    assert isinstance(logits, list)
    assert len(logits) == 151936