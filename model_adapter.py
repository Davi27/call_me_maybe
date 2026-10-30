from typing import Any, cast

import llm_sdk


class ModelAdapter:
    """Small project-owned boundary around the supplied SDK."""

    def __init__(self) -> None:
        model_factory: Any = getattr(llm_sdk, "Small_LLM_Model")
        self._sdk_model: Any = model_factory()

    def encode_prompt(self, prompt: str) -> list[int]:
        return cast(list[int], self._sdk_model.encode(prompt)[0].tolist())

    def score_ids(self, ids: list[int]) -> Any:
        return self._sdk_model.get_logits_from_input_ids(ids)

    def decode_id(self, token_id: int) -> str:
        return cast(str, self._sdk_model.decode([token_id]))
