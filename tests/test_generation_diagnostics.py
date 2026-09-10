from __future__ import annotations

from types import SimpleNamespace

import torch

from evals.generation import generate_with_diagnostics


class _Batch(dict):
    def to(self, _device):
        return self


class _Tokenizer:
    eos_token_id = 0

    def __call__(self, prompts, **_kwargs):
        assert len(prompts) == 2
        return _Batch(
            input_ids=torch.tensor([[0, 0, 8, 9], [6, 7, 8, 9]]),
            attention_mask=torch.tensor([[0, 0, 1, 1], [1, 1, 1, 1]]),
        )

    def decode(self, ids, skip_special_tokens=False):
        del skip_special_tokens
        if isinstance(ids, torch.Tensor):
            ids = ids.tolist()
        return " ".join(str(int(value)) for value in ids)


class _Model(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(()))
        self.generate_kwargs = None

    def generate(self, **kwargs):
        self.generate_kwargs = kwargs
        prefix = kwargs["input_ids"]
        suffix = torch.tensor([[4, 5], [5, 4]])
        sequences = torch.cat((prefix, suffix), dim=1)
        scores = [torch.zeros((2, 10)), torch.zeros((2, 10))]
        return SimpleNamespace(sequences=sequences, scores=scores)


def test_generation_suffix_starts_after_padded_batch_width() -> None:
    model = _Model()
    results = generate_with_diagnostics(
        model,
        _Tokenizer(),
        ["short", "a much longer prompt"],
        max_new_tokens=2,
        diagnostic_tokens=2,
        do_sample=False,
    )

    assert results[0].generated_text == "4 5"
    assert results[1].generated_text == "5 4"
    assert [row.token_id for row in results[0].token_diagnostics] == [4, 5]
    assert "temperature" not in model.generate_kwargs
    assert "top_p" not in model.generate_kwargs

