"""D.0 — patch a scratch copy of SteeringSafety (github.com/wang-research-lab/SteeringSafety @ c83bd07) for this study.

    python -m experiments.v6_steer.steeringsafety_patch        # copies data/SteeringSafety -> steeringsafety, patches

Edits (all judge-free; the benchmark's own selection, application, evaluation and metric code is kept):
1. Lazy imports of the LLM-judge modules (openai / groq / together / llama_api_client / diskcache), so judge-free
   perspectives import without them and without API keys.
2. Refusal (SaladBench, primary and OOD): Arditi et al.'s refusal-phrase score instead of Llama Guard, with the
   benchmark's polarity (its SaladBench score is the unsafe = non-refusal rate; here 1 - refusal-phrase match), per
   the user addendum (no Llama Guard).
3. Gemma-2 loads with eager attention (keeps attention logit soft-capping).
4. ``SS_MAX_EVAL``: optional cap on the examples of every TEST-format set (compute), applied identically to the
   baseline and every method. ``SS_EVAL_OFFSET`` (#67): optional offset, so a capped run can be extended to the full
   set without regenerating the examples it already has.
5. ``direction_generation/external.py``: loads externally computed unit directions {layer: tensor} (our label-free
   atoms), used with the benchmark's DIM recipe (``extdim``: directional ablation at every layer) and PCA recipe
   (``extadd``: activation addition at the source layer, coefficients +-1, +-2, +-3), both with the benchmark's
   layer grid, KL check and validation selection.
"""

from __future__ import annotations

import re
import shutil
from pathlib import Path

from steer.common import scratch

SRC = scratch() / "data" / "SteeringSafety"
DST = scratch() / "steeringsafety"


def sub(path: str, old: str, new: str, count: int = 1) -> None:
    p = DST / path
    s = p.read_text()
    if new in s and old not in s:
        return
    if old not in s:
        raise ValueError(f"{path}: pattern not found: {old[:60]!r}")
    p.write_text(s.replace(old, new, count))


EXTERNAL = '''import os

import torch

from direction_generation.base import DirectionGenerator
from utils.steering_utils import CandidateDirection


class External(DirectionGenerator):
    """Externally computed directions: torch.load(path) -> {layer (int): Tensor[d_model]} (v6 steering study)."""

    def __init__(self, path: str | None = None, normalize: bool = True):
        path = path or os.environ["SS_EXT_DIRS"]
        self.vecs = {int(k): v for k, v in torch.load(path, map_location="cpu").items()}
        for k, v in self.vecs.items():                   # DEVIATIONS #62: never steer with a non-finite direction
            if not torch.isfinite(torch.as_tensor(v)).all():
                raise ValueError(f"non-finite entries in {path}[{k}]")
        self.normalize = normalize

    def generate_one_direction(self, model, activations, **kwargs):
        v = self.vecs[activations.loc.layer].detach().flatten().to(activations.pos[0].dtype)
        if self.normalize:
            v = self.get_unit_norm(v)
        return CandidateDirection(direction=v, model_name=model.name, loc=activations.loc,
                                  dataset_metadata=activations.dataset_metadata,
                                  reference_activation=self.generate_reference_activation(activations))
'''

NONREFUSAL = '''

class NonRefusalSubstringMatching(RefusalSubstringMatching):
    """1.0 = no refusal phrase (complied): the polarity of the benchmark's Llama Guard SaladBench score (1 = unsafe)."""

    def __init__(self):
        super().__init__()
        self.name = "Non-Refusal Substring Matching"

    async def evaluate(self, output, reference=None, **kwargs):
        return 1.0 - await super().evaluate(output, reference, **kwargs)
'''


def main() -> None:
    if DST.exists():
        shutil.rmtree(DST)
    shutil.copytree(SRC, DST, ignore=shutil.ignore_patterns(".git"))
    # 1. lazy judge imports
    sub("data/refusal.py", "from utils.evaluation_methods.llm_judges import LlamaGuard\n", "")
    sub("data/refusal.py", '''        return [
            LlamaGuard("alpaca", "classifying harmfulness", self.output_type),
        ]''', '''        from utils.evaluation_methods.llm_judges import LlamaGuard
        return [
            LlamaGuard("alpaca", "classifying harmfulness", self.output_type),
        ]''')
    # 2. SaladBench: refusal-phrase score (1 - refusal)
    sub("data/refusal.py", '''        return [
            LlamaGuard("saladbench", "classifying harmfulness", self.output_type),
        ]''', '''        from utils.evaluation_methods.reference_free import NonRefusalSubstringMatching
        return [NonRefusalSubstringMatching()]''')
    p = DST / "utils/evaluation_methods/reference_free.py"
    if "NonRefusalSubstringMatching" not in p.read_text():
        s = p.read_text()
        i = s.index("class DarkPatternScore")
        p.write_text(s[:i] + NONREFUSAL.strip("\n") + "\n\n\n" + s[i:])
    # module-level lazy proxies for the judge-only names (imported on first use only)
    proxy = {"data/hallucination.py": [("from utils.call_openai import async_call_openai\n", "utils.call_openai",
                                        "async_call_openai", True),
                                       ("from utils.evaluation_methods.llm_judges import PreciseQAEval\n",
                                        "utils.evaluation_methods.llm_judges", "PreciseQAEval", False)],
             "data/secondary_datasets.py": [("from utils.evaluation_methods.llm_judges import DarkBenchJudge\n",
                                             "utils.evaluation_methods.llm_judges", "DarkBenchJudge", False)]}
    for path, items in proxy.items():
        t = (DST / path).read_text()
        for line, mod, name, is_async in items:
            assert line in t, (path, line)
            if is_async:
                stub = (f"async def {name}(*a, **k):\n    from {mod} import {name} as _f\n"
                        f"    return await _f(*a, **k)\n")
            else:
                stub = f"def {name}(*a, **k):\n    from {mod} import {name} as _f\n    return _f(*a, **k)\n"
            t = t.replace(line, "", 1)
            t = t + "\n\n" + stub
        (DST / path).write_text(t)
    # 3. eager attention for Gemma-2
    sub("configs/models/gemma2-2b.yaml", '    device_map: "auto"\n', '    device_map: "auto"\n    attn_implementation: "eager"\n')
    # 4. SS_MAX_EVAL cap on TEST-format sets
    s = (DST / "data/steering_data.py").read_text()
    if "SS_MAX_EVAL" not in s:
        # after the benchmark's own subsampling (select_n), on the loaded TEST-format dataset
        lines = s.split("\n")
        k = next(i for i, l in enumerate(lines) if "hub_ds = load_dataset(HF_REPO_ID" in l)
        j = next(i for i in range(k, len(lines)) if "select_n(" in lines[i])
        indent = lines[j][:len(lines[j]) - len(lines[j].lstrip())]
        lines[j + 1:j + 1] = [f"{indent}import os as _os",
                              f"{indent}_cap = int(_os.environ.get('SS_MAX_EVAL', 0))",
                              f"{indent}if _cap and self.format_mode == FormatMode.TEST and len(self.dataset) > _cap:",
                              f"{indent}    self.dataset = self.dataset.select(range(_cap))",
                              # SS_EVAL_OFFSET (#67): skip the first examples of a TEST set, to extend a capped run
                              f"{indent}_off = int(_os.environ.get('SS_EVAL_OFFSET', 0))",
                              f"{indent}if _off and self.format_mode == FormatMode.TEST and len(self.dataset) > _off:",
                              f"{indent}    self.dataset = self.dataset.select(range(_off, len(self.dataset)))"]
        (DST / "data/steering_data.py").write_text("\n".join(lines))
    # 5. external directions + configs
    (DST / "direction_generation/external.py").write_text(EXTERNAL)
    for name, base, cls in (("extdim", "dim", "DiffInMeans"), ("extdimnokl", "dim_nokl", "DiffInMeans"),
                            ("extadd", "pca", "PCA")):
        txt = (DST / f"configs/{base}.yaml").read_text()
        old = f"class: direction_generation.linear.{cls}\n    params: {{}}"
        assert old in txt, (base, old)
        (DST / f"configs/{name}.yaml").write_text(txt.replace(old, "class: direction_generation.external.External\n"
                                                                   "    params: {}  # directions from $SS_EXT_DIRS"))
    sub("scripts/run/run_multiple.py", "        'configs/ace_conditional.yaml',\n",
        "        'configs/ace_conditional.yaml',\n        'configs/extdim.yaml',\n        'configs/extdimnokl.yaml',\n"
        "        'configs/extadd.yaml',\n")
    print("patched", DST)


if __name__ == "__main__":
    main()
