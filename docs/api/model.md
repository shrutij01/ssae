# Model API

The public model name records the input family. An `SSAE` is trained on paired
differences (`delta_z`); an `SAE` is trained on raw, individual activations.
Activation and optimization are independent axes:

```python
from ssae import SAE, SSAE, JumpReLU, ReLU, TopK

model = SSAE(
    rep_dim=768,
    hid=2048,
    activation=ReLU(),
    optimization="constrained",
    constraint="l1",
)

topk_ablation = SSAE(
    rep_dim=768,
    hid=2048,
    activation=TopK(k=32),
    optimization="penalized",
    constraint="none",
)

raw_baseline = SAE(
    rep_dim=768,
    hid=2048,
    activation=JumpReLU(),
    optimization="penalized",
    constraint="none",
)
```

Every model exposes `model.model_spec`, whose stable axes are:

| Axis | Values |
|---|---|
| `family` | `SAE`, `SSAE` |
| `activation` | `linear`, `relu`, `softshrink`, `topk`, `jumprelu` |
| `optimization` | `penalized`, `constrained` |
| `constraint` | `none`, `l1`, `step_l0` |

Penalized models always have `constraint="none"`; their regularizer or
structural activation is not a Cooper constraint. `DictLinearAE` and the
historical `ssae.sae.SAE` remain available only for checkpoint and import
compatibility. Use `normalize_model_spec(...)` to translate historical names
such as `sae_relu_diff` into the canonical four-axis identity.

## Reference

::: ssae.model
