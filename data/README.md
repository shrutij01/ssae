# Data sources and licenses

The repository MIT License covers the project's original loaders, generators,
small controlled vocabularies, and categorical data. It does not relicense
third-party corpora or model weights.

| Dataset | Distribution here | Source and terms |
|---|---|---|
| Semi-synthetic vocabularies and categorical contrasts | Included | Project-generated; covered by the repository MIT License |
| Sycophancy | Included as `behavioral/sycophancy.json` | Adapted from [Anthropic evals](https://github.com/anthropics/evals/tree/main/sycophancy), licensed [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/); packaged here as one JSON collection |
| Refusal | Not included | Expected at `behavioral/refusal.json`; the research source is [LM-exp](https://github.com/nrimsky/LM-exp/tree/main/datasets/refusal), whose repository does not currently declare redistribution terms |
| OOD probes | Not included | Expected below `oodprobe/`; available from [`timaeus/lang5_probes`](https://huggingface.co/datasets/timaeus/lang5_probes), whose dataset card does not currently declare a license |
| TruthfulQA | Fetched by the loader | [TruthfulQA on Hugging Face](https://huggingface.co/datasets/truthfulqa/truthful_qa); follow its dataset card and upstream terms |
| Bias in Bios | Fetched by the loader | [`LabHC/bias_in_bios`](https://huggingface.co/datasets/LabHC/bias_in_bios); follow its dataset card and upstream terms |
| AxBench | Fetched by its evaluator | [`pyvene/axbench-concept500`](https://huggingface.co/datasets/pyvene/axbench-concept500); follow its dataset card and upstream terms |

To use an unbundled dataset, download it only if its upstream terms permit your
use and place it at the path shown above. Generated activations and checkpoints
belong below `SSAE_DATA_ROOT`, never in Git.
