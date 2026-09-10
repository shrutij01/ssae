# Semi-synthetic held-out metrics (2,000 steps)

These results use models trained exclusively on `cfc_train` and evaluated on
the final checkpoint exclusively on `cfc_test`. Each cell is the mean ±
population standard deviation across two LLMs and three model seeds (six runs).
MCC denotes Pearson MCC.

The presentation taxonomy treats all difference-trained models as SSAEs:

- **SSAE / Linear + L1** is the existing linear constrained-optimization model
  trained with an L1 sparsity constraint.
- **SSAE / Linear + Step-L0** uses the same linear constrained-optimization
  model with the rectangular Step-L0 estimator. Its cells remain pending until
  the matched held-out three-seed runs finish.
- **SSAE / ReLU, TopK, JumpReLU** are the corresponding difference-trained
  models.
- **SAE / ReLU, TopK, JumpReLU** are trained on raw representations.

DCI-D is structurally 1.0 when a dataset has only one ground-truth factor: the
factor-importance distribution has one entry and therefore zero entropy. This
does not provide evidence of disentanglement, so presentation tables should
show DCI-D as N/A for `eng-french`, `eng-german`, and `masc-fem-eng`. DCI-D is
meaningful for `2-binary` and `corr-binary`.

## MCC

| Family | Method | eng-french | eng-german | masc-fem-eng | 2-binary | corr-binary |
|:--|:--|--:|--:|--:|--:|--:|
| SSAE | Linear + L1 | 0.671 ± 0.082 | 0.693 ± 0.051 | 0.738 ± 0.101 | 0.414 ± 0.057 | 0.637 ± 0.118 |
| SSAE | Linear + Step-L0 | pending | pending | pending | pending | pending |
| SSAE | ReLU | 0.876 ± 0.046 | 0.868 ± 0.063 | 0.865 ± 0.079 | 0.428 ± 0.021 | 0.636 ± 0.072 |
| SSAE | TopK | 0.673 ± 0.139 | 0.674 ± 0.171 | 0.508 ± 0.058 | 0.310 ± 0.072 | 0.599 ± 0.214 |
| SSAE | JumpReLU | 0.798 ± 0.076 | 0.799 ± 0.097 | 0.777 ± 0.029 | 0.437 ± 0.022 | 0.727 ± 0.097 |
| SAE | ReLU | 0.495 ± 0.161 | 0.499 ± 0.129 | 0.548 ± 0.069 | 0.218 ± 0.009 | 0.431 ± 0.070 |
| SAE | TopK | 0.739 ± 0.154 | 0.692 ± 0.194 | 0.621 ± 0.076 | 0.342 ± 0.108 | 0.651 ± 0.178 |
| SAE | JumpReLU | 0.734 ± 0.136 | 0.750 ± 0.130 | 0.736 ± 0.045 | 0.405 ± 0.059 | 0.670 ± 0.149 |

## AUC

| Family | Method | eng-french | eng-german | masc-fem-eng | 2-binary | corr-binary |
|:--|:--|--:|--:|--:|--:|--:|
| SSAE | Linear + L1 | 0.908 ± 0.059 | 0.917 ± 0.041 | 0.934 ± 0.056 | 0.781 ± 0.038 | 0.890 ± 0.061 |
| SSAE | Linear + Step-L0 | pending | pending | pending | pending | pending |
| SSAE | ReLU | 0.988 ± 0.011 | 0.984 ± 0.014 | 0.981 ± 0.019 | 0.798 ± 0.013 | 0.921 ± 0.032 |
| SSAE | TopK | 0.862 ± 0.060 | 0.872 ± 0.097 | 0.752 ± 0.033 | 0.683 ± 0.040 | 0.812 ± 0.097 |
| SSAE | JumpReLU | 0.964 ± 0.030 | 0.960 ± 0.033 | 0.957 ± 0.010 | 0.795 ± 0.014 | 0.916 ± 0.054 |
| SAE | ReLU | 0.760 ± 0.109 | 0.766 ± 0.099 | 0.827 ± 0.038 | 0.626 ± 0.009 | 0.699 ± 0.037 |
| SAE | TopK | 0.915 ± 0.097 | 0.889 ± 0.088 | 0.870 ± 0.076 | 0.710 ± 0.094 | 0.826 ± 0.090 |
| SAE | JumpReLU | 0.929 ± 0.069 | 0.922 ± 0.061 | 0.940 ± 0.030 | 0.773 ± 0.038 | 0.900 ± 0.073 |

## DCI-C

| Family | Method | eng-french | eng-german | masc-fem-eng | 2-binary | corr-binary |
|:--|:--|--:|--:|--:|--:|--:|
| SSAE | Linear + L1 | 0.826 ± 0.085 | 0.826 ± 0.086 | 0.824 ± 0.103 | 0.625 ± 0.061 | 0.779 ± 0.087 |
| SSAE | Linear + Step-L0 | pending | pending | pending | pending | pending |
| SSAE | ReLU | 0.752 ± 0.135 | 0.749 ± 0.114 | 0.706 ± 0.049 | 0.652 ± 0.039 | 0.771 ± 0.094 |
| SSAE | TopK | 0.772 ± 0.049 | 0.803 ± 0.106 | 0.681 ± 0.068 | 0.572 ± 0.082 | 0.717 ± 0.181 |
| SSAE | JumpReLU | 0.829 ± 0.053 | 0.753 ± 0.088 | 0.869 ± 0.027 | 0.615 ± 0.073 | 0.794 ± 0.084 |
| SAE | ReLU | 0.682 ± 0.135 | 0.720 ± 0.106 | 0.794 ± 0.077 | 0.489 ± 0.068 | 0.660 ± 0.093 |
| SAE | TopK | 0.832 ± 0.130 | 0.748 ± 0.068 | 0.754 ± 0.131 | 0.625 ± 0.126 | 0.748 ± 0.164 |
| SAE | JumpReLU | 0.769 ± 0.125 | 0.739 ± 0.055 | 0.858 ± 0.066 | 0.598 ± 0.097 | 0.774 ± 0.104 |

## DCI-I

| Family | Method | eng-french | eng-german | masc-fem-eng | 2-binary | corr-binary |
|:--|:--|--:|--:|--:|--:|--:|
| SSAE | Linear + L1 | 0.812 ± 0.173 | 0.688 ± 0.095 | 0.771 ± 0.221 | 0.546 ± 0.013 | 0.745 ± 0.113 |
| SSAE | Linear + Step-L0 | pending | pending | pending | pending | pending |
| SSAE | ReLU | 0.875 ± 0.144 | 0.833 ± 0.118 | 0.938 ± 0.095 | 0.529 ± 0.016 | 0.818 ± 0.075 |
| SSAE | TopK | 0.854 ± 0.221 | 0.812 ± 0.095 | 0.583 ± 0.224 | 0.552 ± 0.000 | 0.844 ± 0.094 |
| SSAE | JumpReLU | 0.833 ± 0.156 | 0.833 ± 0.059 | 0.812 ± 0.062 | 0.552 ± 0.000 | 0.833 ± 0.076 |
| SAE | ReLU | 0.646 ± 0.254 | 0.583 ± 0.212 | 0.562 ± 0.173 | 0.552 ± 0.000 | 0.766 ± 0.084 |
| SAE | TopK | 0.812 ± 0.268 | 0.667 ± 0.186 | 0.792 ± 0.138 | 0.552 ± 0.000 | 0.828 ± 0.080 |
| SAE | JumpReLU | 0.854 ± 0.168 | 0.625 ± 0.144 | 0.812 ± 0.120 | 0.552 ± 0.000 | 0.812 ± 0.092 |

Machine-readable outputs are under [`metrics/`](metrics/). The split audit is
in [`data_audit.json`](data_audit.json).
