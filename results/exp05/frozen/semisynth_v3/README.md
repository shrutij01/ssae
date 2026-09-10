# Semi-synthetic feature-atlas v3: selection and stability

Concept eligibility uses a joint studentized factorial model with BH FDR on `cfc_train`. Cross-seed response recurrence, decoder recurrence, empirical coupling, and strong atom identity are reported separately.

| Family | Method | Selected | Null-control selected | Response matches | Strong geometry | Stable atoms | Aggregate direction cosine |
|---|---|---:|---:|---:|---:|---:|---:|
| SAE | JumpReLU | 22065 | 1 | 5240 | 0 | 0 | 0.975 |
| SAE | ReLU | 163 | 4 | 109 | 108 | 40 | 0.984 |
| SAE | TopK | 750 | 2 | 344 | 8 | 4 | 0.955 |
| SSAE | JumpReLU | 36605 | 0 | 12192 | 0 | 0 | 0.919 |
| SSAE | Linear const. | 16817 | 0 | 11420 | 54 | 32 | 0.957 |
| SSAE | ReLU | 41257 | 0 | 39081 | 6191 | 2387 | 0.953 |
| SSAE | TopK | 1040 | 1 | 343 | 8 | 5 | 0.867 |

A response match without strong decoder geometry is functional recurrence, not stable individual-atom identity. Models with zero stable atoms remain in every table.
