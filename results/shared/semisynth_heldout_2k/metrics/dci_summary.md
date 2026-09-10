| Dataset | LLM | Model | Factors | DCI-D | DCI-C | DCI-I |
|:---|:---|:---|---:|---:|---:|---:|
| 2-binary | gemma-2-2b-it | SAE JUMPRELU diff (gamma=0.001) | 2000 epochs, absolute, input=none | 2 | 0.9855 ± 0.0078 | 0.6834 ± 0.0018 | 0.5517 ± 0.0000 |
| 2-binary | gemma-2-2b-it | SAE JUMPRELU raw (gamma=0.001) | 2000 epochs, absolute, input=none | 2 | 0.9821 ± 0.0083 | 0.6930 ± 0.0200 | 0.5517 ± 0.0000 |
| 2-binary | gemma-2-2b-it | SAE RELU diff (gamma=0.001) | 2000 epochs, absolute, input=none | 2 | 0.9148 ± 0.0317 | 0.6881 ± 0.0197 | 0.5402 ± 0.0163 |
| 2-binary | gemma-2-2b-it | SAE RELU raw (gamma=0.001) | 2000 epochs, absolute, input=none | 2 | 0.6153 ± 0.0208 | 0.5539 ± 0.0157 | 0.5517 ± 0.0000 |
| 2-binary | gemma-2-2b-it | SAE TOPK diff (gamma=0.001) | 2000 epochs, absolute, input=none | 2 | 0.6230 ± 0.0469 | 0.6532 ± 0.0119 | 0.5517 ± 0.0000 |
| 2-binary | gemma-2-2b-it | SAE TOPK raw (gamma=0.001) | 2000 epochs, absolute, input=none | 2 | 0.6716 ± 0.1050 | 0.7487 ± 0.0317 | 0.5517 ± 0.0000 |
| 2-binary | gemma-2-2b-it | SSAE L1 LN (t=0.005, d=2) | 2000 epochs, absolute, input=standardize | 2 | 0.9917 ± 0.0074 | 0.6802 ± 0.0245 | 0.5517 ± 0.0000 |
| 2-binary | pythia-70m-deduped | SAE JUMPRELU diff (gamma=0.001) | 2000 epochs, absolute, input=none | 2 | 0.9181 ± 0.0242 | 0.5459 ± 0.0332 | 0.5517 ± 0.0000 |
| 2-binary | pythia-70m-deduped | SAE JUMPRELU raw (gamma=0.001) | 2000 epochs, absolute, input=none | 2 | 0.9245 ± 0.0327 | 0.5036 ± 0.0178 | 0.5517 ± 0.0000 |
| 2-binary | pythia-70m-deduped | SAE RELU diff (gamma=0.001) | 2000 epochs, absolute, input=none | 2 | 0.8959 ± 0.0188 | 0.6157 ± 0.0079 | 0.5172 ± 0.0000 |
| 2-binary | pythia-70m-deduped | SAE RELU raw (gamma=0.001) | 2000 epochs, absolute, input=none | 2 | 0.8924 ± 0.0244 | 0.4234 ± 0.0207 | 0.5517 ± 0.0000 |
| 2-binary | pythia-70m-deduped | SAE TOPK diff (gamma=0.001) | 2000 epochs, absolute, input=none | 2 | 0.6246 ± 0.0518 | 0.4906 ± 0.0153 | 0.5517 ± 0.0000 |
| 2-binary | pythia-70m-deduped | SAE TOPK raw (gamma=0.001) | 2000 epochs, absolute, input=none | 2 | 0.5318 ± 0.0628 | 0.5019 ± 0.0059 | 0.5517 ± 0.0000 |
| 2-binary | pythia-70m-deduped | SSAE L1 LN (t=0.005, d=2) | 2000 epochs, absolute, input=standardize | 2 | 0.9615 ± 0.0131 | 0.5703 ± 0.0279 | 0.5402 ± 0.0163 |
| corr-binary | gemma-2-2b-it | SAE JUMPRELU diff (gamma=0.001) | 2000 epochs, absolute, input=none | 2 | 0.9998 ± 0.0002 | 0.8668 ± 0.0561 | 0.9062 ± 0.0255 |
| corr-binary | gemma-2-2b-it | SAE JUMPRELU raw (gamma=0.001) | 2000 epochs, absolute, input=none | 2 | 0.9801 ± 0.0277 | 0.8709 ± 0.0332 | 0.8750 ± 0.0255 |
| corr-binary | gemma-2-2b-it | SAE RELU diff (gamma=0.001) | 2000 epochs, absolute, input=none | 2 | 0.9937 ± 0.0089 | 0.8594 ± 0.0237 | 0.8854 ± 0.0147 |
| corr-binary | gemma-2-2b-it | SAE RELU raw (gamma=0.001) | 2000 epochs, absolute, input=none | 2 | 0.9878 ± 0.0162 | 0.7506 ± 0.0276 | 0.8125 ± 0.0442 |
| corr-binary | gemma-2-2b-it | SAE TOPK diff (gamma=0.001) | 2000 epochs, absolute, input=none | 2 | 1.0000 ± 0.0000 | 0.8943 ± 0.0314 | 0.9167 ± 0.0147 |
| corr-binary | gemma-2-2b-it | SAE TOPK raw (gamma=0.001) | 2000 epochs, absolute, input=none | 2 | 0.9916 ± 0.0119 | 0.9102 ± 0.0063 | 0.9062 ± 0.0000 |
| corr-binary | gemma-2-2b-it | SSAE L1 LN (t=0.005, d=2) | 2000 epochs, absolute, input=standardize | 2 | 0.7449 ± 0.3608 | 0.8481 ± 0.0422 | 0.8125 ± 0.0920 |
| corr-binary | pythia-70m-deduped | SAE JUMPRELU diff (gamma=0.001) | 2000 epochs, absolute, input=none | 2 | 0.9908 ± 0.0046 | 0.7217 ± 0.0241 | 0.7604 ± 0.0147 |
| corr-binary | pythia-70m-deduped | SAE JUMPRELU raw (gamma=0.001) | 2000 epochs, absolute, input=none | 2 | 0.9958 ± 0.0051 | 0.6778 ± 0.0415 | 0.7500 ± 0.0920 |
| corr-binary | pythia-70m-deduped | SAE RELU diff (gamma=0.001) | 2000 epochs, absolute, input=none | 2 | 0.9470 ± 0.0250 | 0.6819 ± 0.0374 | 0.7500 ± 0.0442 |
| corr-binary | pythia-70m-deduped | SAE RELU raw (gamma=0.001) | 2000 epochs, absolute, input=none | 2 | 0.9817 ± 0.0101 | 0.5699 ± 0.0151 | 0.7188 ± 0.0884 |
| corr-binary | pythia-70m-deduped | SAE TOPK diff (gamma=0.001) | 2000 epochs, absolute, input=none | 2 | 0.7008 ± 0.0125 | 0.5405 ± 0.0419 | 0.7708 ± 0.0820 |
| corr-binary | pythia-70m-deduped | SAE TOPK raw (gamma=0.001) | 2000 epochs, absolute, input=none | 2 | 0.7949 ± 0.0164 | 0.5857 ± 0.0369 | 0.7500 ± 0.0255 |
| corr-binary | pythia-70m-deduped | SSAE L1 LN (t=0.005, d=2) | 2000 epochs, absolute, input=standardize | 2 | 0.9929 ± 0.0072 | 0.7100 ± 0.0623 | 0.6771 ± 0.0896 |
| eng-french | gemma-2-2b-it | SAE JUMPRELU diff (gamma=0.001) | 2000 epochs, absolute, input=none | 1 | N/A | 0.8438 ± 0.0284 | 0.9583 ± 0.0589 |
| eng-french | gemma-2-2b-it | SAE JUMPRELU raw (gamma=0.001) | 2000 epochs, absolute, input=none | 1 | N/A | 0.8721 ± 0.0686 | 1.0000 ± 0.0000 |
| eng-french | gemma-2-2b-it | SAE RELU diff (gamma=0.001) | 2000 epochs, absolute, input=none | 1 | N/A | 0.6593 ± 0.0040 | 1.0000 ± 0.0000 |
| eng-french | gemma-2-2b-it | SAE RELU raw (gamma=0.001) | 2000 epochs, absolute, input=none | 1 | N/A | 0.8061 ± 0.0654 | 0.8750 ± 0.0000 |
| eng-french | gemma-2-2b-it | SAE TOPK diff (gamma=0.001) | 2000 epochs, absolute, input=none | 1 | N/A | 0.7783 ± 0.0321 | 0.9583 ± 0.0589 |
| eng-french | gemma-2-2b-it | SAE TOPK raw (gamma=0.001) | 2000 epochs, absolute, input=none | 1 | N/A | 0.9536 ± 0.0172 | 1.0000 ± 0.0000 |
| eng-french | gemma-2-2b-it | SSAE L1 LN (t=0.005, d=2) | 2000 epochs, absolute, input=standardize | 1 | N/A | 0.8794 ± 0.0587 | 0.8750 ± 0.1021 |
| eng-french | pythia-70m-deduped | SAE JUMPRELU diff (gamma=0.001) | 2000 epochs, absolute, input=none | 1 | N/A | 0.8137 ± 0.0654 | 0.7083 ± 0.1179 |
| eng-french | pythia-70m-deduped | SAE JUMPRELU raw (gamma=0.001) | 2000 epochs, absolute, input=none | 1 | N/A | 0.6651 ± 0.0712 | 0.7083 ± 0.1179 |
| eng-french | pythia-70m-deduped | SAE RELU diff (gamma=0.001) | 2000 epochs, absolute, input=none | 1 | N/A | 0.8451 ± 0.1376 | 0.7500 ± 0.1021 |
| eng-french | pythia-70m-deduped | SAE RELU raw (gamma=0.001) | 2000 epochs, absolute, input=none | 1 | N/A | 0.5577 ± 0.0383 | 0.4167 ± 0.1559 |
| eng-french | pythia-70m-deduped | SAE TOPK diff (gamma=0.001) | 2000 epochs, absolute, input=none | 1 | N/A | 0.7652 ± 0.0607 | 0.7500 ± 0.2700 |
| eng-french | pythia-70m-deduped | SAE TOPK raw (gamma=0.001) | 2000 epochs, absolute, input=none | 1 | N/A | 0.7104 ± 0.0621 | 0.6250 ± 0.2700 |
| eng-french | pythia-70m-deduped | SSAE L1 LN (t=0.005, d=2) | 2000 epochs, absolute, input=standardize | 1 | N/A | 0.7717 ± 0.0719 | 0.7500 ± 0.2041 |
| eng-german | gemma-2-2b-it | SAE JUMPRELU diff (gamma=0.001) | 2000 epochs, absolute, input=none | 1 | N/A | 0.6652 ± 0.0039 | 0.8333 ± 0.0589 |
| eng-german | gemma-2-2b-it | SAE JUMPRELU raw (gamma=0.001) | 2000 epochs, absolute, input=none | 1 | N/A | 0.6924 ± 0.0197 | 0.7500 ± 0.0000 |
| eng-german | gemma-2-2b-it | SAE RELU diff (gamma=0.001) | 2000 epochs, absolute, input=none | 1 | N/A | 0.6456 ± 0.0067 | 0.8750 ± 0.0000 |
| eng-german | gemma-2-2b-it | SAE RELU raw (gamma=0.001) | 2000 epochs, absolute, input=none | 1 | N/A | 0.8102 ± 0.0705 | 0.7500 ± 0.1021 |
| eng-german | gemma-2-2b-it | SAE TOPK diff (gamma=0.001) | 2000 epochs, absolute, input=none | 1 | N/A | 0.8706 ± 0.0677 | 0.7917 ± 0.1179 |
| eng-german | gemma-2-2b-it | SAE TOPK raw (gamma=0.001) | 2000 epochs, absolute, input=none | 1 | N/A | 0.7961 ± 0.0023 | 0.8333 ± 0.0589 |
| eng-german | gemma-2-2b-it | SSAE L1 LN (t=0.005, d=2) | 2000 epochs, absolute, input=standardize | 1 | N/A | 0.8510 ± 0.0950 | 0.7500 ± 0.0000 |
| eng-german | pythia-70m-deduped | SAE JUMPRELU diff (gamma=0.001) | 2000 epochs, absolute, input=none | 1 | N/A | 0.8399 ± 0.0112 | 0.8333 ± 0.0589 |
| eng-german | pythia-70m-deduped | SAE JUMPRELU raw (gamma=0.001) | 2000 epochs, absolute, input=none | 1 | N/A | 0.7852 ± 0.0363 | 0.5000 ± 0.1021 |
| eng-german | pythia-70m-deduped | SAE RELU diff (gamma=0.001) | 2000 epochs, absolute, input=none | 1 | N/A | 0.8524 ± 0.0663 | 0.7917 ± 0.1559 |
| eng-german | pythia-70m-deduped | SAE RELU raw (gamma=0.001) | 2000 epochs, absolute, input=none | 1 | N/A | 0.6289 ± 0.0342 | 0.4167 ± 0.1559 |
| eng-german | pythia-70m-deduped | SAE TOPK diff (gamma=0.001) | 2000 epochs, absolute, input=none | 1 | N/A | 0.7359 ± 0.0934 | 0.8333 ± 0.0589 |
| eng-german | pythia-70m-deduped | SAE TOPK raw (gamma=0.001) | 2000 epochs, absolute, input=none | 1 | N/A | 0.6990 ± 0.0663 | 0.5000 ± 0.1021 |
| eng-german | pythia-70m-deduped | SSAE L1 LN (t=0.005, d=2) | 2000 epochs, absolute, input=standardize | 1 | N/A | 0.8007 ± 0.0662 | 0.6250 ± 0.1021 |
| masc-fem-eng | gemma-2-2b-it | SAE JUMPRELU diff (gamma=0.001) | 2000 epochs, absolute, input=none | 1 | N/A | 0.8747 ± 0.0337 | 0.7917 ± 0.0589 |
| masc-fem-eng | gemma-2-2b-it | SAE JUMPRELU raw (gamma=0.001) | 2000 epochs, absolute, input=none | 1 | N/A | 0.9028 ± 0.0137 | 0.8750 ± 0.1021 |
| masc-fem-eng | gemma-2-2b-it | SAE RELU diff (gamma=0.001) | 2000 epochs, absolute, input=none | 1 | N/A | 0.6748 ± 0.0014 | 1.0000 ± 0.0000 |
| masc-fem-eng | gemma-2-2b-it | SAE RELU raw (gamma=0.001) | 2000 epochs, absolute, input=none | 1 | N/A | 0.8322 ± 0.0268 | 0.5833 ± 0.2125 |
| masc-fem-eng | gemma-2-2b-it | SAE TOPK diff (gamma=0.001) | 2000 epochs, absolute, input=none | 1 | N/A | 0.7008 ± 0.0241 | 0.7500 ± 0.2041 |
| masc-fem-eng | gemma-2-2b-it | SAE TOPK raw (gamma=0.001) | 2000 epochs, absolute, input=none | 1 | N/A | 0.8378 ± 0.0567 | 0.9167 ± 0.0589 |
| masc-fem-eng | gemma-2-2b-it | SSAE L1 LN (t=0.005, d=2) | 2000 epochs, absolute, input=standardize | 1 | N/A | 0.8911 ± 0.0988 | 0.9583 ± 0.0589 |
| masc-fem-eng | pythia-70m-deduped | SAE JUMPRELU diff (gamma=0.001) | 2000 epochs, absolute, input=none | 1 | N/A | 0.8643 ± 0.0174 | 0.8333 ± 0.0589 |
| masc-fem-eng | pythia-70m-deduped | SAE JUMPRELU raw (gamma=0.001) | 2000 epochs, absolute, input=none | 1 | N/A | 0.8140 ± 0.0667 | 0.7500 ± 0.1021 |
| masc-fem-eng | pythia-70m-deduped | SAE RELU diff (gamma=0.001) | 2000 epochs, absolute, input=none | 1 | N/A | 0.7380 ± 0.0531 | 0.8750 ± 0.1021 |
| masc-fem-eng | pythia-70m-deduped | SAE RELU raw (gamma=0.001) | 2000 epochs, absolute, input=none | 1 | N/A | 0.7566 ± 0.0903 | 0.5417 ± 0.1179 |
| masc-fem-eng | pythia-70m-deduped | SAE TOPK diff (gamma=0.001) | 2000 epochs, absolute, input=none | 1 | N/A | 0.6614 ± 0.0891 | 0.4167 ± 0.0589 |
| masc-fem-eng | pythia-70m-deduped | SAE TOPK raw (gamma=0.001) | 2000 epochs, absolute, input=none | 1 | N/A | 0.6711 ± 0.1317 | 0.6667 ± 0.0589 |
| masc-fem-eng | pythia-70m-deduped | SSAE L1 LN (t=0.005, d=2) | 2000 epochs, absolute, input=standardize | 1 | N/A | 0.7561 ± 0.0486 | 0.5833 ± 0.1559 |
