# Third-party notices

The SSAE source distribution includes or adapts small portions of third-party
software. Those portions remain subject to their original licenses.

## DisentanglementLib DCI metrics

`evals/metrics/dci.py` adapts the DCI implementation from
[DisentanglementLib](https://github.com/google-research/disentanglement_lib),
Copyright 2018 The DisentanglementLib Authors. It is distributed under the
Apache License 2.0; a complete copy is included at
[`LICENSES/Apache-2.0.txt`](LICENSES/Apache-2.0.txt).

## ICE-BeeM correlation metrics

`evals/metrics/correlation.py` and its deprecated compatibility copy
`utils/metrics.py` adapt correlation-metric code from
[ICE-BeeM](https://github.com/ilkhem/icebeem), distributed under the MIT
License:

> Copyright (c) 2020 Ilyes Khemakhem

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

## Data and model artifacts

The repository's MIT License covers the original code and documentation in
this repository. It does not relicense third-party datasets, pretrained model
weights, or other artifacts fetched from external providers. Users are
responsible for following the applicable upstream licenses and terms. See
[`data/README.md`](data/README.md) for the source-level inventory.

### Anthropic sycophancy evaluations

`data/behavioral/sycophancy.json` is adapted from the
[Anthropic evaluations repository](https://github.com/anthropics/evals/tree/main/sycophancy)
and packaged here as a single JSON collection. The source dataset is licensed
under [Creative Commons Attribution 4.0 International](https://creativecommons.org/licenses/by/4.0/).
Attribution: Anthropic. Changes: collection and JSON packaging for the SSAE
evaluation loader.
