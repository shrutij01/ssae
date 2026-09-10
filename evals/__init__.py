"""
Evaluation infrastructure for SSAE and SAE models.

Submodules:
    config          EvalConfig dataclass
    common          Model discovery, embedding loading, result I/O
    model_loading   Unified model loading (SSAE, SAE, external)
    concept_mapping Concept-to-feature mapping and scoring
    steering        Hook-based steering generation
    generation      Text generation with per-token diagnostics
    tables          LaTeX table generation
    metrics/        Metric implementations by category
    external/       Third-party benchmark adapters (AxBench, SAEBench)
"""
