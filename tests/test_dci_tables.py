from evals.tables import generate_dci_markdown_table


def _summary(dataset, n_factors):
    return {
        "dataset": dataset,
        "llm": "gemma",
        "model_label": "SSAE",
        "n_factors": n_factors,
        "dci_d_mean": 1.0 if n_factors == 1 else 0.75,
        "dci_d_std": 0.0 if n_factors == 1 else 0.05,
        "dci_c_mean": 0.6,
        "dci_c_std": 0.1,
        "dci_i_mean": 0.9,
        "dci_i_std": 0.02,
    }


def test_single_factor_dci_d_is_rendered_na():
    table = generate_dci_markdown_table([_summary("refusal", 1)])

    assert "| refusal | gemma | SSAE | 1 | N/A |" in table
    assert "0.6000 ± 0.1000" in table
    assert "1.0000 ± 0.0000" not in table


def test_multifactor_dci_d_is_reported():
    table = generate_dci_markdown_table([_summary("bias-in-bios", 2)])

    assert "| bias-in-bios | gemma | SSAE | 2 | 0.7500 ± 0.0500 |" in table
