"""Print swapdiag2's table for a split: python -m experiments.v6_rows.swapdiag2_agg val | test"""
import sys

from experiments.v6_rows import swapdiag2 as S2

t = S2.agg(sys.argv[1])
print(t.pivot_table(index=["dataset", "model", "method", "layer"], columns=["variant", "k"], values="cause").round(2).to_string())
