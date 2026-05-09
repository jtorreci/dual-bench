# Analysis pipeline

The `dual_bench.analysis` module provides non-parametric statistical tests,
performance metrics, and LaTeX table generation for comparing optimisation
results.  It requires SciPy (`pip install dual-bench[analysis]`).

## Statistical tests

### Friedman test with Nemenyi post-hoc

```python
import numpy as np
from dual_bench.analysis import friedman_test, nemenyi_posthoc

# Each array has one value per problem (block)
data = {
    "NSGA-II": np.array([0.75, 0.77, 0.91, 0.56, 0.85]),
    "GDE3":    np.array([0.75, 0.76, 0.91, 0.56, 0.83]),
    "SPEA2":   np.array([0.74, 0.77, 0.91, 0.56, 0.68]),
}

result = friedman_test(data)
print(f"p = {result['p_value']:.4f}")
print(result["rankings"])  # mean ranks per algorithm

post = nemenyi_posthoc(data)
print(f"Critical difference: {post['cd']:.3f}")
for c in post["comparisons"]:
    print(f"  {c['group_a']} vs {c['group_b']}: "
          f"diff={c['rank_diff']:.2f}, sig={c['significant']}")
```

### Kruskal-Wallis and Cliff's delta

```python
from dual_bench.analysis import kruskal_wallis, cliff_delta

kw = kruskal_wallis(data)
print(f"H = {kw['statistic']:.2f}, p = {kw['p_value']:.4f}")

delta, magnitude = cliff_delta(data["NSGA-II"], data["SPEA2"])
print(f"Cliff's delta = {delta:.3f} ({magnitude})")
```

## Performance metrics

```python
import numpy as np
from dual_bench.analysis import hypervolume_2d, igd, delta_f_star, feasibility_rate

# 2-D hypervolume (minimisation; points dominated by ref)
front = np.array([[0.013, 0.95], [0.014, 0.80], [0.015, 0.60]])
hv = hypervolume_2d(front, ref=np.array([1.1, 1.1]))
print(f"HV = {hv:.4f}")

# Inverted Generational Distance
reference_pf = np.array([[0.012, 1.0], [0.013, 0.5]])
print(f"IGD = {igd(front, reference_pf):.4f}")

# Relative distance to known optimum
print(f"df* = {delta_f_star(f_best=0.0130, f_star=0.012665):.4f}")

# Feasibility rate from d/c ratio matrix
ratios = np.array([[0.9, 0.8], [1.1, 0.7], [0.5, 0.6]])
print(f"Feasibility = {feasibility_rate(ratios):.2f}")  # 0.67
```

## LaTeX table generation

```python
from dual_bench.analysis import to_latex, summary_table
import dual_bench.references as ref

# Direct table from list of dicts
rows = ref.paper_a_results("Spring")
latex = to_latex(
    rows,
    columns=["formulation", "hv_mean", "df_mean"],
    caption="Spring: formulation comparison",
    label="tab:spring",
)
print(latex)

# Pivot-style summary table
algo_rows = ref.paper_a_results("Spring", algorithm="NSGAII")
latex = summary_table(algo_rows, metric="hv_mean",
                      rows="formulation", cols="algorithm")
print(latex)
```
