# Statistical Tests

`two_sided_wilcoxon_summary.csv` is recomputed from the public per-page result
tables using `code/analysis/recompute_two_sided_wilcoxon_tables.py`.

Use this file for manuscript-facing best-of-K paired Wilcoxon p-values. Its
one-output selector rows are recomputed from the strict full-feature candidate
scores. The canonical policy-matched selector comparisons are separately reported
under `results/oracle_free_selector/strict/comparisons/`.

The CSV reports both `p_two_sided` and `p_holm`. For best-of-K tests, each
`holm_family` is one policy contrast within one evaluation block (12 tests for
each Vietnamese block and 8 for SROIE). In the strict static-selector table,
each comparator is one family across eight setting-recognizer combinations. In
the strict policy table, the primary 16 rows and rank-only 16 rows are separate
Holm families.
`significant_holm_0_05` is the manuscript-facing significance indicator.
