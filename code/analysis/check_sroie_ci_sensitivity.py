"""Post hoc CI diagnostics only; no OCR, fitting, or replacement of published CIs.

Run with --tables DIR --output-dir NEW_DIR. Fixed B=10000, stored seeds for
BCa comparisons, seeds 0..199 for the two near-zero casefold EasyOCR bounds.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
import scipy
from scipy.stats import bootstrap, gaussian_kde


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def excludes_zero(ci):
    return bool(ci[0] > 0 or ci[1] < 0)


def bootstrap_means(d, seed, draws=10000):
    rng = np.random.default_rng(seed)
    return np.concatenate([
        d[rng.integers(len(d), size=(min(500, draws-i), len(d)))].mean(axis=1)
        for i in range(0, draws, 500)
    ])


def run(tables, out):
    tables, out = tables.resolve(), out.resolve()
    if out == tables or tables in out.parents or out in tables.parents:
        raise ValueError('Use separate sibling input/output directories')
    if out.exists():
        raise ValueError('Use a fresh output directory; no overwrite or resume')
    hashes = {n: sha(tables/n) for n in ['paired_values.csv', 'statistics.csv']}
    paired = pd.read_csv(tables/'paired_values.csv')
    stats = pd.read_csv(tables/'statistics.csv')
    assert len(paired) == 9856 and len(stats) == 80
    assert not paired.duplicated(['metric', 'engine', 'k', 'image_name']).any()
    assert not stats.duplicated(['metric', 'engine', 'k', 'method_a', 'method_b']).any()
    assert not paired.isna().any().any() and not stats.isna().any().any()
    assert (stats.bootstrap_replicates == 10000).all()
    rows, seed_rows = [], []
    for r in stats.itertuples():
        g = paired[(paired.metric == r.metric) & (paired.engine == r.engine)
                   & (paired.k == r.k)].sort_values('image_name')
        assert len(g) == 616 and g.image_name.nunique() == 616
        d = 100*(g[r.method_a]-g[r.method_b]).to_numpy()
        assert np.isfinite(d).all()
        result = bootstrap((d,), np.mean, method='BCa', n_resamples=10000,
                           batch=500, random_state=np.random.default_rng(r.bootstrap_seed))
        dist = result.bootstrap_distribution
        pct = np.quantile(dist, [.025, .975])
        bca = np.asarray(result.confidence_interval)
        assert np.isfinite(bca).all()
        np.testing.assert_allclose(pct, [r.ci_low_pp, r.ci_high_pp], rtol=0, atol=1e-12)
        # Quantile MCSE approximation: sqrt(q*(1-q)/B)/f(quantile).
        # Scott-bandwidth Gaussian KDE estimates f; these are NOT error bounds.
        mcse = np.sqrt(.025*.975/10000)/gaussian_kde(dist)(pct)
        rows.append(dict(metric=r.metric, engine=r.engine, k=r.k,
                         method_a=r.method_a, method_b=r.method_b, seed=r.bootstrap_seed,
                         percentile_low_pp=pct[0], percentile_high_pp=pct[1],
                         bca_low_pp=bca[0], bca_high_pp=bca[1],
                         changed_zero_exclusion=excludes_zero(pct) != excludes_zero(bca),
                         mcse_low_pp=mcse[0], mcse_high_pp=mcse[1],
                         sampling_se_pp=np.std(dist, ddof=1)))
        if r.metric == 'casefold' and r.engine == 'easyocr' and r.k in (3, 5) and (r.method_a, r.method_b) == ('NN', 'Surrogate'):
            for seed in range(200):
                ci = np.quantile(bootstrap_means(d, seed), [.025, .975])
                seed_rows.append(dict(k=r.k, seed=seed, ci_low_pp=ci[0], ci_high_pp=ci[1]))
    frame, seeds = pd.DataFrame(rows), pd.DataFrame(seed_rows)
    assert len(seeds) == 400
    summary = {
        'change_id': 'CHG-20260926-14', 'scope': 'post_hoc_CI_diagnostics_only',
        'bootstrap_replicates': 10000, 'seed_set': 'integers 0..199 for K3/K5',
        'input_sha256': hashes, 'script_sha256': sha(__file__),
        'versions': {'numpy': np.__version__, 'pandas': pd.__version__, 'scipy': scipy.__version__},
        'bca_changed_zero_exclusion': int(frame.changed_zero_exclusion.sum()),
        'bca_max_endpoint_change_pp': float(np.max(np.abs(frame[['bca_low_pp', 'bca_high_pp']].to_numpy()-frame[['percentile_low_pp', 'percentile_high_pp']].to_numpy()))),
        'mcse_method': 'quantile asymptotic formula; Gaussian KDE, Scott bandwidth; approximate standard errors, not absolute bounds',
        'mcse_median_pp': float(np.median(frame[['mcse_low_pp', 'mcse_high_pp']])),
        'mcse_max_pp': float(np.max(frame[['mcse_low_pp', 'mcse_high_pp']].to_numpy())),
        'seed_checks': {str(k): {'upper_negative': int((g.ci_high_pp < 0).sum()),
                                'upper_positive': int((g.ci_high_pp > 0).sum()),
                                'upper_sd_pp': float(g.ci_high_pp.std(ddof=1))}
                        for k, g in seeds.groupby('k')},
        'limitations': 'No CI coverage validation, no model-refit uncertainty, no multiplicity-adjusted CI; finite seed checks do not establish invariance for every seed.'
    }
    buffers = {'bca_comparison.csv': frame.to_csv(index=False),
               'seed_checks.csv': seeds.to_csv(index=False)}
    summary['outputs_sha256'] = {n: hashlib.sha256(v.encode()).hexdigest() for n,v in buffers.items()}
    buffers['summary.json'] = json.dumps(summary, indent=2, sort_keys=True)+'\n'
    assert hashes == {n: sha(tables/n) for n in hashes}
    out.mkdir(parents=True)
    for name, content in buffers.items():
        (out/name).write_text(content, encoding='utf-8')
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tables', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    run(args.tables, args.output_dir)
