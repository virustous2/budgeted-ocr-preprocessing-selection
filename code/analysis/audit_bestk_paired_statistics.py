"""Independent arithmetic replay from original saved per-page CER, no fits."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    meta = json.loads((args.output_dir / "run_metadata.json").read_text())
    for paths in meta['sources'].values():
        for name, digest in paths.items():
            assert hashlib.sha256(Path(name).read_bytes()).hexdigest() == digest
    for name, digest in meta['outputs'].items():
        assert hashlib.sha256((args.output_dir / name).read_bytes()).hexdigest() == digest
    config = json.loads(Path(meta['config']['path']).read_text())
    root = Path(__file__).resolve().parents[1]
    stats = pd.read_csv(args.output_dir / 'statistics.csv')
    assert len(stats) == 160 and not stats.duplicated(['block','engine','k','method_a','method_b']).any()
    for row in stats.itertuples():
        paths = config[row.block]
        s = pd.read_csv(root / paths['surrogate'] / f'per_page_results_{row.engine}.csv').set_index('image_name').sort_index()
        nn = pd.read_csv(root / paths['nn'] / 'nn_per_image.csv')
        n = nn[nn.engine == row.engine].set_index('image_name').sort_index()
        assert s.index.equals(n.index)
        values = {'Surrogate':s[f'surr_k{row.k}'].to_numpy(), 'NN':n[f'nn_k{row.k}_cer'].to_numpy(),
                  'Random':s[f'rand_k{row.k}'].to_numpy(), 'FixedTopK':s[f'fixed_topk_k{row.k}'].to_numpy()}
        a,b = values[row.method_a],values[row.method_b];diff=a-b
        p = 1. if np.all(diff == 0) else wilcoxon(a,b,alternative='two-sided',zero_method='wilcox').pvalue
        np.testing.assert_allclose(p,row.p_two_sided,rtol=1e-12,atol=1e-15)
        np.testing.assert_allclose(100*diff.mean(),row.difference_pp,rtol=0,atol=1e-12)
        rng=np.random.default_rng(row.bootstrap_seed)
        ci=100*np.quantile(diff[rng.integers(len(diff),size=(10000,len(diff)))].mean(axis=1),[.025,.975])
        np.testing.assert_allclose(ci,[row.ci_low_pp,row.ci_high_pp],rtol=0,atol=1e-12)
    for (block,a,b),g in stats.groupby(['block','method_a','method_b'],sort=False):
        size=8 if block=='sroie' else 12
        assert len(g)==size and g.holm_family_size.eq(size).all()
        p=g.p_two_sided.to_numpy();order=np.argsort(p);adjust=np.empty(size)
        adjust[order]=np.minimum(1,np.maximum.accumulate(p[order]*(size-np.arange(size))))
        np.testing.assert_allclose(adjust,g.p_holm,rtol=1e-12,atol=1e-15)
        sig=g.p_holm<.05
        print(block,a,'vs',b,'significant lower/higher',int((sig&(g.difference_pp<0)).sum()),int((sig&(g.difference_pp>0)).sum()))
    print('PASS: 160 paired means/p-values/CIs; 15 complete Holm families; original-source/output hashes')


if __name__ == '__main__': main()
