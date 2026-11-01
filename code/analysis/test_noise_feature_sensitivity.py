"""Cheap tests only: synthetic data and mocked estimators, no production fit/OCR."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import warnings

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.model_selection import GroupKFold

import run_noise_feature_sensitivity as a


def fixture():
    rng = np.random.default_rng(402)
    features = a.surrogate.FEATURE_COLS_BASE + a.surrogate.FEATURE_COLS_NEW
    rows = []
    for page in range(10):
        vals = dict(zip(features, rng.uniform(.1, 2., len(features))))
        vals["orig_noise"] = np.sqrt(vals["orig_blur"])
        vals["orig_aspect_ratio"] = float(page+1)  # Spy-only group marker.
        for candidate in range(50):
            rows.append(dict(image_name=f"p{page:02d}", **vals,
                gaussian_ksize=candidate, bilateral_d=candidate % 5,
                clahe_clip=.2, gamma=1., sharpen=.1, cer_vietocr=.05+candidate/500.))
    df = pd.DataFrame(rows)
    groups = df.image_name.to_numpy(); mapping = {}
    for fold, (_, te) in enumerate(GroupKFold(5).split(df, groups=groups), 1):
        mapping.update({name:fold for name in set(groups[te])})
    folds = pd.DataFrame(dict(image_name=list(mapping), fold=list(mapping.values())))
    nopre = {"vietocr": {name:.17 for name in mapping}}
    return df, folds, nopre


class SpyRegressor(RegressorMixin, BaseEstimator):
    fits = []
    def __init__(self, group_index=10): self.group_index=group_index
    def fit(self, X, y):
        type(self).fits.append(set(X[:,self.group_index]))
        self.mean_ = np.mean(y); return self
    def predict(self, X): return np.full(len(X), self.mean_)


class SensitivityTests(unittest.TestCase):
    def test_only_standalone_column_removed_interaction_kept(self):
        df, _, _ = fixture()
        full, sc, nc = a.feature_lists(df, "full")
        drop, sd, nd = a.feature_lists(df, "drop_orig_noise")
        self.assertEqual((len(sc),len(sd),len(nc),len(nd)), (22,21,12,11))
        self.assertEqual(sd,[c for c in sc if c != "orig_noise"])
        self.assertEqual(nd,[c for c in nc if c != "orig_noise"])
        self.assertIn("inter_noise_bilateral_d", sd)
        pd.testing.assert_series_equal(full.inter_noise_bilateral_d,drop.inter_noise_bilateral_d)
        pd.testing.assert_frame_equal(df,fixture()[0])

    def test_valid_cohort_and_frozen_folds(self):
        df, f, n = fixture(); a.validate_data(df,n,f,["vietocr"],10)

    def test_invalid_schema_nonfinite_negative_and_feature_variation(self):
        df, f, n = fixture()
        with self.assertRaises(ValueError): a.feature_lists(df.drop(columns="orig_noise"),"full")
        for col,value in [("orig_blur",np.nan),("cer_vietocr",-.1),("orig_contrast",100.)]:
            changed=df.copy(); changed.loc[0,col]=value
            with self.assertRaises(ValueError): a.validate_data(changed,n,f,["vietocr"],10)

    def test_duplicate_candidate_and_nopre_cohort_rejected(self):
        df,f,n=fixture(); changed=pd.concat([df.iloc[:-1],df.iloc[[0]]],ignore_index=True)
        with self.assertRaises(ValueError): a.validate_data(changed,n,f,["vietocr"],10)
        n["vietocr"].pop(next(iter(n["vietocr"])))
        with self.assertRaises(ValueError): a.validate_data(df,n,f,["vietocr"],10)

    def test_frozen_fold_change_rejected(self):
        df,f,n=fixture(); f.loc[0,"fold"]=f.loc[0,"fold"]%5+1
        with self.assertRaises(ValueError): a.validate_data(df,n,f,["vietocr"],10)

    def test_distance_ties_detected(self):
        df,_,_=fixture()
        cols=a.surrogate.FEATURE_COLS_BASE+a.surrogate.FEATURE_COLS_NEW
        self.assertEqual(a.distance_ties(df,cols),0)
        for c in cols: df[c]=1.
        self.assertGreater(a.distance_ties(df,cols),0)

    def test_actual_nested_routine_excludes_outer_pages_mock_models(self):
        df,_,n=fixture(); df,cols,_=a.feature_lists(df,"full")
        SpyRegressor.fits=[]
        with patch.object(a.surrogate,"get_models",return_value={"Spy":SpyRegressor()}), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result=a.surrogate.train_one_engine_kfold(df,"cer_vietocr",cols,n["vietocr"],5)
        # Five inner fits + one outer refit per outer fold, then six all-data fits.
        self.assertEqual(len(SpyRegressor.fits),36)
        for i,(tr,te) in enumerate(GroupKFold(5).split(df,groups=df.image_name)):
            train=set(df.iloc[tr].orig_aspect_ratio); test=set(df.iloc[te].orig_aspect_ratio)
            for fitted in SpyRegressor.fits[i*6:(i+1)*6]:
                self.assertTrue(fitted <= train); self.assertFalse(fitted & test)
            self.assertEqual(SpyRegressor.fits[i*6+5],train)
        self.assertEqual(len(result["per_page_results"]),10)

    def test_manifest_guard_and_completed_hash_guard(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory); a.write_json(root/"manifest.json",{"inputs":{},"code":{},"x":1})
            a.guard(root,{"inputs":{},"code":{},"x":1})
            with self.assertRaises(ValueError): a.guard(root,{"x":2})
            job=root/"job"; attempt=job/"attempts/a"; attempt.mkdir(parents=True)
            (attempt/"v.csv").write_text("a\n1\n")
            a.write_json(job/"complete.json",dict(attempt="attempts/a",files={"attempts/a/v.csv":a.sha(attempt/"v.csv")}))
            self.assertEqual(a.completed(job),attempt)
            (attempt/"v.csv").write_text("a\n2\n")
            with self.assertRaises(ValueError): a.completed(job)

    def test_one_writer_and_completed_jobs_not_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            with a.writer_lock(root):
                with self.assertRaises(FileExistsError):
                    with a.writer_lock(root): pass
            self.assertFalse((root/".writer.lock").exists())

    def test_interrupted_attempt_retained_and_restart_gets_new_attempt(self):
        df,f,n=fixture()
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory); a.write_json(root/"manifest.json",{"inputs":{},"code":{}})
            with patch.object(a,"load_block",return_value=(df,f,n)), patch.object(a.surrogate,"train_one_engine_kfold",side_effect=KeyboardInterrupt):
                with self.assertRaises(KeyboardInterrupt): a.run_job({},root,"raw","full","vietocr")
            job=root/"jobs/full/raw/vietocr"
            self.assertIsNone(a.completed(job))
            first=list((job/"attempts").iterdir()); self.assertEqual(len(first),1)
            self.assertEqual(json.loads((first[0]/"failed.json").read_text())["exception_type"],"KeyboardInterrupt")
            with patch.object(a,"load_block",return_value=(df,f,n)), patch.object(a.surrogate,"train_one_engine_kfold",side_effect=RuntimeError("test")):
                with self.assertRaises(RuntimeError): a.run_job({},root,"raw","full","vietocr")
            self.assertEqual(len(list((job/"attempts").iterdir())),2)
            self.assertTrue((first[0]/"failed.json").exists())

    def test_success_job_reuse_and_no_model_pickle(self):
        df,f,n=fixture(); prepared,cols,_=a.feature_lists(df,"full")
        with patch.object(a.surrogate,"get_models",return_value={"Spy":SpyRegressor()}), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result=a.surrogate.train_one_engine_kfold(prepared,"cer_vietocr",cols,n["vietocr"],5)
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory); a.write_json(root/"manifest.json",{"inputs":{},"code":{}})
            with patch.object(a,"load_block",return_value=(df,f,n)), patch.object(a.surrogate,"train_one_engine_kfold",return_value=result) as fit:
                self.assertTrue(a.run_job({},root,"raw","full","vietocr"))
                self.assertFalse(a.run_job({},root,"raw","full","vietocr"))
                self.assertEqual(fit.call_count,1)
            attempt=a.completed(root/"jobs/full/raw/vietocr")
            self.assertTrue((attempt/"nn_per_image.csv").is_file())
            self.assertEqual(len(pd.read_csv(attempt/"nn_per_image.csv")),10)
            self.assertEqual(list(root.rglob("*.pkl")),[])

    def test_dependency_changed_during_run_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory); source=root/"input.csv";source.write_text("a\n1\n")
            a.write_json(root/"manifest.json",{"inputs":{str(source):a.sha(source)},"code":{}})
            a.validate_frozen_files(root)
            source.write_text("a\n2\n")
            with self.assertRaises(ValueError): a.validate_frozen_files(root)

    def test_direct_paired_statistics_families_and_scipy_p(self):
        rows=[]
        for b,engines in a.ENGINES.items():
            for e in engines:
                for k in a.KS:
                    for i in range(6):
                        for c in a.CONDITIONS:
                            offset=(i+1)/1000 if c=="drop_orig_noise" else 0
                            rows.append(dict(block=b,engine=e,k=k,image_name=f"p{i}",condition=c,
                                Surrogate=.1+offset,NN=.2-offset))
        table=a.direct_statistics(pd.DataFrame(rows))
        self.assertEqual(len(table),64)
        self.assertEqual(table.groupby(["block","policy"]).size().tolist(),[12,12,12,12,8,8])
        expected=wilcoxon(np.array([.1+(i+1)/1000 for i in range(6)]),np.full(6,.1),alternative="two-sided",zero_method="wilcox").pvalue
        self.assertEqual(table[table.policy.eq("Surrogate")].p_two_sided.iloc[0],expected)
        self.assertTrue((table[table.policy.eq("Surrogate")].difference_pp>0).all())

    def test_direction_and_significance_changes_not_only_counts(self):
        frame=pd.DataFrame([dict(block="raw",engine="vietocr",k=3,method_a="NN",method_b="Surrogate",difference_pp=-.1,p_holm=.01)])
        same=frame.copy(); self.assertFalse(a.significance_changes(frame,same).changed.any())
        same.loc[0,"difference_pp"] = .1
        self.assertTrue(a.significance_changes(frame,same).changed.all())
        same.loc[0,"p_holm"] = .2
        self.assertTrue(a.significance_changes(frame,same).changed.all())


if __name__ == "__main__": unittest.main()
