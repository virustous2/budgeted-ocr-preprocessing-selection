#!/usr/bin/env python3
"""Compute policy-agnostic FixedTopK-to-oracle headroom controls from public outputs.

The original diagnostic correlated two quantities that share FixedTopK:

    headroom_i        = FixedTopK_i - sampled_oracle_i
    policy_advantage_i = FixedTopK_i - Policy_i

Because policy_advantage <= headroom, this relationship is descriptive and is
not specific evidence for a learned policy. This script makes that limitation
explicit by computing the same statistics for Surrogate, Random K, and NN-K.
It also computes non-shared policy differences such as Random K - Surrogate,
whose outcome does not contain FixedTopK.

All inputs are cached held-out per-page results. No OCR, surrogate training, or
NN-K evaluation is rerun.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
from scipy.stats import spearmanr


DEFAULT_K_VALUES = (3, 5, 10, 15)
POLICY_ORDER = ("Surrogate", "Random", "NN-K")


def as_float(value: str) -> float:
    return float(value) if value not in {"", None} else float("nan")


def read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"No rows to write: {path}")
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def split_quartiles(values: Sequence[float]) -> List[np.ndarray]:
    order = np.argsort(np.asarray(values, dtype=float), kind="mergesort")
    return [np.asarray(idx, dtype=int) for idx in np.array_split(order, 4)]


def safe_spearman(x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    mask = np.isfinite(x) & np.isfinite(y)
    if int(mask.sum()) < 3:
        return float("nan"), float("nan")
    rho, p_value = spearmanr(x[mask], y[mask])
    return float(rho), float(p_value)


def holm_adjust(p_values: Sequence[float]) -> List[float]:
    """Return Holm-adjusted p-values in the original row order."""
    order = sorted(range(len(p_values)), key=lambda index: p_values[index])
    adjusted = [float("nan")] * len(p_values)
    running_max = 0.0
    for rank, index in enumerate(order):
        candidate = min(1.0, (len(p_values) - rank) * float(p_values[index]))
        running_max = max(running_max, candidate)
        adjusted[index] = running_max
    return adjusted


def add_holm_adjustments(
    rows: List[Dict[str, object]], family_key: str, family_prefix: str
) -> None:
    grouped: Dict[str, List[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        family = f"{family_prefix}:{row[family_key]}:all_32_configurations"
        row["holm_family"] = family
        grouped[family].append(index)
    for indices in grouped.values():
        adjusted = holm_adjust(
            [float(rows[index]["spearman_p_value"]) for index in indices]
        )
        for index, p_holm in zip(indices, adjusted):
            rows[index]["spearman_p_holm"] = p_holm
            rows[index]["significant_holm_0_05"] = p_holm < 0.05


def assert_close(label: str, a: np.ndarray, b: np.ndarray) -> None:
    if not np.allclose(a, b, rtol=0.0, atol=1e-12, equal_nan=True):
        max_diff = float(np.nanmax(np.abs(a - b)))
        raise ValueError(f"Cached result mismatch for {label}: max diff={max_diff}")


def default_specs(release_root: Path):
    return [
        (
            "Vietnamese",
            "raw",
            release_root / "results/per_page_bestk/raw",
            release_root / "results/nn_baseline/raw/nn_per_image.csv",
            ("vietocr", "tesseract", "easyocr"),
        ),
        (
            "Vietnamese",
            "contour",
            release_root / "results/per_page_bestk/contour",
            release_root / "results/nn_baseline/contour/nn_per_image.csv",
            ("vietocr", "tesseract", "easyocr"),
        ),
        (
            "SROIE",
            "raw",
            release_root / "results/per_page_bestk/sroie",
            release_root / "results/nn_baseline/sroie/nn_per_image.csv",
            ("tesseract", "easyocr"),
        ),
    ]


def merge_engine_rows(
    per_page_path: Path,
    nn_path: Path,
    engine: str,
) -> List[Dict[str, str]]:
    per_page = {row["image_name"]: row for row in read_csv(per_page_path)}
    nn_rows = {
        row["image_name"]: row
        for row in read_csv(nn_path)
        if row["engine"].strip().lower() == engine.lower()
    }
    if set(per_page) != set(nn_rows):
        only_base = sorted(set(per_page) - set(nn_rows))[:5]
        only_nn = sorted(set(nn_rows) - set(per_page))[:5]
        raise ValueError(
            f"Image-key mismatch for {engine}: only per-page={only_base}, only NN={only_nn}"
        )

    merged = []
    for image_name in sorted(per_page):
        row = dict(per_page[image_name])
        row.update({f"nn_cache__{key}": value for key, value in nn_rows[image_name].items()})
        merged.append(row)
    return merged


def arrays_for_k(rows: List[Dict[str, str]], k: int) -> Mapping[str, np.ndarray]:
    arrays = {
        "oracle": np.asarray([as_float(r["oracle"]) for r in rows]),
        "fixed": np.asarray([as_float(r[f"fixed_topk_k{k}"]) for r in rows]),
        "Surrogate": np.asarray([as_float(r[f"surr_k{k}"]) for r in rows]),
        "Random": np.asarray([as_float(r[f"rand_k{k}"]) for r in rows]),
        "NN-K": np.asarray([as_float(r[f"nn_cache__nn_k{k}_cer"]) for r in rows]),
    }

    # NN-K outputs retain copies of the paired baselines. Fail loudly if the
    # two authoritative caches do not describe the same held-out evaluation.
    assert_close(
        f"FixedTopK K={k}",
        arrays["fixed"],
        np.asarray([as_float(r[f"nn_cache__fixed_topk_k{k}_cer"]) for r in rows]),
    )
    assert_close(
        f"Surrogate K={k}",
        arrays["Surrogate"],
        np.asarray([as_float(r[f"nn_cache__surr_k{k}_cer"]) for r in rows]),
    )
    assert_close(
        f"Random K={k}",
        arrays["Random"],
        np.asarray([as_float(r[f"nn_cache__random_k{k}_cer"]) for r in rows]),
    )
    return arrays


def policy_rows(
    dataset: str,
    setting: str,
    engine: str,
    k: int,
    arrays: Mapping[str, np.ndarray],
) -> Tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    headroom = arrays["fixed"] - arrays["oracle"]
    quartile_indices = split_quartiles(headroom)
    summaries: List[Dict[str, object]] = []
    quartiles: List[Dict[str, object]] = []

    for policy in POLICY_ORDER:
        policy_cer = arrays[policy]
        advantage = arrays["fixed"] - policy_cer
        rho, p_value = safe_spearman(headroom, advantage)
        q_advantages = [float(100 * np.mean(advantage[idx])) for idx in quartile_indices]
        summaries.append(
            {
                "dataset": dataset,
                "setting": setting,
                "engine": engine,
                "k": k,
                "policy": policy,
                "n_pages": int(len(headroom)),
                "mean_headroom_pp": float(100 * np.mean(headroom)),
                "median_headroom_pp": float(100 * np.median(headroom)),
                "mean_policy_cer_pct": float(100 * np.mean(policy_cer)),
                "mean_policy_advantage_pp": float(100 * np.mean(advantage)),
                "spearman_headroom_vs_advantage": rho,
                "spearman_p_value": p_value,
                "q1_advantage_pp": q_advantages[0],
                "q4_advantage_pp": q_advantages[3],
                "q4_minus_q1_advantage_pp": q_advantages[3] - q_advantages[0],
            }
        )

        for q_idx, idx in enumerate(quartile_indices, start=1):
            quartiles.append(
                {
                    "dataset": dataset,
                    "setting": setting,
                    "engine": engine,
                    "k": k,
                    "policy": policy,
                    "quartile": f"Q{q_idx}",
                    "n_pages": int(len(idx)),
                    "mean_headroom_pp": float(100 * np.mean(headroom[idx])),
                    "median_headroom_pp": float(100 * np.median(headroom[idx])),
                    "mean_policy_advantage_pp": float(100 * np.mean(advantage[idx])),
                    "median_policy_advantage_pp": float(100 * np.median(advantage[idx])),
                }
            )
    return summaries, quartiles


def nonshared_rows(
    dataset: str,
    setting: str,
    engine: str,
    k: int,
    arrays: Mapping[str, np.ndarray],
) -> List[Dict[str, object]]:
    headroom = arrays["fixed"] - arrays["oracle"]
    quartile_indices = split_quartiles(headroom)
    comparisons = (
        ("Surrogate_vs_Random", arrays["Random"] - arrays["Surrogate"]),
        ("NN-K_vs_Random", arrays["Random"] - arrays["NN-K"]),
        ("Surrogate_vs_NN-K", arrays["NN-K"] - arrays["Surrogate"]),
    )
    rows: List[Dict[str, object]] = []
    for comparison, gain in comparisons:
        rho, p_value = safe_spearman(headroom, gain)
        q_gains = [float(100 * np.mean(gain[idx])) for idx in quartile_indices]
        rows.append(
            {
                "dataset": dataset,
                "setting": setting,
                "engine": engine,
                "k": k,
                "comparison": comparison,
                "n_pages": int(len(headroom)),
                "mean_gain_pp": float(100 * np.mean(gain)),
                "spearman_headroom_vs_gain": rho,
                "spearman_p_value": p_value,
                "q1_gain_pp": q_gains[0],
                "q4_gain_pp": q_gains[3],
                "q4_minus_q1_gain_pp": q_gains[3] - q_gains[0],
            }
        )
    return rows


def collect(release_root: Path, k_values: Iterable[int]):
    summary_rows: List[Dict[str, object]] = []
    quartile_rows: List[Dict[str, object]] = []
    nonshared: List[Dict[str, object]] = []

    for dataset, setting, folder, nn_path, engines in default_specs(release_root):
        for engine in engines:
            per_page_path = folder / f"per_page_results_{engine}.csv"
            if not per_page_path.exists():
                raise FileNotFoundError(per_page_path)
            if not nn_path.exists():
                raise FileNotFoundError(nn_path)
            rows = merge_engine_rows(per_page_path, nn_path, engine)
            for k in k_values:
                arrays = arrays_for_k(rows, int(k))
                summaries, quartiles = policy_rows(
                    dataset, setting, engine, int(k), arrays
                )
                summary_rows.extend(summaries)
                quartile_rows.extend(quartiles)
                nonshared.extend(
                    nonshared_rows(dataset, setting, engine, int(k), arrays)
                )
    return summary_rows, quartile_rows, nonshared


def count_positive_significant(
    rows: List[Dict[str, object]], rho_key: str, p_key: str
) -> int:
    return sum(float(row[rho_key]) > 0 and float(row[p_key]) < 0.05 for row in rows)


def write_report(
    path: Path,
    summaries: List[Dict[str, object]],
    nonshared: List[Dict[str, object]],
) -> Dict[str, object]:
    by_policy = {
        policy: [row for row in summaries if row["policy"] == policy]
        for policy in POLICY_ORDER
    }
    by_comparison = {
        comparison: [row for row in nonshared if row["comparison"] == comparison]
        for comparison in ("Surrogate_vs_Random", "NN-K_vs_Random", "Surrogate_vs_NN-K")
    }

    paired_rho = []
    for key in {
        (row["dataset"], row["setting"], row["engine"], row["k"])
        for row in summaries
    }:
        values = {
            row["policy"]: float(row["spearman_headroom_vs_advantage"])
            for row in summaries
            if (row["dataset"], row["setting"], row["engine"], row["k"]) == key
        }
        paired_rho.append(values)

    headline = {
        "n_configurations": len(by_policy["Surrogate"]),
        "positive_significant_coupled": {
            policy: count_positive_significant(
                rows, "spearman_headroom_vs_advantage", "spearman_p_value"
            )
            for policy, rows in by_policy.items()
        },
        "positive_significant_coupled_holm": {
            policy: count_positive_significant(
                rows, "spearman_headroom_vs_advantage", "spearman_p_holm"
            )
            for policy, rows in by_policy.items()
        },
        "random_rho_larger_than_surrogate": sum(
            item["Random"] > item["Surrogate"] for item in paired_rho
        ),
        "positive_significant_nonshared": {
            comparison: count_positive_significant(
                rows, "spearman_headroom_vs_gain", "spearman_p_value"
            )
            for comparison, rows in by_comparison.items()
        },
        "positive_significant_nonshared_holm": {
            comparison: count_positive_significant(
                rows, "spearman_headroom_vs_gain", "spearman_p_holm"
            )
            for comparison, rows in by_comparison.items()
        },
    }

    lines = [
        "Policy-agnostic FixedTopK-to-oracle headroom controls",
        "====================================================",
        "",
        "Coupled statistic: headroom = FixedTopK - Oracle; policy advantage = FixedTopK - Policy.",
        "A positive correlation is descriptive and is not learned-specific evidence.",
        "",
        json.dumps(headline, indent=2, sort_keys=True),
        "",
        "Non-shared outcomes do not contain FixedTopK:",
        "  Surrogate_vs_Random = Random - Surrogate",
        "  NN-K_vs_Random      = Random - NN-K",
        "  Surrogate_vs_NN-K   = NN-K - Surrogate",
        "",
    ]
    for comparison, rows in by_comparison.items():
        rhos = [float(row["spearman_headroom_vs_gain"]) for row in rows]
        lines.append(
            f"{comparison}: rho range [{min(rhos):.3f}, {max(rhos):.3f}], "
            f"positive significant raw/Holm="
            f"{headline['positive_significant_nonshared'][comparison]}/"
            f"{headline['positive_significant_nonshared_holm'][comparison]}/{len(rows)}"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return headline


def main() -> None:
    default_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser()
    parser.add_argument("--release-root", type=Path, default=default_root)
    parser.add_argument("--k-values", nargs="+", type=int, default=list(DEFAULT_K_VALUES))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
    )
    args = parser.parse_args()

    release_root = args.release_root.resolve()
    summaries, quartiles, nonshared = collect(release_root, args.k_values)
    add_holm_adjustments(summaries, "policy", "coupled_headroom")
    add_holm_adjustments(nonshared, "comparison", "nonshared_headroom")
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else release_root / "results/headroom_controls"
    )
    write_csv(output_dir / "headroom_policy_summary_allK.csv", summaries)
    write_csv(output_dir / "headroom_policy_quartiles_allK.csv", quartiles)
    write_csv(output_dir / "headroom_nonshared_controls_allK.csv", nonshared)
    headline = write_report(
        output_dir / "headroom_policy_controls_report.txt", summaries, nonshared
    )
    (output_dir / "headroom_policy_controls_headline.json").write_text(
        json.dumps(headline, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"Wrote policy controls to {output_dir}")
    print(json.dumps(headline, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
