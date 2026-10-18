"""Evaluation utilities for environmental binary-event forecasts.

Uses calendar-ordered out-of-sample evaluation and cluster bootstrap uncertainty.
Not intended to establish causal effects or publication-worthiness.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss


def evaluate(y_true, probabilities):
    y = np.asarray(y_true, dtype=int)
    p = np.asarray(probabilities, dtype=float)
    if y.ndim != 1 or y.shape != p.shape or len(y) == 0:
        raise ValueError('y_true and probabilities must be nonempty equally sized 1-D arrays')
    if not np.isin(y, [0, 1]).all() or not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError('binary truth and finite probabilities in [0,1] required')
    return {
        'n': int(len(y)), 'event_rate': float(y.mean()),
        'auc': float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else None,
        'average_precision': float(average_precision_score(y, p)) if y.sum() > 0 else None,
        'brier_score': float(brier_score_loss(y, p)),
    }


def temporal_holdout(frame, date_col, cutoff):
    """Split by explicit cutoff; missing dates are rejected, no random sampling."""
    if date_col not in frame:
        raise ValueError(f'missing column {date_col}')
    dates = pd.to_datetime(frame[date_col], errors='raise', utc=True)
    if dates.isna().any():
        raise ValueError('missing dates')
    cut = pd.to_datetime(cutoff, utc=True)
    before, after = frame.loc[dates < cut].copy(), frame.loc[dates >= cut].copy()
    if before.empty or after.empty:
        raise ValueError('both training and test periods must be populated')
    return before, after


def cluster_bootstrap_delta(frame, truth='event', baseline='baseline', candidate='candidate', cluster='site', n_boot=1000, seed=2026):
    """Cluster-resample complete sites, return candidate minus baseline metrics.

    Each replicate draws sites with replacement, preserving all within-site rows.
    Undefined metrics for single-class replicates are skipped, with their number reported.
    """
    required = [truth, baseline, candidate, cluster]
    missing = [k for k in required if k not in frame]
    if missing:
        raise ValueError(f'missing columns: {missing}')
    if frame[required].isna().any().any():
        raise ValueError('missing input values')
    if n_boot < 1:
        raise ValueError('n_boot must be positive')
    clusters = list(frame.groupby(cluster, sort=False, dropna=False))
    if len(clusters) < 2:
        raise ValueError('at least two independent clusters are required')
    groups = [v for _, v in clusters]
    rng = np.random.default_rng(seed)
    deltas = {'auc': [], 'average_precision': [], 'brier_score': []}
    for _ in range(n_boot):
        picks = rng.integers(0, len(groups), len(groups))
        sample = pd.concat([groups[i] for i in picks], ignore_index=True)
        a = evaluate(sample[truth], sample[baseline])
        b = evaluate(sample[truth], sample[candidate])
        for metric in deltas:
            if a[metric] is not None and b[metric] is not None:
                deltas[metric].append(b[metric] - a[metric])
    full_a = evaluate(frame[truth], frame[baseline])
    full_b = evaluate(frame[truth], frame[candidate])
    output = {'clusters':len(groups), 'rows':len(frame), 'bootstrap_replicates':n_boot, 'metrics':{}}
    for metric, values in deltas.items():
        estimate = None if full_a[metric] is None or full_b[metric] is None else full_b[metric] - full_a[metric]
        output['metrics'][metric] = {'delta':estimate,
           'ci95':list(map(float,np.quantile(values,[0.025,0.975]))) if values else None,
           'valid_replicates':len(values)}
    return output


def main():
    p = argparse.ArgumentParser(description='Compare environmental binary-event forecasts (candidate minus baseline)')
    p.add_argument('csv', help='CSV with event, baseline, candidate, site columns')
    p.add_argument('--cutoff', help='Optional evaluation start date; requires date column')
    p.add_argument('--date-col', default='date')
    p.add_argument('--n-boot', type=int, default=1000)
    p.add_argument('--seed', type=int, default=2026)
    p.add_argument('--output', default='evaluation.json')
    args = p.parse_args()
    df = pd.read_csv(args.csv)
    if args.cutoff:
        _, df = temporal_holdout(df, args.date_col, args.cutoff)
    result = cluster_bootstrap_delta(df, n_boot=args.n_boot, seed=args.seed)
    result['baseline'] = evaluate(df['event'], df['baseline'])
    result['candidate'] = evaluate(df['event'], df['candidate'])
    result['interpretation'] = 'Positive delta AUC/AP or negative delta Brier favors candidate; exploratory uncertainty only.'
    Path(args.output).write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))

if __name__ == '__main__':
    main()
