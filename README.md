# Environmental Forecast Evaluation Toolkit (EFT)

**Version 0.1.0 — prototype, not yet an established scientific software contribution.**

Small Python research utility to compare **probabilistic predictions of binary environmental events** (e.g., high-pollution days). It reports AUC, average precision, Brier score and a *site-cluster bootstrap* confidence interval for the difference between a candidate forecast and a baseline. Includes an explicit temporal holdout helper to reduce accidental random train–test mixing.

## Quick start

```bash
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
python eft.py examples/synthetic_forecasts.csv --cutoff 2024-01-01 --n-boot 500 --output evaluation.json
```

Input CSV fields: `date`, `site`, `event` (0/1), `baseline` and `candidate` (probabilities in [0, 1]). `--cutoff` discards rows before the cutoff; it **does not train any model**. The predictions must have been generated using a valid out-of-sample protocol upstream. The sample data are **synthetic** and do not constitute empirical findings.

## Interpretation

Report `delta = candidate − baseline`. Higher AUC and average precision are better; lower Brier scores are better. Bootstrap intervals use percentile quantiles over cluster-resampled sites, keeping within-site observations together. This is an exploratory interval, not a robust hypothesis test; when sites are spatially dependent, site resampling may understate uncertainty. Also investigate event prevalence, calibration, time dependence, measurement quality, threshold choice, spatial generalization and temporal leakage. Undefined AUC in single-class bootstrap replicates is explicitly counted.

## Reproducibility & attribution

No external data are included. Synthetic demonstration data are generated deterministically and are not research evidence. This tool reimplements standard, established metrics using scikit-learn, and **does not claim a novel statistical method**. Review code, tests and fit for intended use before public release. Do not list this as a peer-reviewed publication.

## Release checklist

1. Review methods and software behavior independently; run tests in your environment.
2. Choose and confirm attribution, license, code provenance and any collaborators' rights.
3. Create a public GitHub repository and commit the files.
4. Link GitHub to Zenodo; enable the repository in Zenodo and make a versioned GitHub release (`v0.1.0`). Zenodo can then archive it and issue a DOI.
5. Add the issued DOI to the README, ORCID and CV **under Research Software**, not Journal Publications.

This is a small prototype; a DOI by itself does not establish significance or scholarly adoption.
