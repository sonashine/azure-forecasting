# Cloud Workload Forecasting: Solution

Forecasts the final 2 hours (24 five-minute timestamps) of two fleet-level targets built from the
curated Microsoft Azure Public Dataset V2 VM trace (10,000 VMs x 134 CPU readings):

- `active_vm_count`
- `aggregate_cpu_utilization` (mean of VM-level `avg_cpu`, in percent)

The final report is in `reports/final_report.md`.

## Headline result

Selected model: **BlendLin**, a linear-decay blend of persistence (last value) and the training mean.
It has no parameter fitted to the validation folds.

| Rolling-origin CV (training window only, 10 folds) | MAE | RMSE | RMSE vs Naive |
|---|---:|---:|---:|
| Naive (persistence) | 0.2392 | 0.2920 | 1.00 |
| HistoricAverage | 0.2037 | 0.2400 | 0.82 |
| **BlendLin (selected)** | **0.1853** | **0.2201** | **0.75** |

| Final 24-step holdout (scored once, reporting only) | MAE | RMSE |
|---|---:|---:|
| BlendLin | 0.0823 | 0.1073 |
| Naive | 0.1468 | 0.1848 |

`active_vm_count` is constant at 10,000 for every VM and timestamp (a closed cohort), so persistence is
exact and its holdout error is 0. The holdout window turned out easier than the CV folds, so the CV
numbers are the more representative estimate of typical accuracy.

## Setup

Python 3.11 or newer.

```bash
python3 -m venv .venv
source .venv/bin/activate            # Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev]" # or python -m pip install -r requirements.txt
```

## Dataset location

The data is not committed. Download `data-v1.tar.gz` from the
[`data-v1` release](https://github.com/atomityhq/forecasting-challenge/releases/tag/data-v1)
and extract it into `challenge_data/` at the repository root:

```bash
mkdir -p challenge_data
tar -xzf data-v1.tar.gz -C challenge_data
cd challenge_data && sha256sum -c SHA256SUMS && cd ..     # macOS: shasum -a 256 -c SHA256SUMS
```

Expected files: `challenge_data/vm_cpu_readings.csv.gz` and `challenge_data/vm_metadata.csv.gz`.
Only `vm_cpu_readings.csv.gz` is needed to produce the predictions.

## Generate the predictions

One command, from the repository root:

```bash
python3 src/cwfc/pipeline.py --data challenge_data --output predictions.csv
```

It builds both targets, runs rolling-origin cross-validation on the training window, fits the selected
model, writes `predictions.csv`, and prints the CV table and a one-time holdout report.
Add `--skip-holdout-score` to suppress the holdout report.

Output schema (24 rows, chronological, no index column):

```text
timestamp,active_vm_count,aggregate_cpu_utilization
```

The script is not tied to the supplied files: the holdout is always the last 24 timestamps, the training
window is everything before it, and the number of CV folds adapts to the amount of history.

## Tests

```bash
python -m pytest -q
```

## Repository layout

```text
src/cwfc/pipeline.py           Full pipeline (targets, split, CV, forecast, predictions.csv)
notebooks/                     EDA and model-comparison notebooks
reports/final_report.md        final report
tests/                         unit tests
predictions.csv                submission file
```

## Main modeling decisions

1. **Targets first, split second.** Both targets are built from the full panel, then split
   chronologically: the first 110 timestamps train, the last 24 are the holdout. Each timestamp's value
   uses only that timestamp's readings, so building the series cannot leak.
2. **Native five-minute cadence.** Only 134 observations exist, so no coarser aggregation is used, and
   the mapping back to required timestamps is the identity.
3. **Validation.** Expanding-window rolling-origin CV inside the 110 training points: horizon 24
   (same as the holdout), up to 10 folds, forecast origins 4 steps apart, last fold ending exactly at
   step 110. No random shuffling anywhere. Every model runs through the same folds.
4. **No features.** The selected model uses only the history of the target. Metadata CPU statistics
   (`max_cpu`, `avg_cpu`, `p95_max_cpu`) are lifetime statistics (up to 30 days) that include the
   holdout period, so they are excluded as leakage. Calendar features are excluded because timestamps
   are relative offsets and the window is under 12 hours.
5. **Model.** The ACF/PACF show short memory (lag-1 autocorrelation about 0.66, near zero by lag 16-24)
   and no usable seasonality, and the per-horizon error table shows persistence is best for the first
   steps while the long-run mean is best later. BlendLin encodes that directly:
   `forecast_h = w_h * last_value + (1 - w_h) * training_mean`, with `w_h = 1 - (h - 1) / 23`.
6. **Candidates evaluated.** Naive, moving average, seasonal naive (12 steps), HistoricAverage, a 50/50
   blend, AR(3) (CV RMSE 0.2306), exponentially decaying blends tuned on the folds, and an LSTM (CV RMSE
   0.4014, worse than all baselines because there are only tens of training windows per fold).
7. **Selection rule.** Chosen before looking at the holdout: lowest worst-half CV RMSE among candidates
   with no parameter tuned on the folds. A tuned blend looked best pooled (RMSE 0.2154), but choosing its
   decay parameter on one half of the folds and testing on the other half did not hold up (the chosen
   value flipped between halves and the advantage vanished), so it was not selected.
8. **Holdout discipline.** The holdout is scored once, after `predictions.csv` is written, and is never
   used for fitting, feature selection, tuning or model selection.

## Assumptions and limitations

- `aggregate_cpu_utilization` is an unweighted mean over VMs and is clipped to [0, 100]. Missing readings
  are skipped (mean over observed VMs); none exist in the release. A 24-core VM counts the same as a
  2-core VM.
- The cohort is closed: every VM exists for the whole window, so `active_vm_count` is constant and says
  nothing about arrivals or departures.
- If a timestamp has no readings at all, it is dropped and the remaining timestamps are treated as
  consecutive steps. This does not occur in the release.
- The CV folds overlap heavily (24-step targets, origins 4 apart), so fold results are correlated and
  small differences between top candidates are not statistically meaningful.
- Eleven hours of data cannot establish daily or weekly seasonality.