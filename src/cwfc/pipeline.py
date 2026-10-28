import argparse
import numpy as np
import pandas as pd
from statsforecast import StatsForecast
from statsforecast.models import Naive, HistoricAverage

H = 24                  # holdout horizon: final 24 timestamps (2 hours at 5-minute cadence)
CV_STEP = 4             # spacing between rolling-origin CV forecast origins
CV_MAX_WINDOWS = 10     # maximum 10 folds
MIN_TRAIN = 10          # smallest training length allowed in CV fold
COUNT_COL, CPU_COL = "active_vm_count", "aggregate_cpu_utilization"
 
 
def load_readings(data_dir):
    return pd.read_csv(f"{data_dir}/vm_cpu_readings.csv.gz", usecols=["timestamp", "vm_id", "avg_cpu"])
 
 
def build_targets(df):
    """Both targets from the full panel, one row per timestamp.
 
    active_vm_count: distinct VMs with a reading at the timestamp (each VM counted once).
    aggregate_cpu_utilization: mean of avg_cpu over the VMs that have a reading at that timestamp
    (missing readings are skipped, not imputed), clipped to [0, 100].
    """
    df = df.dropna(subset=["avg_cpu"]).drop_duplicates(["vm_id", "timestamp"])
    g = df.groupby("timestamp")
    return pd.DataFrame({
        COUNT_COL: g["vm_id"].nunique(),
        CPU_COL: g["avg_cpu"].mean().clip(0, 100),
    }).sort_index()
 
 
def split_train_holdout(targets):
    """Last 24 timestamps = holdout, everything before = training window (110 in the release)."""
    if len(targets) <= H:
        raise ValueError(f"need more than {H} timestamps, found {len(targets)}")
    return targets.iloc[:-H], targets.iloc[-H:]
 
 
def to_statsforecast_frame(series, name="series"):
    return pd.DataFrame({"unique_id": name, "ds": np.arange(len(series)), "y": series.values})
 
 
def blend_weights(n_steps):
    """Weight on Naive for steps 1..n_steps: 1 at the first step, 0 at the last."""
    return 1 - np.arange(n_steps) / (n_steps - 1)
 
 
def n_cv_windows(n_train):
    """Largest number of folds (up to 10) whose first fold still has MIN_TRAIN points."""
    last_origin = n_train - H                      # training length of the final fold
    if last_origin < MIN_TRAIN:
        return 0
    return min(CV_MAX_WINDOWS, (last_origin - MIN_TRAIN) // CV_STEP + 1)
 
 
def run_cv(sf, sf_df, n_windows):
    """Rolling-origin CV on the training window only, plus the BlendLin column."""
    cv = sf.cross_validation(df=sf_df, h=H, step_size=CV_STEP, n_windows=n_windows)
    cv["step"] = cv.groupby("cutoff").cumcount() + 1
    w = 1 - (cv["step"] - 1) / (H - 1)
    cv["BlendLin"] = w * cv["Naive"] + (1 - w) * cv["HistoricAverage"]
    return cv
 
 
def summarize_cv(cv, models):
    rows = []
    for m in models:
        e = cv["y"] - cv[m]
        rows.append({"model": m, "MAE": e.abs().mean(), "RMSE": (e ** 2).mean() ** 0.5})
    table = pd.DataFrame(rows).set_index("model")
    table["RMSE_vs_Naive"] = table["RMSE"] / table.loc["Naive", "RMSE"]
    return table
 
 
def cv_count(train_count, n_windows):
    """Persistence on active_vm_count under the same rolling-origin folds as the CPU target."""
    sf_c = StatsForecast(models=[Naive()], freq=1)
    cv = sf_c.cross_validation(df=to_statsforecast_frame(train_count, "active"), h=H,
                               step_size=CV_STEP, n_windows=n_windows)
    e = cv["y"] - cv["Naive"]
    return e.abs().mean(), (e ** 2).mean() ** 0.5
 
 
def forecast_cpu(sf, sf_df):
    """Fit on all training points and forecast H steps from one origin."""
    fc = sf.forecast(df=sf_df, h=H)
    w = blend_weights(H)
    blend = w * fc["Naive"].values + (1 - w) * fc["HistoricAverage"].values
    return np.clip(blend, 0, 100)
 
 
def forecast_count(train_count):
    """Persistence of the last training count (exact when the cohort is closed)."""
    return np.repeat(max(int(round(train_count.iloc[-1])), 0), H)
 
 
def write_predictions(holdout_index, count_forecast, cpu_forecast, path):
    pred = pd.DataFrame({
        "timestamp": holdout_index,
        COUNT_COL: count_forecast,
        CPU_COL: cpu_forecast,
    })
    assert len(pred) == H, "must have exactly 24 rows"
    assert pred["timestamp"].is_monotonic_increasing and not pred["timestamp"].duplicated().any()
    assert np.isfinite(pred[[COUNT_COL, CPU_COL]].to_numpy(dtype=float)).all(), "non-finite prediction"
    assert (pred[COUNT_COL] >= 0).all() and pred[CPU_COL].between(0, 100).all()
    pred.to_csv(path, index=False)
    return pred
 
 
def score_holdout(hold, train, cpu_forecast, count_forecast):
    """Reporting only. Called once after predictions.csv is written; never used to change the model."""
    naive_forecast = np.repeat(train[CPU_COL].iloc[-1], H)
    y = hold[CPU_COL].values
    rows = []
    for name, p in [("BlendLin", cpu_forecast), ("Naive", naive_forecast)]:
        e = y - p
        rows.append({"model": name, "MAE": np.abs(e).mean(), "RMSE": (e ** 2).mean() ** 0.5})
    table = pd.DataFrame(rows).set_index("model")
    table["RMSE_vs_Naive"] = table["RMSE"] / table.loc["Naive", "RMSE"]
    return table, np.abs(hold[COUNT_COL].values - count_forecast).mean()
 

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate predictions.csv")
    parser.add_argument("--data", default="challenge_data", help="folder containing vm_cpu_readings.csv.gz")
    parser.add_argument("--output", default="predictions.csv")
    parser.add_argument("--skip-holdout-score", action="store_true", help="do not print the holdout report")
    args = parser.parse_args()
 
    targets = build_targets(load_readings(args.data))
    train, hold = split_train_holdout(targets)
    sf_df = to_statsforecast_frame(train[CPU_COL])
    sf = StatsForecast(models=[Naive(), HistoricAverage()], freq=1)
 
    n_windows = n_cv_windows(len(train))
    if n_windows:
        cv = run_cv(sf, sf_df, n_windows)
        print(f"Rolling-origin CV on the training window ({n_windows} folds):")
        print(summarize_cv(cv, ["Naive", "HistoricAverage", "BlendLin"]).round(4), "\n")
        count_mae, count_rmse = cv_count(train[COUNT_COL], n_windows)
        print(f"{COUNT_COL} CV (persistence, same folds): MAE {count_mae:.4f}  RMSE {count_rmse:.4f}")
        if count_rmse == 0:
            print("  zero error at every fold, so a ratio to the baseline is undefined (0/0)\n")
        else:
            print()
    else:
        print("Training window too short for cross-validation; skipping it.\n")
 
    cpu_forecast = forecast_cpu(sf, sf_df)
    count_forecast = forecast_count(train[COUNT_COL])
    write_predictions(hold.index, count_forecast, cpu_forecast, args.output)
    print(f"Wrote {args.output} ({H} rows)")
 
    if not args.skip_holdout_score:
        table, count_mae = score_holdout(hold, train, cpu_forecast, count_forecast)
        print("\nHoldout score (reporting only):")
        print(table.round(4))
        print(f"{COUNT_COL} MAE:", count_mae)