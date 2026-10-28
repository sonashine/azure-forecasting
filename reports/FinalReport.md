## Cloud Workload Forecasting: Final Report

#### Executive summary
Every one of the 10,000 VMs has exactly 134 readings, with no missing values and no duplicates, and every VM exists for the whole window. The active VM count is constant at 10,000, and persistence forecasts it exactly.
Aggregate CPU utilization is persistent over short lags (lag-1 autocorrelation about 0.66) but mean-reverting: its autocorrelation is near zero by lag 16 to 24, and there is no seasonality that 11 hours of data can support.
A simple model wins. `BlendLin` starts at the last observed value and decays linearly toward the training mean over the 24-step horizon. Under rolling-origin cross-validation it reaches RMSE 0.2201 against 0.2920 for persistence (25% lower), and it also beats the training mean (0.2400), an AR(3) (0.2306) and an LSTM (0.4014).
On the final 24-step holdout, scored once, BlendLin has MAE 0.0823 and RMSE 0.1073 against 0.1468 and 0.1848 for persistence (RMSE ratio 0.58). The holdout window was easier than the CV folds, so the CV figures are the better guide to typical accuracy.
A more heavily tuned blend looked better pooled (RMSE 0.2154), but its tuned parameter did not hold up when chosen on half the folds and tested on the other half, so it was not selected.
Operationally, provision capacity at the forecast plus a safety margin sized from cross-validation residuals (about 0.28 percentage points, roughly 3% of the fleet level), and review the policy as data accumulates.

#### EDA and preprocessing
The readings table has 1,340,000 rows (10,000 VMs x 134 timestamps). There are no null cells and no duplicate (VM, timestamp) pairs. Every VM has exactly 134 readings, and exactly 10,000 distinct VMs report at every timestamp. The 134 timestamps run from 0 to 39,900 seconds, consistent with an exact five-minute (300 s) grid [TODO: confirm with np.unique(np.diff(sorted timestamps))]. No imputation, gap filling or resampling was needed.

Lifetimes and churn: In the metadata, every VM has timestamp_vm_created = 0, and the earliest deletion is at 40,500 s, just after the last observation at 39,900 s. More than 75% of VMs have a deletion time of 2,591,400 s (about 30 days), which is probably the end of the trace, so those VMs were still running, not deleted. The release therefore contains only VMs alive for the entire window: a closed cohort with no arrivals or departures.

Targets: Both targets are built from the full panel first, and the 110/24 split is applied afterward. Each timestamp's value uses only that timestamp's readings, so constructing the series cannot leak holdout information.

active_vm_count(t): number of distinct VMs with a reading at t (each VM counted once).
aggregate_cpu_utilization(t): unweighted mean of VM-level avg_cpu over the VMs that have a reading at t, clipped to [0, 100]. If readings were missing, they would be skipped, not imputed or back-filled. This never occurs in the release.

Cadence: The native five-minute cadence is kept. With only 134 observations, coarsening would discard signal, and the mapping back to the required timestamps is the identity.

Excluded columns: The metadata columns max_cpu, avg_cpu and p95_max_cpu are lifetime statistics for VMs that live up to 30 days. Their avg_cpu correlates 0.87 with each VM's mean inside the window, so they carry information about the holdout and are not used as features.

#### Workload analysis

Level and variability of the fleet: The aggregate sits at roughly 9 to 10% CPU and moves step to step by tenths of a percentage point (one-step-ahead persistence has a mean absolute error of 0.098 points). Averaging across 10,000 VMs smooths out idiosyncratic noise, so the aggregate is far steadier than a small segment: the ten-VM >24-core line swings between roughly 4% and 15%, while the fleet moves by fractions of a point.

Temporal structure: On the 110 training points, the ACF starts near 0.66 at lag 1 and decays slowly, staying inside the 95% band beyond roughly lag 6. The PACF is significant at lags 1 to 3. A small bump at lag 12 (one hour) is not significant and is not supported by neighbouring lags. ADF tests (statistic -1.81, p = 0.70 with trend; -1.49, p = 0.54 with 3 lags) fail to reject a unit root, while KPSS (statistic 0.307, p >= 0.10) fails to reject stationarity. With 110 points these tests lack the power to separate the two, so the choice between modeling levels and differences was left to out-of-sample error, not to the tests. Eleven hours cannot establish daily or weekly seasonality, so none is claimed.

Segments: vm_category splits the fleet into 6,180 delay-insensitive, 3,091 interactive and 729 unknown VMs. Core buckets are 2 (7,534), 4 (1,908), 8 (486), 24 (62) and >24 (10); memory buckets are 2 GB (3,904), 4 (2,475), 8 (2,146), 32 (1,383), 64 (82) and >64 (10). Interactive VMs run hotter than delay-insensitive ones. The two largest-bucket lines (>24 cores, >64 GB) are very spiky, but each contains only 10 VMs, so that is small-sample noise and cannot move the fleet aggregate.

Because the cohort is closed, segment shares are constant, so the aggregate decomposes exactly into share x segment mean. Delay-insensitive VMs contribute about 5 percentage points of the roughly 9 to 10 point aggregate and rise visibly between about 15,000 and 20,000 s; interactive VMs contribute about 3.5 and are flat; the unknown category drifts down but is only 7% of VMs, so its effect on the aggregate is small. Figure: contribution plot by category.

Why this matters:

A segment matters in proportion to its share times its change, not by how dramatic its own line looks. Monitoring should follow contribution to the fleet, and the delay-insensitive workload is where the aggregate's movements originate.
Near-term demand is predictable only for a short horizon, after which forecasts converge to the long-run level, so planning should use short, frequently refreshed horizons with an uncertainty margin.
The unweighted aggregate treats a 24-core VM like a 2-core VM, although 75% of VMs have two cores. A core-weighted utilization would be a better capacity metric and is a natural extension.

#### Forecasting approach

active_vm_count: The series is constant, so it is forecast by persistence (the last training count) with no model fitted. If counts varied, persistence would remain the baseline and a count model (Poisson or negative binomial on lagged counts and churn features) would be compared under the same folds.

aggregate_cpu_utilization: The diagnostics point to a short-memory, mean-reverting series, and the per-horizon errors show where each simple forecaster is best. Persistence is most accurate at the start of the horizon (mean absolute error 0.098 at step 1 against 0.197 for the training mean) and the training mean is most accurate at the end (0.118 at step 24 against 0.231 for persistence). `BlendLin` combines them:

forecast_h = w_h * last_value + (1 - w_h) * training_mean,   w_h = 1 - (h - 1) / 23,   h = 1..24

The weight on persistence falls linearly from 1 at the first step to 0 at the last. It has no fitted parameter, uses no features, and is computed only from history available at the forecast origin.

Candidates compared under the same folds: persistence, a 24-point window average, seasonal naive (12 steps), the training mean, a 50/50 blend, AR(3) (order chosen from the PACF), exponentially decaying blends with a tuned decay, and an LSTM.

#### Validation methodology
Split: The first 110 timestamps train and the last 24 are the holdout. No shuffling is used anywhere.
Rolling-origin cross-validation inside the 110 training points, expanding window, horizon 24 (the holdout horizon), 10 folds with forecast origins 4 steps apart. The first fold trains on 50 points and the last on 86, so the final fold's target ends exactly at step 110. This mirrors production: a forecast issued at an origin, scored over the next two hours, using only earlier data.
Same folds for every model: Baselines and candidates are scored on identical folds.
Selection rule, fixed before the holdout was scored: lowest worst-half CV RMSE among candidates with no parameter tuned on the folds.
Leakage controls: Targets are built before the split; models see only data before each origin; the holdout is never used for fitting, features, tuning or selection; the holdout is scored once, after predictions.csv is written.
Caveat: The folds overlap heavily (24-step targets, origins 4 apart), so fold errors are correlated and differences between top candidates are not statistically meaningful.

##### Results versus baseline

Cross-validation (training window, 10 folds).

Model	MAE	RMSE	RMSE vs persistence
Persistence (Naive)	0.2392	0.2920	1.00
Window average (24)	0.2768	0.3077	1.05
Seasonal naive (12)	0.2750	0.3238	1.11
Training mean (HistoricAverage)	0.2037	0.2400	0.82
50/50 blend	0.2028	0.2382	0.82
AR(3)	0.1918	0.2306	0.79
BlendLin (Final Model)	0.1853	0.2201	0.75
LSTM (1 layer, 16 units)	0.3422	0.4014	1.37

Mean fold RMSE is 0.2140 for BlendLin, 0.2220 for AR(3), 0.2329 for the training mean and 0.2733 for persistence. The worst single fold is 0.496 for persistence, 0.337 for the training mean and AR(3), and 0.321 for BlendLin.

Split-half check. RMSE on the first five folds (training lengths 50 to 66) and the last five (70 to 86):

Model	Early folds	Late folds
Persistence	0.3441	0.2282
Training mean	0.2739	0.2005
AR(3)	0.2574	0.2003
50/50 blend	0.2649	0.2081
BlendLin	0.2315	0.2080

BlendLin wins clearly on the harder early folds and is about 4% behind AR(3) and the training mean on the late folds, a gap well within the noise of five overlapping folds.

Tuned blends were not selected. An exponentially decaying blend with tuning parameter 8 reached a pooled RMSE of 0.2154, but picking the parameter on the early folds (value 8) and testing on the late folds gave 0.2021 against 0.2005 for the training mean and 0.2003 for AR(3). Picking on the late folds (value 2) and testing on the early folds gave 0.2564 against 0.2739 and 0.2574. The best value flipped between halves and the advantage vanished, so the pooled figure was optimistic.

Holdout (scored once, reporting only).

Model	MAE	RMSE	RMSE vs persistence
BlendLin	0.0823	0.1073	0.58
Persistence	0.1468	0.1848	1.00

active_vm_count: zero error in every CV fold and on the holdout. A percentage metric is well defined (0%), but a relative improvement over the baseline is 0/0 and is therefore not reported. For CPU, mean utilization is near 9 to 10%, far from zero, so percentage errors would not blow up. They are not used here because MAE, RMSE and the ratio to persistence are enough.

#### Error and failure analysis
Persistence fails after a temporary rise: Its worst fold (RMSE 0.496) is the one whose origin falls near the end of the delay-insensitive bump, so the last value was temporarily high and was carried forward while the series fell back. The training mean and BlendLin both avoid this, with worst folds near 0.32 to 0.34.
Early folds are hardest for every model (about 0.23 to 0.34), because with 50 to 66 points of history there is little to learn from.
Horizon pattern: Persistence is best at the first steps and degrades steadily (mean absolute error 0.10 at step 1 to 0.23 to 0.36 over steps 13 to 24), because autocorrelation fades to about zero by lag 16 to 24. The training mean is best at long horizons. AR(3) lies between them and wins the middle steps.
The LSTM fails for a data reason: Each fold gives it only tens of training windows and there is no seasonal structure for it to exploit, so it is worse than every baseline (RMSE 0.4014).
Holdout: BlendLin beats persistence on both metrics (MAE 0.0823 vs 0.1468, RMSE 0.1073 vs 0.1848). The holdout score is lower than the CV score, which indicates an easier window and not a better model.

####Capacity-planning recommendation

Baseline capacity equals the BlendLin point forecast of aggregate utilization.
Provisioned capacity equals the forecast plus a safety margin taken from BlendLin's cross-validation residuals. As an approximation from the CV RMSE of 0.22 percentage points, assuming roughly centered residuals, the 90th-percentile margin is about 0.28 points (1.28 x 0.2201), around 3% of the 9 to 10% fleet level. The empirical residual quantile is preferable where available because it needs no normality assumption.
Scaling trigger. Add capacity when the upper line approaches a fleet resource limit set by the team. Do not scale down below the point forecast inside the horizon.
Segment rule. Track delay-insensitive VMs separately, since they generate most of the fleet's movement, and monitor the volatile small segments for tail risk.
Right-sizing. Independently of forecasting, a fleet averaging under about 10% CPU suggests room to review VM sizes for downsizing or consolidation.

Trade-off: Under-provisioning risks throttling and service-level breaches at the moments demand rises faster than forecast; over-provisioning wastes money. With a fleet averaging under about 10% CPU, a margin of a fraction of a percentage point is cheap insurance, whereas a zero margin would be breached whenever the series moves against the forecast, as persistence does after a temporary rise. The margin should be re-estimated as more data accumulates, because the residual quantiles here come from heavily overlapping folds.

#### Limitations and future work
Short window: Eleven hours cannot show daily or weekly seasonality, and calendar features cannot be justified because timestamps are relative offsets.
Closed cohort: The release contains only VMs alive for the whole window, so active_vm_count is constant. The results say nothing about arrivals and departures in a real fleet.
Selection noise: Overlapping folds make fold-level comparisons correlated, and the top candidates are not statistically separable. A single 24-point holdout cannot rank them.
Unweighted target: The aggregate treats all VMs equally. A core-weighted utilization and VM- or segment-level forecasts would be more capacity-relevant.
Aggregation hides tail risk: The fleet mean is smooth while individual VMs are volatile.
Metadata leakage: Lifetime statistics in the metadata cannot be used as features, which rules out otherwise useful VM-level information: VM-level features computed only from the readings before each origin could be tested.
Future work: Segment-level bottom-up forecasts; mean-reverting models with VM-level features under the same folds; calibrated step-specific prediction intervals once a longer trace is available; online re-fitting for level shifts; comparison with another public workload such as Google ClusterData 2019.

