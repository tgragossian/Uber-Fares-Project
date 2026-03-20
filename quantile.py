"""
LightGBM quantile regression with conformal calibration for NYC Uber fares.

Trains quantile models (0.05, 0.50, 0.95), applies split-conformal
calibration on a held-out calibration set, and generates diagnostic plots.

Split strategy (no leakage):
    train  — fit LightGBM boosters
    val    — early stopping only
    cal    — conformal scores only (never seen during training)
    test   — final evaluation

Tuning strategy:
    Stage 1 — coarse sweep (~48 combos) to find winning region
    Stage 2 — zoom in on max_depth + l2 around stage 1 winner

Usage:
    python quantile.py
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import lightgbm as lgb
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error

from preprocess import load_and_prepare, BASE_DIR

# ── Config ──────────────────────────────────────────────────────
QUANTILES = [0.05, 0.50, 0.95]
RANDOM_STATE = 42
COVERAGE_TARGET = 0.90  # 95-5 = 90% nominal coverage
MAX_ROUNDS = 10_000
EARLY_STOP_CV = 30       # patience during CV
EARLY_STOP_FINAL = 50    # patience for final models (more room)


# ── 1. Load preprocessed data ──────────────────────────────────
print("=" * 60)
print("LIGHTGBM QUANTILE REGRESSION + CONFORMAL CALIBRATION")
print("=" * 60)

data = load_and_prepare(run_lasso=True, random_state=RANDOM_STATE)
X_tr_full = data["X_tr"]
X_te = data["X_te"]
yf_tr_full = data["yf_tr"]
yf_te = data["yf_te"]
model_df = data["model_df"]


# ── 2. Three-way split: train / val / cal ───────────────────────
#    val  = early stopping (touches training procedure)
#    cal  = conformal scores ONLY (never seen during training)
X_tr, X_tmp, yf_tr, yf_tmp = train_test_split(
    X_tr_full, yf_tr_full, test_size=0.25, random_state=RANDOM_STATE,
)
X_val, X_cal, yf_val, yf_cal = train_test_split(
    X_tmp, yf_tmp, test_size=0.50, random_state=RANDOM_STATE,
)
print(f"\nSplits: train={len(X_tr):,}, val={len(X_val):,}, "
      f"cal={len(X_cal):,}, test={len(X_te):,}")


# ── Helper: run CV for a param grid ────────────────────────────
train_ds = lgb.Dataset(X_tr, label=yf_tr)
kf = KFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
folds = list(kf.split(X_tr))

base_params = {
    "objective": "quantile",
    "alpha": 0.50,
    "metric": "quantile",
    "verbosity": -1,
    "n_jobs": -1,
    "random_state": RANDOM_STATE,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
}


def run_cv_grid(grid, label=""):
    """Run CV over a param grid, return (best_params, best_rounds, best_score, all_results)."""
    best_score = np.inf
    best_params = None
    best_rounds = MAX_ROUNDS
    all_results = []

    print(f"\n{label} ({len(grid)} combos)\n")
    for i, p in enumerate(grid):
        params = {**base_params, **p}
        cv_result = lgb.cv(
            params, train_ds, num_boost_round=MAX_ROUNDS,
            folds=folds,
            callbacks=[lgb.early_stopping(EARLY_STOP_CV, verbose=False)],
            return_cvbooster=False,
        )
        key = [k for k in cv_result.keys() if "mean" in k][0]
        vals = cv_result[key]
        b_iter = int(np.argmin(vals) + 1)
        score = float(vals[b_iter - 1])
        ran = len(vals)

        is_new = score < best_score
        if is_new:
            best_score = score
            best_params = p
            best_rounds = b_iter

        all_results.append({**p, "loss": score, "best_iter": b_iter, "ran": ran})
        marker = " ***" if is_new else ""
        print(f"  [{i+1:3d}/{len(grid)}] "
              f"md={p['max_depth']}, l2={p['lambda_l2']}, "
              f"msg={p.get('min_split_gain',0.0)}, mcs={p['min_child_samples']} "
              f"-> loss={score:.4f} (best_iter={b_iter}, ran={ran}){marker}")

    return best_params, best_rounds, best_score, all_results


# ── 3a. Stage 1 results (already completed) ────────────────────
#    Top 5 from previous run:
#      loss=0.7717 iter=2827 | lr=0.05, leaves=127, mcs=50, md=10,  l2=0.0
#      loss=0.7717 iter=2546 | lr=0.05, leaves=127, mcs=50, md=-1,  l2=1.0
#      loss=0.7724 iter=2309 | lr=0.05, leaves=127, mcs=50, md=10,  l2=1.0
#      loss=0.7725 iter=2186 | lr=0.05, leaves=127, mcs=50, md=-1,  l2=0.0
#      loss=0.7726 iter=3046 | lr=0.05, leaves=63,  mcs=50, md=10,  l2=0.0
#    Conclusion: lr=0.05, leaves=127 locked in. mcs bumped to 60 in Stage 2.
s1_score = 0.7717
s1_rounds = 2827
print(f"\nStage 1 (cached): best loss={s1_score:.4f}, best_iter={s1_rounds}")


# ── 3b. Stage 2: zoom in on max_depth + l2 ─────────────────────
stage2_grid = [
    {"num_leaves": 127, "learning_rate": 0.05, "min_child_samples": 60,
     "max_depth": md, "min_split_gain": msg, "lambda_l2": l2}
    for md in [-1, 8, 10, 13]
    for l2 in [0.0, 0.5, 1.0, 2.0]
    for msg in [0.0, 0.05]
]

s2_params, s2_rounds, s2_score, s2_results = run_cv_grid(stage2_grid, "Stage 2: zoom in (md × l2)")

print(f"\nStage 2 best: {s2_params}")
print(f"  loss={s2_score:.4f}, best_iter={s2_rounds}")

s2_df = pd.DataFrame(s2_results).sort_values("loss")
print(f"\nStage 2 top 5:")
for _, row in s2_df.head(5).iterrows():
    print(f"  loss={row['loss']:.4f} iter={int(row['best_iter']):4d} | "
          f"md={int(row['max_depth'])}, l2={row['lambda_l2']}, "
          f"msg={row['min_split_gain']}, mcs={int(row['min_child_samples'])}")

# Pick overall best across both stages
if s2_score <= s1_score:
    best_params, best_rounds = s2_params, s2_rounds
    print(f"\nFinal winner from Stage 2: loss={s2_score:.4f}")
else:
    best_params = {"num_leaves": 127, "learning_rate": 0.05, "min_child_samples": 50,
                    "max_depth": 10, "min_split_gain": 0.0, "lambda_l2": 0.0}
    best_rounds = s1_rounds
    print(f"\nFinal winner from Stage 1: loss={s1_score:.4f}")
print(f"  params={best_params}, best_iter={best_rounds}")


# ── 4. Train final quantile models (early stop on val, NOT cal) ─
print("\nTraining final quantile models (0.05, 0.50, 0.95) ...")
models = {}

for q in QUANTILES:
    params = {**base_params, "alpha": q, **best_params}
    ds = lgb.Dataset(X_tr, label=yf_tr)
    val_ds = lgb.Dataset(X_val, label=yf_val, reference=ds)

    model = lgb.train(
        params, ds, num_boost_round=best_rounds,
        valid_sets=[val_ds],
        callbacks=[
            lgb.early_stopping(EARLY_STOP_FINAL, verbose=False),
            lgb.log_evaluation(0),
        ],
    )
    models[q] = model
    print(f"  q={q:.2f}: {model.best_iteration} rounds")


# ── 5. Raw predictions on test set ─────────────────────────────
pred_lower_raw = models[0.05].predict(X_te)
pred_median = models[0.50].predict(X_te)
pred_upper_raw = models[0.95].predict(X_te)

# Quantile crossing fix: ensure lower <= median <= upper
n_crossings = int(np.sum((pred_lower_raw > pred_median) |
                          (pred_median > pred_upper_raw) |
                          (pred_lower_raw > pred_upper_raw)))
pred_lower = np.minimum(pred_lower_raw, pred_median)
pred_upper = np.maximum(pred_upper_raw, pred_median)
print(f"  Quantile crossings on test: {n_crossings}/{len(X_te)} ({n_crossings/len(X_te):.2%})")

mae = mean_absolute_error(yf_te, pred_median)
rmse = np.sqrt(mean_squared_error(yf_te, pred_median))
print(f"\nLightGBM Median — MAE: ${mae:.2f}, RMSE: ${rmse:.2f}")

raw_coverage = np.mean((yf_te.values >= pred_lower) & (yf_te.values <= pred_upper))
raw_width = np.mean(pred_upper - pred_lower)
print(f"Raw 90% interval — coverage: {raw_coverage:.3f}, mean width: ${raw_width:.2f}")


# ── 6. Mondrian conformal calibration (per-year, on cal set) ───
print("\nApplying Mondrian (per-year) conformal calibration ...")

cal_lower_raw = models[0.05].predict(X_cal)
cal_upper_raw = models[0.95].predict(X_cal)
cal_median = models[0.50].predict(X_cal)

# Same crossing fix on cal
cal_lower = np.minimum(cal_lower_raw, cal_median)
cal_upper = np.maximum(cal_upper_raw, cal_median)
cal_actual = yf_cal.values

# Nonconformity scores (global, for diagnostics)
scores = np.maximum(0.0, np.maximum(cal_lower - cal_actual, cal_actual - cal_upper))

# Per-year q_hat on cal
cal_years = X_cal["y"].values
te_years = X_te["y"].values
alpha = 1.0 - COVERAGE_TARGET
q_hat_by_year = {}

print(f"\n  {'Year':>4}  {'n_cal':>5}  {'q_hat':>6}  {'mean_score':>10}  {'pct_zero':>8}")
for yr in sorted(np.unique(cal_years)):
    mask = cal_years == yr
    yr_scores = scores[mask]
    n_yr = len(yr_scores)
    q_level = np.ceil((n_yr + 1) * (1 - alpha)) / n_yr
    q_hat_by_year[int(yr)] = float(np.quantile(yr_scores, min(q_level, 1.0)))
    print(f"  {int(yr):>4}  {n_yr:>5}  ${q_hat_by_year[int(yr)]:.2f}  "
          f"${np.mean(yr_scores):>9.2f}  {np.mean(yr_scores == 0):>7.1%}")

# Apply per-year q_hat to test set
q_hat_te = np.array([q_hat_by_year[int(y)] for y in te_years])
conf_lower = np.maximum(pred_lower - q_hat_te, 0.0)
conf_upper = pred_upper + q_hat_te

conf_coverage = np.mean((yf_te.values >= conf_lower) & (yf_te.values <= conf_upper))
conf_width = np.mean(conf_upper - conf_lower)
print(f"\n  Mondrian 90% interval — coverage: {conf_coverage:.3f}, mean width: ${conf_width:.2f}")

# Per-year coverage on test
print(f"\n  {'Year':>4}  {'n_test':>6}  {'coverage':>8}  {'width':>6}  {'q_hat':>6}")
for yr in sorted(np.unique(te_years)):
    mask = te_years == yr
    yr_cov = np.mean((yf_te.values[mask] >= conf_lower[mask]) & (yf_te.values[mask] <= conf_upper[mask]))
    yr_width = np.mean(conf_upper[mask] - conf_lower[mask])
    print(f"  {int(yr):>4}  {mask.sum():>6}  {yr_cov:>8.3f}  ${yr_width:>5.2f}  ${q_hat_by_year[int(yr)]:.2f}")


# ── 7. Plots ───────────────────────────────────────────────────
print("\nGenerating plots ...")

# --- Prediction intervals (100 sampled trips) ---
np.random.seed(RANDOM_STATE)
idx = np.random.choice(len(X_te), 100, replace=False)
order = np.argsort(pred_median[idx])
idx = idx[order]

fig, ax = plt.subplots(figsize=(10, 5))
x = np.arange(len(idx))
ax.fill_between(x, conf_lower[idx], conf_upper[idx],
                alpha=0.2, color="#4CAF50", label="90% conformal interval")
ax.fill_between(x, pred_lower[idx], pred_upper[idx],
                alpha=0.35, color="#2196F3", label="90% raw interval")
ax.scatter(x, yf_te.values[idx], s=20, color="black", zorder=5, label="Actual fare")
ax.scatter(x, pred_median[idx], s=20, color="#FF5722", zorder=4, label="LGB median")
ax.set_xlabel("Sampled trips (sorted by predicted fare)", fontsize=12)
ax.set_ylabel("Fare ($)", fontsize=12)
ax.set_title("LightGBM Fare Predictions with Conformal Intervals", fontsize=14, fontweight="bold")
ax.legend(fontsize=10)
ax.grid(alpha=0.3)
ax.set_xticks([])
plt.tight_layout()
plt.savefig(os.path.join(BASE_DIR, "conformal_intervals.png"), dpi=150)
plt.close()
print("  Saved: conformal_intervals.png")

# --- Uncertainty by time bin ---
te_time_bins = model_df.loc[X_te.index, "time_bin"].values
interval_widths = conf_upper - conf_lower

time_bin_order = ["early_morning", "morning_rush", "midday", "evening_rush", "evening", "late_night"]
time_bin_labels = ["Early\nMorning", "Morning\nRush", "Midday", "Evening\nRush", "Evening", "Late\nNight"]

avg_width_by_bin = {}
for tb in time_bin_order:
    mask = te_time_bins == tb
    if mask.sum() > 0:
        avg_width_by_bin[tb] = np.mean(interval_widths[mask])

bins_present = [tb for tb in time_bin_order if tb in avg_width_by_bin]
labels = [time_bin_labels[time_bin_order.index(tb)] for tb in bins_present]
widths = [avg_width_by_bin[tb] for tb in bins_present]

colors = ["#2196F3" if w == min(widths) else "#FF5722" if w == max(widths) else "#78909C" for w in widths]

fig, ax = plt.subplots(figsize=(9, 5))
bars = ax.bar(labels, widths, color=colors, edgecolor="white", linewidth=1.5)
ax.set_ylabel("Mean Interval Width ($)", fontsize=12)
ax.set_title("Fare Uncertainty by Time of Day", fontsize=14, fontweight="bold")
ax.grid(axis="y", alpha=0.3)
for bar, val in zip(bars, widths):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05,
            f"${val:.2f}", ha="center", va="bottom", fontsize=11, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(BASE_DIR, "conformal_uncertainty_by_time.png"), dpi=150)
plt.close()
print("  Saved: conformal_uncertainty_by_time.png")


# ── 8. Save artifacts ──────────────────────────────────────────
artifact = {
    "models": {q: models[q] for q in QUANTILES},
    "conformal_q_hat_by_year": q_hat_by_year,
    "conformal_scores": scores,
    "best_params": best_params,
    "best_rounds": best_rounds,
}
joblib.dump(artifact, os.path.join(BASE_DIR, "quantile_lgb.pkl"))
print("\nSaved: quantile_lgb.pkl")

# Compare to RF
rf_path = os.path.join(BASE_DIR, "reg_fare_rf.pkl")
rf_mae = None
if os.path.exists(rf_path):
    rf_pred = joblib.load(rf_path).predict(X_te)
    rf_mae = mean_absolute_error(yf_te, rf_pred)
else:
    print("  Warning: reg_fare_rf.pkl not found; skipping RF baseline.")

print(f"\n{'='*60}")
print(f"LightGBM Median MAE:    ${mae:.2f}")
if rf_mae is not None:
    print(f"Random Forest MAE:      ${rf_mae:.2f}")
print(f"Conformal 90% coverage: {conf_coverage:.3f} (target: {COVERAGE_TARGET})")
print(f"Conformal 90% width:    ${conf_width:.2f}")
print(f"{'='*60}")
print("\nDone.")
