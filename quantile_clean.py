"""
LightGBM quantile regression + split-conformal calibration for NYC Uber fares.

Story:
  Uber fares are right-skewed, so we model conditional quantiles rather than
  relying on a mean-based predictor. The median gives a robust point estimate;
  the lower and upper quantiles give an asymmetric prediction interval. Split-conformal
  calibration on a held-out set guarantees the empirical coverage we report.

Split strategy (no leakage):
  train  — fit LightGBM boosters
  val    — early stopping only
  cal    — conformal calibration only (never seen during training)
  test   — final evaluation only (reported once at the end)

Usage:
  python quantile_clean.py
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, roc_auc_score

from preprocess import load_and_prepare, BASE_DIR

# ── Config ────────────────────────────────────────────────────
QUANTILES      = [0.05, 0.50, 0.95]
COVERAGE_TARGET = 0.90   # 95th - 5th = 90% nominal interval
RANDOM_STATE   = 42
EARLY_STOP     = 50

# Tuned parameters (from 2-stage CV search on train split).
# Best median quantile loss = 0.7708, best_iter = 3411.
BEST_PARAMS = {
    "num_leaves": 127, "learning_rate": 0.05, "min_child_samples": 60,
    "max_depth": -1,   "min_split_gain": 0.05, "lambda_l2": 0.0,
}
BEST_ROUNDS = 3411

BASE_LGB_PARAMS = {
    "objective": "quantile", "metric": "quantile",
    "verbosity": -1, "n_jobs": -1, "random_state": RANDOM_STATE,
    "subsample": 0.8, "colsample_bytree": 0.8,
}


# ── 1. Load data ──────────────────────────────────────────────
print("=" * 60)
print("  LightGBM Quantile Regression + Conformal Calibration")
print("=" * 60)

data      = load_and_prepare(run_lasso=False, random_state=RANDOM_STATE)
X_tr_full = data["X_tr"]
X_te      = data["X_te"]
yf_tr_full = data["yf_tr"]
yf_te     = data["yf_te"]
model_df  = data["model_df"]


# ── 2. Three-way split of the training pool ───────────────────
X_tr, X_tmp, yf_tr, yf_tmp = train_test_split(
    X_tr_full, yf_tr_full, test_size=0.25, random_state=RANDOM_STATE)
X_val, X_cal, yf_val, yf_cal = train_test_split(
    X_tmp, yf_tmp, test_size=0.50, random_state=RANDOM_STATE)

print(f"\n  train={len(X_tr):,}  val={len(X_val):,}  "
      f"cal={len(X_cal):,}  test={len(X_te):,}\n")


# ── 3. Train quantile models ──────────────────────────────────
print("  Training quantile models ...")
models = {}
for q in QUANTILES:
    ds     = lgb.Dataset(X_tr,  label=yf_tr)
    val_ds = lgb.Dataset(X_val, label=yf_val, reference=ds)
    model  = lgb.train(
        {**BASE_LGB_PARAMS, "alpha": q, **BEST_PARAMS},
        ds, num_boost_round=BEST_ROUNDS,
        valid_sets=[val_ds],
        callbacks=[lgb.early_stopping(EARLY_STOP, verbose=False),
                   lgb.log_evaluation(0)],
    )
    models[q] = model
    print(f"    q={q:.2f}  →  {model.best_iteration:,} rounds")


# ── 4. Raw test predictions + crossing fix ────────────────────
Q_LO, Q_MED, Q_HI = QUANTILES
pred_lower_raw = models[Q_LO].predict(X_te)
pred_median    = models[Q_MED].predict(X_te)
pred_upper_raw = models[Q_HI].predict(X_te)

n_cross    = int(np.sum((pred_lower_raw > pred_median) |
                        (pred_median > pred_upper_raw) |
                        (pred_lower_raw > pred_upper_raw)))
pred_lower = np.minimum(pred_lower_raw, pred_median)
pred_upper = np.maximum(pred_upper_raw, pred_median)

mae  = mean_absolute_error(yf_te, pred_median)
rmse = np.sqrt(mean_squared_error(yf_te, pred_median))

raw_cov   = np.mean((yf_te.values >= pred_lower) & (yf_te.values <= pred_upper))
raw_width = np.mean(pred_upper - pred_lower)

# High-cost diagnostic: can the predicted median rank top-quartile fares?
q3_train       = yf_tr.quantile(0.75)
high_cost_true = (yf_te >= q3_train).astype(int)
high_cost_auc  = roc_auc_score(high_cost_true, pred_median)

print(f"\n  Quantile crossings:  {n_cross}/{len(X_te)} ({n_cross/len(X_te):.2%})")
print(f"  MAE:   ${mae:.2f}    RMSE: ${rmse:.2f}")
print(f"  High-cost trip ROC-AUC: {high_cost_auc:.3f}  "
      f"(median ranking top-quartile fares, threshold ${q3_train:.2f})")
print(f"  Raw {int(COVERAGE_TARGET*100)}% interval — coverage: {raw_cov:.3f}, mean width: ${raw_width:.2f}")


# ── 5. Split-conformal calibration (global, on cal set) ──────
cal_lower_raw = models[Q_LO].predict(X_cal)
cal_median    = models[Q_MED].predict(X_cal)
cal_upper_raw = models[Q_HI].predict(X_cal)
cal_lower = np.minimum(cal_lower_raw, cal_median)
cal_upper = np.maximum(cal_upper_raw, cal_median)
cal_y     = yf_cal.values

scores  = np.maximum(0.0, np.maximum(cal_lower - cal_y, cal_y - cal_upper))
n_cal   = len(scores)
alpha   = 1.0 - COVERAGE_TARGET
q_level = np.ceil((n_cal + 1) * (1 - alpha)) / n_cal
q_hat   = float(np.quantile(scores, min(q_level, 1.0)))

conf_lower = np.maximum(pred_lower - q_hat, 0.0)
conf_upper = pred_upper + q_hat

conf_cov   = np.mean((yf_te.values >= conf_lower) & (yf_te.values <= conf_upper))
conf_width = np.mean(conf_upper - conf_lower)

print(f"\n  Conformal q_hat: ${q_hat:.2f}")
print(f"  Conformal {int(COVERAGE_TARGET*100)}% interval — coverage: {conf_cov:.3f}, mean width: ${conf_width:.2f}")


# ── 5b. Interval quality by predicted fare bin ───────────────
bins   = [0, 6, 10, 15, 25, np.inf]
labels = ["$0–6", "$6–10", "$10–15", "$15–25", "$25+"]
bin_idx = np.digitize(pred_median, bins) - 1  # 0-indexed

bin_labels_out, bin_n, bin_rw, bin_cov = [], [], [], []

print(f"\n  {'Fare bin':>8}  {'n':>6}  {'width':>7}  {'rel_width':>10}  {'coverage':>9}")
print(f"  {'-'*48}")
for i, lbl in enumerate(labels):
    m = bin_idx == i
    if m.sum() == 0:
        continue
    w   = np.mean(conf_upper[m] - conf_lower[m])
    rw  = np.mean((conf_upper[m] - conf_lower[m]) / pred_median[m])
    cov = np.mean((yf_te.values[m] >= conf_lower[m]) & (yf_te.values[m] <= conf_upper[m]))
    print(f"  {lbl:>8}  {m.sum():>6}  ${w:>6.2f}  {rw:>9.1%}  {cov:>9.3f}")
    bin_labels_out.append(lbl)
    bin_n.append(m.sum())
    bin_rw.append(rw)
    bin_cov.append(cov)
print(f"  {'-'*48}")
print(f"  {'ALL':>8}  {len(pred_median):>6}  ${conf_width:>6.2f}  "
      f"{np.mean((conf_upper - conf_lower) / pred_median):>9.1%}  {conf_cov:>9.3f}")


# ── 6. Plots ──────────────────────────────────────────────────
print("\n  Generating plots ...")

plt.rcParams.update({"font.family": "sans-serif", "axes.spines.top": False,
                     "axes.spines.right": False})

# ── Plot 1: Fare distribution + year trend ───────────────────
raw_df = data["df_raw"].copy()
raw_df["pickup_datetime"] = pd.to_datetime(raw_df["pickup_datetime"])
raw_df["year"] = raw_df["pickup_datetime"].dt.year
raw_df = raw_df[(raw_df["fare_amount"] >= 2.5) & (raw_df["fare_amount"] <= 200)]
fares  = raw_df["fare_amount"]
years  = sorted(raw_df["year"].unique())

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

# Histogram panel
ax1.hist(fares, bins=80, range=(0, 80),
         color="#2196F3", edgecolor="white", linewidth=0.3, alpha=0.85)
ax1.axvline(fares.median(), color="#FF5722", lw=2, ls="--",
            label=f"Median  ${fares.median():.2f}")
ax1.axvline(fares.mean(),   color="#4CAF50", lw=2, ls="--",
            label=f"Mean    ${fares.mean():.2f}")
ax1.set_xlabel("Fare ($)", fontsize=11)
ax1.set_ylabel("Trips", fontsize=11)
ax1.set_title("Fare Distribution", fontsize=12, fontweight="bold")
ax1.legend(fontsize=10)
ax1.grid(axis="y", alpha=0.25)

# Year trend panel
medians = [raw_df[raw_df["year"] == y]["fare_amount"].median() for y in years]
p10s    = [raw_df[raw_df["year"] == y]["fare_amount"].quantile(0.10) for y in years]
p90s    = [raw_df[raw_df["year"] == y]["fare_amount"].quantile(0.90) for y in years]

ax2.fill_between(years, p10s, p90s, alpha=0.18, color="#2196F3", label="10th–90th pct")
ax2.plot(years, medians, marker="o", lw=2.5, color="#2196F3", label="Median fare")
ax2.set_xlabel("Year", fontsize=11)
ax2.set_ylabel("Fare ($)", fontsize=11)
ax2.set_title("Fare Level & Spread by Year\n(~23% median shift 2009→2015)", fontsize=12, fontweight="bold")
ax2.set_xticks(years)
ax2.legend(fontsize=10)
ax2.grid(alpha=0.25)

plt.tight_layout()
p1 = os.path.join(BASE_DIR, "fare_distribution_skew.png")
plt.savefig(p1, dpi=150)
plt.close()
print(f"    Saved: fare_distribution_skew.png")


# ── Plot 2: Conformal intervals on 100 sampled test trips ────
np.random.seed(RANDOM_STATE)
idx   = np.random.choice(len(X_te), 100, replace=False)
order = np.argsort(pred_median[idx])
idx   = idx[order]
x     = np.arange(len(idx))

fig, ax = plt.subplots(figsize=(11, 5))
ax.fill_between(x, conf_lower[idx], conf_upper[idx],
                alpha=0.20, color="#4CAF50", label=f"{int(COVERAGE_TARGET*100)}% conformal interval")
ax.fill_between(x, pred_lower[idx], pred_upper[idx],
                alpha=0.35, color="#2196F3", label=f"{int(COVERAGE_TARGET*100)}% raw interval")
ax.scatter(x, yf_te.values[idx], s=18, color="black",   zorder=5, label="Actual fare")
ax.scatter(x, pred_median[idx],  s=18, color="#FF5722",  zorder=4, label="Predicted median")
ax.set_xlabel("Sampled trips  (sorted by predicted fare)", fontsize=11)
ax.set_ylabel("Fare ($)", fontsize=11)
ax.set_title("Predicted Intervals vs Actual Fares\n(100 test trips, sorted by median prediction)",
             fontsize=12, fontweight="bold")
ax.legend(fontsize=10)
ax.grid(alpha=0.25)
ax.set_xticks([])
plt.tight_layout()
p2 = os.path.join(BASE_DIR, "conformal_intervals.png")
plt.savefig(p2, dpi=150)
plt.close()
print(f"    Saved: conformal_intervals.png")


# ── Plot 3: Key metrics summary ──────────────────────────────
summary_lines = [
    ("Metric", "Value"),
    ("─" * 22, "─" * 10),
    ("LGB Median MAE",    f"${mae:.2f}"),
    ("LGB Median RMSE",   f"${rmse:.2f}"),
] + [
    ("─" * 22, "─" * 10),
    ("Raw coverage",      f"{raw_cov:.3f}"),
    ("Raw mean width",    f"${raw_width:.2f}"),
    ("Conformal coverage",f"{conf_cov:.3f}"),
    ("Conformal width",   f"${conf_width:.2f}"),
    ("Conformal q_hat",   f"${q_hat:.2f}"),
    ("─" * 22, "─" * 10),
    ("High-cost ROC-AUC", f"{high_cost_auc:.3f}"),
]

fig, ax_txt = plt.subplots(figsize=(6, 5))
ax_txt.axis("off")
y_pos = 0.93
for left, right in summary_lines:
    ax_txt.text(0.05, y_pos, left,  transform=ax_txt.transAxes,
                fontsize=11, va="top", family="monospace")
    ax_txt.text(0.72, y_pos, right, transform=ax_txt.transAxes,
                fontsize=11, va="top", family="monospace", fontweight="bold")
    y_pos -= 0.072

ax_txt.set_title("Key Metrics", fontsize=13, fontweight="bold")

plt.tight_layout()
p3 = os.path.join(BASE_DIR, "interval_summary.png")
plt.savefig(p3, dpi=150)
plt.close()
print(f"    Saved: interval_summary.png")


# ── Plot 4: Relative interval width by fare bin ──────────────
fig, ax = plt.subplots(figsize=(8, 5))
x_pos = np.arange(len(bin_labels_out))
bars = ax.bar(x_pos, [rw * 100 for rw in bin_rw],
              color="#2196F3", edgecolor="white", linewidth=1.5, width=0.55)
for bar, rw, n in zip(bars, bin_rw, bin_n):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
            f"{rw:.0%}\nn={n:,}", ha="center", va="bottom", fontsize=10,
            fontweight="bold")
ax.set_xticks(x_pos)
ax.set_xticklabels(bin_labels_out, fontsize=11)
ax.set_xlabel("Predicted Fare Bin", fontsize=11)
ax.set_ylabel("Relative Interval Width (%)", fontsize=11)
ax.set_title("Conformal Interval Width Relative to Predicted Fare\n"
             "(lower = more precise)", fontsize=12, fontweight="bold")
ax.set_ylim(0, max(bin_rw) * 100 * 1.35)
ax.grid(axis="y", alpha=0.25)
plt.tight_layout()
p4 = os.path.join(BASE_DIR, "relative_width_by_bin.png")
plt.savefig(p4, dpi=150)
plt.close()
print(f"    Saved: relative_width_by_bin.png")


# ── 7. Save artifact ──────────────────────────────────────────
artifact = {
    "models":            {q: models[q] for q in QUANTILES},
    "conformal_q_hat":   q_hat,
    "conformal_scores":  scores,
    "best_params":       BEST_PARAMS,
    "best_rounds":       BEST_ROUNDS,
    "coverage_target":   COVERAGE_TARGET,
    "high_cost_threshold": float(q3_train),
    "high_cost_auc":     high_cost_auc,
}
out_pkl = os.path.join(BASE_DIR, "quantile_lgb_clean.pkl")
joblib.dump(artifact, out_pkl)

# ── 8. Final summary ──────────────────────────────────────────
print(f"\n{'='*60}")
print(f"  LightGBM Median MAE:      ${mae:.2f}")
print(f"  Raw {int(COVERAGE_TARGET*100)}% coverage:         {raw_cov:.3f}")
print(f"  Raw {int(COVERAGE_TARGET*100)}% mean width:       ${raw_width:.2f}")
print(f"  Conformal {int(COVERAGE_TARGET*100)}% coverage:   {conf_cov:.3f}  (target: {COVERAGE_TARGET})")
print(f"  Conformal {int(COVERAGE_TARGET*100)}% mean width: ${conf_width:.2f}")
print(f"  High-cost ROC-AUC:        {high_cost_auc:.3f}  (top-quartile, >=${q3_train:.2f})")
print(f"{'='*60}")
print(f"\n  Saved: quantile_lgb_clean.pkl")
print("  Done.")
