"""
Standardized linear LASSO regression for coefficient interpretation.

Fits a LassoCV on standardized features predicting fare_amount.
Saves coefficients to lasso_coefs.csv and lasso_coefs.pkl for inspection.

Usage:
    python lasso_interpret.py
"""

import os
import numpy as np
import pandas as pd
import joblib
from sklearn.linear_model import LassoCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score

from preprocess import load_and_prepare, BASE_DIR

# ── Load data ──────────────────────────────────────────────────
print("Loading data ...")
data   = load_and_prepare(run_lasso=False, random_state=42)
X_tr   = data["X_tr"]
X_te   = data["X_te"]
yf_tr  = data["yf_tr"]
yf_te  = data["yf_te"]

# ── Standardize ────────────────────────────────────────────────
scaler = StandardScaler()
X_tr_s = scaler.fit_transform(X_tr)
X_te_s = scaler.transform(X_te)

# ── Fit LassoCV ────────────────────────────────────────────────
print("Fitting LassoCV ...")
lasso = LassoCV(cv=5, max_iter=5000, n_jobs=-1, random_state=42)
lasso.fit(X_tr_s, yf_tr)

print(f"  Best alpha: {lasso.alpha_:.6f}")

# ── Evaluate ───────────────────────────────────────────────────
preds = lasso.predict(X_te_s)
mae   = mean_absolute_error(yf_te, preds)
r2    = r2_score(yf_te, preds)
print(f"  Test MAE:   ${mae:.2f}")
print(f"  Test R²:    {r2:.4f}")

# ── Coefficients ───────────────────────────────────────────────
coef_df = pd.DataFrame({
    "feature":     X_tr.columns,
    "coefficient": lasso.coef_,
}).sort_values("coefficient", key=abs, ascending=False).reset_index(drop=True)

coef_df["abs_coef"] = coef_df["coefficient"].abs()
nonzero = coef_df[coef_df["coefficient"] != 0]
zeroed  = coef_df[coef_df["coefficient"] == 0]

print(f"\n  {len(nonzero)} non-zero / {len(coef_df)} total features\n")
print(f"  {'Feature':<30}  {'Coefficient':>12}")
print(f"  {'-'*44}")
for _, row in nonzero.iterrows():
    print(f"  {row['feature']:<30}  {row['coefficient']:>+12.4f}")
if len(zeroed):
    print(f"\n  Zeroed out: {', '.join(zeroed['feature'].tolist())}")

# ── Save ───────────────────────────────────────────────────────
csv_path = os.path.join(BASE_DIR, "lasso_coefs.csv")
pkl_path = os.path.join(BASE_DIR, "lasso_coefs.pkl")

coef_df.to_csv(csv_path, index=False)
joblib.dump({
    "coef_df":    coef_df,
    "scaler":     scaler,
    "lasso":      lasso,
    "alpha":      lasso.alpha_,
    "test_mae":   mae,
    "test_r2":    r2,
    "feature_names": list(X_tr.columns),
}, pkl_path)

print(f"\n  Saved: lasso_coefs.csv")
print(f"  Saved: lasso_coefs.pkl")
