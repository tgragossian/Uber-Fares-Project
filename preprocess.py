"""
Shared preprocessing pipeline for NYC Uber fare prediction.

Usage:
    from preprocess import load_and_prepare
    data = load_and_prepare()
    X_tr, X_te = data["X_tr"], data["X_te"]
    ...
"""

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV

# ── Config ──────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "uber.csv")
NYC_BOUNDS = {"lon": (-75, -72), "lat": (40, 42)}
AIRPORTS = {
    "JFK": (40.6418, -73.7810, 2.2),
    "LGA": (40.7769, -73.8740, 0.9),
    "EWR": (40.6895, -74.1745, 1.6),
}

CAT_COLS = ["pickup_airport", "dropoff_airport", "time_bin"]
NUM_COLS = [
    "trip_distance_mi", "passenger_count",
    "pickup_longitude", "pickup_latitude",
    "dropoff_longitude", "dropoff_latitude",
    "fri_sat_peak_night", "weekday_rush", "y",
]


# ── Utilities ───────────────────────────────────────────────────
def haversine(lat1, lon1, lat2, lon2):
    """Haversine distance in miles (vectorised)."""
    R = 3958.8
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    a = (np.sin(dlat / 2) ** 2
         + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2))
         * np.sin(dlon / 2) ** 2)
    return R * 2 * np.arcsin(np.sqrt(a))


def get_time_bin(h):
    if pd.isna(h):
        return None
    bins = [
        (4, 6, "early_morning"), (6, 10, "morning_rush"),
        (10, 16, "midday"), (16, 20, "evening_rush"),
        (20, 23, "evening"),
    ]
    return next((b for s, e, b in bins if s <= int(h) < e), "late_night")


def engineer_features(df):
    """Add all engineered columns in-place and return df."""
    df["pickup_datetime"] = pd.to_datetime(df["pickup_datetime"], errors="coerce")
    df["h"] = df["pickup_datetime"].dt.hour
    df["wd"] = df["pickup_datetime"].dt.dayofweek
    df["y"] = df["pickup_datetime"].dt.year
    df["trip_distance_mi"] = haversine(
        df["pickup_latitude"], df["pickup_longitude"],
        df["dropoff_latitude"], df["dropoff_longitude"],
    )
    df["time_bin"] = df["h"].apply(get_time_bin)
    df["fri_sat_peak_night"] = (
        (df["wd"].isin([4, 5])) & ((df["h"] >= 23) | (df["h"] < 4))
    ).astype(int)
    df["weekday_rush"] = (
        (df["wd"] < 5)
        & (((df["h"] >= 6) & (df["h"] < 10)) | ((df["h"] >= 16) & (df["h"] < 20)))
    ).astype(int)

    for loc in ["pickup", "dropoff"]:
        df[f"{loc}_airport"] = None
        for name, (alat, alon, radius) in AIRPORTS.items():
            mask = haversine(df[f"{loc}_latitude"], df[f"{loc}_longitude"], alat, alon) <= radius
            df.loc[mask, f"{loc}_airport"] = name
    return df


# ── Main loader ─────────────────────────────────────────────────
def load_and_prepare(run_lasso=True, random_state=42):
    """
    Load uber.csv, filter to NYC, engineer features, OHE, split, and
    optionally run LASSO feature selection.

    Returns a dict with everything downstream scripts need.
    """
    print("Loading data ...")
    df_raw = pd.read_csv(DATA_PATH)
    df = df_raw[
        df_raw["pickup_longitude"].between(*NYC_BOUNDS["lon"])
        & df_raw["dropoff_longitude"].between(*NYC_BOUNDS["lon"])
        & df_raw["pickup_latitude"].between(*NYC_BOUNDS["lat"])
        & df_raw["dropoff_latitude"].between(*NYC_BOUNDS["lat"])
    ].copy()
    print(f"  Raw: {len(df_raw):,} -> Filtered: {len(df):,}")

    df = engineer_features(df)
    model_df = df[
        (df["trip_distance_mi"] > 0.1)
        & (df["fare_amount"] > 0)
        & (df["fare_amount"] < 500)
    ].dropna(subset=["trip_distance_mi", "passenger_count", "pickup_datetime", "fare_amount"]).copy()
    model_df["fare_per_mile"] = model_df["fare_amount"] / model_df["trip_distance_mi"]

    thresh = model_df["fare_amount"].quantile(0.75)
    model_df["is_expensive"] = (model_df["fare_amount"] > thresh).astype(int)
    print(f"  Expensive threshold: ${thresh:.2f}")

    # One-hot encode
    X = pd.get_dummies(model_df[NUM_COLS + CAT_COLS], columns=CAT_COLS, drop_first=True)
    y_cls = model_df["is_expensive"]
    y_fare = model_df["fare_amount"]
    y_fpm = model_df["fare_per_mile"]

    # Train / test split (80/20, stratified on classification target)
    X_tr, X_te, ycls_tr, ycls_te, yf_tr, yf_te, ym_tr, ym_te = train_test_split(
        X, y_cls, y_fare, y_fpm,
        test_size=0.2, random_state=random_state, stratify=y_cls,
    )

    # LASSO feature selection
    sel_feats = list(X.columns)
    if run_lasso:
        print("  Running LASSO feature selection ...")
        lasso = GridSearchCV(
            Pipeline([
                ("scaler", StandardScaler()),
                ("logreg", LogisticRegression(solver="liblinear", penalty="l1", max_iter=1000)),
            ]),
            {"logreg__C": [0.01, 0.1, 1.0, 10.0]},
            cv=5, scoring="roc_auc", n_jobs=-1,
        )
        lasso.fit(X_tr, ycls_tr)
        sel_feats = X.columns[lasso.best_estimator_.named_steps["logreg"].coef_[0] != 0].tolist()
        print(f"  Selected {len(sel_feats)} / {len(X.columns)} features")

    X_tr, X_te = X_tr[sel_feats], X_te[sel_feats]

    return {
        "X_tr": X_tr, "X_te": X_te,
        "ycls_tr": ycls_tr, "ycls_te": ycls_te,
        "yf_tr": yf_tr, "yf_te": yf_te,
        "ym_tr": ym_tr, "ym_te": ym_te,
        "sel_feats": sel_feats,
        "thresh": thresh,
        "model_df": model_df,
        "df_raw": df_raw, "df_filtered": df,
    }


if __name__ == "__main__":
    data = load_and_prepare()
    print(f"\nReady: {len(data['X_tr']):,} train, {len(data['X_te']):,} test, "
          f"{len(data['sel_feats'])} features")
