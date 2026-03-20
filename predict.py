"""
Fare prediction tool for NYC Uber trips.

Loads the trained LightGBM quantile models + conformal calibration and
produces a point estimate (median) and a calibrated prediction interval
for a single trip you specify.

Usage (interactive):
    python predict.py

Usage (command-line):
    python predict.py \
        --pickup-lat 40.7128 --pickup-lon -74.0060 \
        --dropoff-lat 40.7589 --dropoff-lon -73.9851 \
        --datetime "2015-06-15 14:30" \
        --passengers 2
"""

import argparse
import os
import numpy as np
import pandas as pd
import joblib

from preprocess import BASE_DIR, haversine, get_time_bin, AIRPORTS, CAT_COLS, NUM_COLS

# ── Load artifacts ─────────────────────────────────────────────
ARTIFACT_PATH = os.path.join(BASE_DIR, "quantile_lgb_clean.pkl")
SEL_FEATS_PATH = os.path.join(BASE_DIR, "sel_feats.pkl")

artifact  = joblib.load(ARTIFACT_PATH)
models    = artifact["models"]
q_hat     = artifact["conformal_q_hat"]
cov_pct   = int(artifact["coverage_target"] * 100)
hc_thresh = artifact.get("high_cost_threshold")

sel_feats = joblib.load(SEL_FEATS_PATH)

# Derive quantile keys from the stored models dict
q_keys = sorted(models.keys())
Q_LO, Q_MED, Q_HI = q_keys[0], q_keys[1], q_keys[2]


# ── Feature builder ────────────────────────────────────────────
def build_features(pickup_lat, pickup_lon, dropoff_lat, dropoff_lon,
                   dt: pd.Timestamp, passengers: int) -> pd.DataFrame:
    h  = dt.hour
    wd = dt.dayofweek   # 0=Mon … 6=Sun
    y  = dt.year

    dist = haversine(pickup_lat, pickup_lon, dropoff_lat, dropoff_lon)
    tbin = get_time_bin(h)

    fri_sat_peak = int((wd in [4, 5]) and (h >= 23 or h < 4))
    wday_rush    = int((wd < 5) and ((6 <= h < 10) or (16 <= h < 20)))

    def nearest_airport(lat, lon):
        for name, (alat, alon, radius) in AIRPORTS.items():
            if haversine(lat, lon, alat, alon) <= radius:
                return name
        return None

    row = {
        "trip_distance_mi":   dist,
        "passenger_count":    passengers,
        "pickup_longitude":   pickup_lon,
        "pickup_latitude":    pickup_lat,
        "dropoff_longitude":  dropoff_lon,
        "dropoff_latitude":   dropoff_lat,
        "fri_sat_peak_night": fri_sat_peak,
        "weekday_rush":       wday_rush,
        "y":                  y,
        "pickup_airport":     nearest_airport(pickup_lat, pickup_lon),
        "dropoff_airport":    nearest_airport(dropoff_lat, dropoff_lon),
        "time_bin":           tbin,
    }

    df = pd.DataFrame([row])
    X  = pd.get_dummies(df[NUM_COLS + CAT_COLS], columns=CAT_COLS, drop_first=True)

    # Align to training columns: add missing as 0, drop extras
    X = X.reindex(columns=sel_feats, fill_value=0)
    return X, row


# ── Predict ────────────────────────────────────────────────────
def predict(X):
    lo_raw  = models[Q_LO].predict(X)[0]
    med     = models[Q_MED].predict(X)[0]
    hi_raw  = models[Q_HI].predict(X)[0]

    lo = min(lo_raw, med)
    hi = max(hi_raw, med)

    conf_lo = max(lo - q_hat, 0.0)
    conf_hi = hi + q_hat

    return med, conf_lo, conf_hi


# ── I/O helpers ────────────────────────────────────────────────
def prompt_float(msg, lo=None, hi=None):
    while True:
        try:
            v = float(input(msg).strip())
            if lo is not None and v < lo:
                print(f"  Must be >= {lo}")
                continue
            if hi is not None and v > hi:
                print(f"  Must be <= {hi}")
                continue
            return v
        except ValueError:
            print("  Please enter a number.")


def prompt_int(msg, lo=None, hi=None):
    while True:
        try:
            v = int(input(msg).strip())
            if lo is not None and v < lo:
                print(f"  Must be >= {lo}")
                continue
            if hi is not None and v > hi:
                print(f"  Must be <= {hi}")
                continue
            return v
        except ValueError:
            print("  Please enter a whole number.")


def prompt_datetime(msg):
    fmts = ["%Y-%m-%d %H:%M", "%Y-%m-%d %H:%M:%S", "%Y-%m-%d"]
    while True:
        raw = input(msg).strip()
        for fmt in fmts:
            try:
                return pd.Timestamp(raw)
            except Exception:
                pass
        print("  Use format: YYYY-MM-DD HH:MM  (e.g. 2014-08-22 19:45)")


# ── Main ───────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Predict NYC Uber fare")
    parser.add_argument("--pickup-lat",  type=float)
    parser.add_argument("--pickup-lon",  type=float)
    parser.add_argument("--dropoff-lat", type=float)
    parser.add_argument("--dropoff-lon", type=float)
    parser.add_argument("--datetime",    type=str,
                        help="e.g. '2015-06-15 14:30'")
    parser.add_argument("--passengers",  type=int, default=None)
    args = parser.parse_args()

    print("\n" + "=" * 52)
    print("  NYC Uber Fare Predictor")
    print("=" * 52)

    # Gather inputs — use CLI args if provided, else prompt
    if args.pickup_lat is not None:
        pu_lat = args.pickup_lat
        pu_lon = args.pickup_lon
        do_lat = args.dropoff_lat
        do_lon = args.dropoff_lon
        dt     = pd.Timestamp(args.datetime)
        pax    = args.passengers if args.passengers else 1
    else:
        print("\nPickup location:")
        pu_lat = prompt_float("  Latitude  (e.g. 40.7128): ", 40, 42)
        pu_lon = prompt_float("  Longitude (e.g. -74.006): ", -75, -72)

        print("\nDropoff location:")
        do_lat = prompt_float("  Latitude  (e.g. 40.7589): ", 40, 42)
        do_lon = prompt_float("  Longitude (e.g. -73.985): ", -75, -72)

        print("\nTrip datetime  (model trained on 2009–2015):")
        dt = prompt_datetime("  Date & time (YYYY-MM-DD HH:MM): ")

        print("\nPassenger count:")
        pax = prompt_int("  Passengers (1–6): ", 1, 6)

    # Build features and predict
    X, info = build_features(pu_lat, pu_lon, do_lat, do_lon, dt, pax)
    med, conf_lo, conf_hi = predict(X)

    # Derived info for display
    dist_mi  = info["trip_distance_mi"]
    tbin     = info["time_bin"].replace("_", " ")
    pu_apt   = info["pickup_airport"]  or "none"
    do_apt   = info["dropoff_airport"] or "none"
    is_rush  = bool(info["weekday_rush"])
    is_peak  = bool(info["fri_sat_peak_night"])

    print("\n" + "─" * 52)
    print("  Trip summary")
    print("─" * 52)
    print(f"  Distance:         {dist_mi:.2f} mi")
    print(f"  Time bin:         {tbin}")
    print(f"  Weekday rush:     {'yes' if is_rush else 'no'}")
    print(f"  Fri/Sat peak:     {'yes' if is_peak else 'no'}")
    print(f"  Pickup airport:   {pu_apt}")
    print(f"  Dropoff airport:  {do_apt}")

    print("\n" + "─" * 52)
    print("  Fare prediction")
    print("─" * 52)
    print(f"  Predicted fare:   ${med:.2f}  (median)")
    print(f"  {cov_pct}% interval:    ${conf_lo:.2f} – ${conf_hi:.2f}  "
          f"(width ${conf_hi - conf_lo:.2f})")

    if hc_thresh is not None:
        flag = "YES — top-quartile fare" if med >= hc_thresh else "no"
        print(f"  High-cost flag:   {flag}  (threshold ${hc_thresh:.2f})")

    print("─" * 52 + "\n")


if __name__ == "__main__":
    main()
