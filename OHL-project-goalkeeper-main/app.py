"""
OHL Goalkeeper Scouting App — Flask backend
Run with: python app.py
Then open: http://localhost:5000
"""
import json, pickle, os
from pathlib import Path
from flask import Flask, request, jsonify, send_from_directory

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

app = Flask(__name__, static_folder=".")

# ── Paths — adjust if needed ──────────────────────────────────────────────
BASE   = Path(__file__).parent
CSV_UP = BASE / "gk_features_up_sample.csv"   # produced by Step 3f of the notebook

# ── Load data & train model on startup ────────────────────────────────────
print("Loading data...")
up_df = pd.read_csv(CSV_UP)

META_COLS    = ["playerId", "name", "status", "direction", "y_plays", "y_up"]
feature_cols = [c for c in up_df.columns if c not in META_COLS]

X     = up_df[feature_cols].values
y_up  = up_df["y_up"].values

print(f"  {len(up_df)} keepers, {len(feature_cols)} features")

final_model = Pipeline([
    ("scaler", StandardScaler()),
    ("clf",    LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42))
])
final_model.fit(X, y_up)
print("Model trained ✓")

# ── Pre-compute probabilities for all keepers ─────────────────────────────
all_probas = final_model.predict_proba(X)[:, 1]
up_df      = up_df.copy()
up_df["proba"] = all_probas

def confidence_label(p):
    if p >= 0.70: return "HIGH"
    if p >= 0.50: return "MODERATE"
    if p >= 0.35: return "BORDERLINE"
    return "LOW"

def format_feature_name(col):
    return col.replace("mean_", "").replace("_", " ").title()

# ── Routes ────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return send_from_directory(".", "index.html")

@app.route("/api/ranking")
def ranking():
    """Return all keepers ranked by probability."""
    status_filter = request.args.get("status", "ALL")
    top_n         = int(request.args.get("top_n", 218))

    df = up_df.copy()
    if status_filter != "ALL":
        df = df[df["status"] == status_filter]

    df = df.sort_values("proba", ascending=False).head(top_n)

    rows = []
    for rank, (_, row) in enumerate(df.iterrows(), 1):
        rows.append({
            "rank"       : rank,
            "playerId"   : int(row["playerId"]),
            "name"       : row["name"],
            "status"     : row["status"],
            "probability": round(float(row["proba"]) * 100, 1),
            "confidence" : confidence_label(row["proba"]),
            "matches"    : int(row["n_matches_loaded"]),
            "actual_up"  : int(row["y_up"]),
        })

    return jsonify({
        "total"  : len(rows),
        "keepers": rows
    })

@app.route("/api/predict")
def predict():
    """Return full scouting report for a single goalkeeper."""
    query = request.args.get("q", "").strip()
    if not query:
        return jsonify({"error": "No query provided"}), 400

    # Match by name or playerId
    if query.isdigit():
        matches = up_df[up_df["playerId"] == int(query)]
    else:
        matches = up_df[up_df["name"].str.contains(query, case=False, na=False)]

    if len(matches) == 0:
        return jsonify({"error": f"No goalkeeper found for '{query}'"}), 404

    row   = matches.iloc[0]
    feat  = row[feature_cols].values.reshape(1, -1)
    proba = final_model.predict_proba(feat)[0][1]

    # Feature contributions (coef * scaled_value)
    coefs       = final_model.named_steps["clf"].coef_[0]
    feat_scaled = final_model.named_steps["scaler"].transform(feat)[0]
    contribs    = list(zip(feature_cols, (coefs * feat_scaled).tolist()))
    contribs_sorted = sorted(contribs, key=lambda x: x[1], reverse=True)

    top_for     = [{"feature": format_feature_name(f), "value": round(v, 3)} for f, v in contribs_sorted[:4] if v > 0]
    top_against = [{"feature": format_feature_name(f), "value": round(v, 3)} for f, v in contribs_sorted[-4:] if v < 0]

    actual_label = "UP (PLAYS or BENCH)" if row["y_up"] == 1 else "NOT UP (STAYED or DROPPED)"

    return jsonify({
        "playerId"    : int(row["playerId"]),
        "name"        : row["name"],
        "status"      : row["status"],
        "actual_label": actual_label,
        "actual_up"   : int(row["y_up"]),
        "matches"     : int(row["n_matches_loaded"]),
        "probability" : round(float(proba) * 100, 1),
        "confidence"  : confidence_label(proba),
        "prediction"  : "YES" if proba >= 0.5 else "NO",
        "correct"     : bool((proba >= 0.5) == bool(row["y_up"])),
        "top_for"     : top_for,
        "top_against" : top_against,
    })

@app.route("/api/features")
def features():
    """Return normalised feature values for a keeper (for radar chart)."""
    query = request.args.get("q", "").strip()
    if not query:
        return jsonify({"error": "No query provided"}), 400
    if query.isdigit():
        matches = up_df[up_df["playerId"] == int(query)]
    else:
        matches = up_df[up_df["name"].str.contains(query, case=False, na=False)]
    if len(matches) == 0:
        return jsonify({"error": f"No goalkeeper found for '{query}'"}), 404
    row  = matches.iloc[0]
    raw_vals = row[feature_cols].values.astype(float)
    mins = up_df[feature_cols].values.min(axis=0)
    maxs = up_df[feature_cols].values.max(axis=0)
    rng  = np.where(maxs - mins == 0, 1, maxs - mins)
    norm = ((raw_vals - mins) / rng).tolist()
    return jsonify({
        "playerId"     : int(row["playerId"]),
        "name"         : row["name"],
        "feature_names": [format_feature_name(c) for c in feature_cols],
        "raw_values"   : [round(float(v), 4) for v in raw_vals],
        "norm_values"  : [round(float(v), 4) for v in norm],
    })


@app.route("/api/compare")
def compare():
    """Return side-by-side data for two goalkeepers."""
    q1 = request.args.get("q1", "").strip()
    q2 = request.args.get("q2", "").strip()
    if not q1 or not q2:
        return jsonify({"error": "Provide q1 and q2"}), 400

    raw_all = up_df[feature_cols].values.astype(float)
    mins    = raw_all.min(axis=0)
    maxs    = raw_all.max(axis=0)
    rng     = np.where(maxs - mins == 0, 1, maxs - mins)

    results = []
    for q in [q1, q2]:
        if q.isdigit():
            row = up_df[up_df["playerId"] == int(q)]
        else:
            row = up_df[up_df["name"].str.contains(q, case=False, na=False)]
        if len(row) == 0:
            return jsonify({"error": f"No goalkeeper found for '{q}'"}), 404
        row   = row.iloc[0]
        feat  = row[feature_cols].values.reshape(1, -1).astype(float)
        proba = final_model.predict_proba(feat)[0][1]
        coefs       = final_model.named_steps["clf"].coef_[0]
        feat_scaled = final_model.named_steps["scaler"].transform(feat)[0]
        contribs    = sorted(zip(feature_cols, (coefs * feat_scaled).tolist()),
                             key=lambda x: x[1], reverse=True)
        top_for     = [{"feature": format_feature_name(f), "value": round(v, 3)}
                       for f, v in contribs[:4] if v > 0]
        top_against = [{"feature": format_feature_name(f), "value": round(v, 3)}
                       for f, v in contribs[-4:] if v < 0]
        norm = ((feat[0] - mins) / rng).tolist()
        results.append({
            "playerId"   : int(row["playerId"]),
            "name"       : row["name"],
            "status"     : row["status"],
            "actual_up"  : int(row["y_up"]),
            "matches"    : int(row["n_matches_loaded"]),
            "probability": round(float(proba) * 100, 1),
            "confidence" : confidence_label(proba),
            "prediction" : "YES" if proba >= 0.5 else "NO",
            "correct"    : bool((proba >= 0.5) == bool(row["y_up"])),
            "top_for"    : top_for,
            "top_against": top_against,
            "norm_values": [round(float(v), 4) for v in norm],
        })

    return jsonify({
        "feature_names": [format_feature_name(c) for c in feature_cols],
        "keeper1"      : results[0],
        "keeper2"      : results[1],
    })


@app.route("/api/stats")
def stats():
    """Return dataset overview stats."""
    status_counts = up_df["status"].value_counts().to_dict()
    return jsonify({
        "total_keepers"   : len(up_df),
        "features_used"   : len(feature_cols),
        "positive_rate"   : round(float(y_up.mean()) * 100, 1),
        "status_counts"   : status_counts,
        "predicted_up"    : int((all_probas >= 0.5).sum()),
        "high_confidence" : int((all_probas >= 0.70).sum()),
    })

if __name__ == "__main__":
    import sys
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 5000
    print(f"\n  OHL GK Scouting App")
    print(f"  Open: http://localhost:{port}\n")
    app.run(debug=False, port=port, use_reloader=False, threaded=True)
