import pandas as pd
import numpy as np
import json
from evidently import Dataset, DataDefinition
from evidently.presets import DataDriftPreset
from evidently import Report
from scipy import stats

# ── Load raw ratings ─────────────────────────────────────────────
print("Loading data...")
cols = ["user_id", "item_id", "rating", "timestamp"]
df = pd.read_csv("data/raw/ml-100k/u.data", sep="\t", names=cols)
df = df.drop(columns=["timestamp"])

# ── Reference = first 5000 ratings (training era) ────────────────
reference = df.head(5000)[["user_id", "item_id", "rating"]].reset_index(drop=True)

# ── Current = last 5000 ratings with simulated drift ─────────────
np.random.seed(42)
current = df.tail(5000)[["user_id", "item_id", "rating"]].copy().reset_index(drop=True)
# Stronger drift — simulate users rating much lower over time
current["rating"] = current["rating"] - 1.8
current["rating"] = current["rating"] + np.random.normal(0, 0.8, len(current))
current["rating"] = current["rating"].clip(1, 5).round(1)

# Also shift item_id distribution to simulate new movies being watched
current["item_id"] = current["item_id"] + np.random.randint(50, 200, len(current))
current["item_id"] = current["item_id"].clip(1, 1682)

print(f"Reference data: {len(reference)} rows | avg rating: {reference['rating'].mean():.2f}")
print(f"Current data:   {len(current)} rows | avg rating: {current['rating'].mean():.2f}")

# ── Define data schema ───────────────────────────────────────────
definition = DataDefinition(
    numerical_columns=["user_id", "item_id", "rating"],
)

reference_dataset = Dataset.from_pandas(reference, data_definition=definition)
current_dataset   = Dataset.from_pandas(current,   data_definition=definition)

# ── Generate drift report ────────────────────────────────────────
print("Generating drift report...")
report = Report([DataDriftPreset()])
my_eval = report.run(reference_dataset, current_dataset)

# Save HTML
my_eval.save_html("drift_report.html")
print("✅ Drift report saved to drift_report.html")

# ── Extract drift score ──────────────────────────────────────────
# Calculate drift manually from the data since Evidently 0.7.x
# changed their dict structure

ref_ratings = reference["rating"].values
cur_ratings = current["rating"].values

# KS test — same test Evidently uses internally
ks_stat, ks_pvalue = stats.ks_2samp(ref_ratings, cur_ratings)

# KS stat > 0.1 and p-value < 0.05 = drift detected
drift_detected = bool(ks_stat > 0.1 and ks_pvalue < 0.05)
drift_share = round(float(ks_stat), 4)

print(f"📊 KS Statistic:   {ks_stat:.4f}")
print(f"📊 P-Value:        {ks_pvalue:.6f}")
print(f"📊 Drift detected: {drift_detected}")
print(f"📊 Drift share:    {drift_share:.2%}")

# Save drift score for Streamlit
with open("drift_score.json", "w") as f:
    json.dump({
        "drift_detected": bool(drift_detected),
        "drift_share": round(float(drift_share), 4),
        "reference_avg_rating": round(reference["rating"].mean(), 3),
        "current_avg_rating": round(current["rating"].mean(), 3),
    }, f, indent=2)
print("✅ Drift score saved to drift_score.json")