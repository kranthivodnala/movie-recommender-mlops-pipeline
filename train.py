import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import os
import pickle

# ── Load MovieLens data ──────────────────────────────────────────
cols = ["user_id", "item_id", "rating", "timestamp"]
df = pd.read_csv("data/raw/ml-100k/u.data", sep="\t", names=cols)
df = df.drop(columns=["timestamp"])

# ── Build user-item matrix ───────────────────────────────────────
matrix = df.pivot_table(index="user_id", columns="item_id", values="rating").fillna(0)

# ── Train/test split ─────────────────────────────────────────────
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
train_matrix = train_df.pivot_table(index="user_id", columns="item_id", values="rating").fillna(0)

# ── MLflow experiment ────────────────────────────────────────────
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("movie-recommender")

# Run 3 experiments with different n_factors
for n_factors in [50, 100, 150]:
    with mlflow.start_run(run_name=f"SVD_factors_{n_factors}"):

        # Train
        svd = TruncatedSVD(n_components=n_factors, random_state=42)
        svd.fit(train_matrix)

        # Reconstruct matrix & predict
        reconstructed = svd.inverse_transform(svd.transform(train_matrix))
        reconstructed_df = pd.DataFrame(
            reconstructed,
            index=train_matrix.index,
            columns=train_matrix.columns
        )

        # Evaluate on test set (only rows/cols that exist in train)
        test_subset = test_df[
            test_df["user_id"].isin(train_matrix.index) &
            test_df["item_id"].isin(train_matrix.columns)
        ]
        preds = test_subset.apply(
            lambda row: reconstructed_df.loc[row["user_id"], row["item_id"]]
            if row["item_id"] in reconstructed_df.columns else 0,
            axis=1
        )
        rmse = np.sqrt(mean_squared_error(test_subset["rating"], preds))
        mae  = np.mean(np.abs(test_subset["rating"] - preds))

        # Log to MLflow
        mlflow.log_param("n_factors", n_factors)
        mlflow.log_param("model_type", "TruncatedSVD")
        mlflow.log_metric("rmse", round(rmse, 4))
        mlflow.log_metric("mae",  round(mae,  4))

        # Save the training matrix
        train_matrix.to_csv("data/processed/user_item_matrix.csv")
        mlflow.log_artifact("data/processed/user_item_matrix.csv")

        # Log model to MLflow
        mlflow.sklearn.log_model(svd, "model")

        # ── Save best model to disk (inside loop!) ───────────────
        if n_factors == 50:
            os.makedirs("models", exist_ok=True)
            with open("models/recommender.pkl", "wb") as f:
                pickle.dump(svd, f)
            print("✅ Best model saved to models/recommender.pkl")

        print(f"n_factors={n_factors} | RMSE={rmse:.4f} | MAE={mae:.4f}")
        print(f"  Run ID: {mlflow.active_run().info.run_id}")