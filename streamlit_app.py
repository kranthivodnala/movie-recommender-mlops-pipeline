import streamlit as st
import requests
import pandas as pd
import json
import os

# ── Page config ──────────────────────────────────────────────────
st.set_page_config(
    page_title="🎬 Movie Recommender MLOps",
    page_icon="🎬",
    layout="wide"
)

# ── Navigation ───────────────────────────────────────────────────
page = st.sidebar.radio(
    "Navigation",
    ["🎬 Recommendations", "📊 Drift Monitoring", "⚙️ Model Health"]
)

# ════════════════════════════════════════════════════════════════
# PAGE 1 — Recommendations
# ════════════════════════════════════════════════════════════════
if page == "🎬 Recommendations":
    st.title("🎬 Movie Recommender")
    st.markdown("*Powered by SVD Collaborative Filtering · MLOps Project by Kranthi*")
    st.divider()

    with st.sidebar:
        st.header("⚙️ Settings")
        api_url = st.text_input("API URL", value="http://localhost:8000")
        user_id = st.number_input("User ID", min_value=1, max_value=943, value=42, step=1)
        top_n = st.slider("Number of Recommendations", min_value=5, max_value=20, value=10)
        get_recs = st.button("🎯 Get Recommendations", use_container_width=True)

        st.divider()
        st.markdown("**About**")
        st.markdown("Built with MovieLens 100K dataset — 100,000 real ratings from 943 users across 1,682 movies.")

        try:
            health = requests.get(f"{api_url}/health", timeout=3)
            if health.status_code == 200:
                st.success("✅ API Connected")
            else:
                st.error("❌ API Error")
        except:
            st.error("❌ API Unreachable")

    if get_recs:
        with st.spinner(f"Finding top {top_n} movies for User {user_id}..."):
            try:
                response = requests.get(
                    f"{api_url}/recommend/{user_id}",
                    params={"top_n": top_n},
                    timeout=10
                )
                if response.status_code == 200:
                    data = response.json()
                    recs = data["recommendations"]

                    col1, col2, col3 = st.columns(3)
                    col1.metric("User ID", user_id)
                    col2.metric("Recommendations", len(recs))
                    col3.metric("Top Predicted Rating", f"{recs[0]['predicted_rating']} ⭐")
                    st.divider()

                    cols = st.columns(2)
                    for i, rec in enumerate(recs):
                        with cols[i % 2]:
                            with st.container(border=True):
                                st.markdown(f"### {i+1}. {rec['title']}")
                                rating = rec['predicted_rating']
                                st.progress(
                                    min(rating / 5.0, 1.0),
                                    text=f"⭐ Predicted Rating: {rating}"
                                )
                                genres = rec['genres'].split(" | ")
                                genre_html = " ".join([
                                    f'<span style="background:#1f4e79;color:white;padding:3px 10px;border-radius:12px;font-size:12px;margin:2px;display:inline-block">{g}</span>'
                                    for g in genres
                                ])
                                st.markdown(genre_html, unsafe_allow_html=True)
                                st.caption(f"Movie ID: {rec['movie_id']}")

                    st.divider()
                    st.subheader("📊 Genre Distribution")
                    all_genres = []
                    for rec in recs:
                        all_genres.extend(rec['genres'].split(" | "))
                    genre_counts = pd.Series(all_genres).value_counts()
                    st.bar_chart(genre_counts)

            except requests.exceptions.ConnectionError:
                st.error("Cannot connect to FastAPI. Make sure it's running on port 8000!")
            except Exception as e:
                st.error(f"Something went wrong: {e}")
    else:
        st.info("👈 Enter a User ID in the sidebar and click **Get Recommendations** to start!")
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Users", "943")
        col2.metric("Total Movies", "1,682")
        col3.metric("Total Ratings", "100,000")

# ════════════════════════════════════════════════════════════════
# PAGE 2 — Drift Monitoring
# ════════════════════════════════════════════════════════════════
elif page == "📊 Drift Monitoring":
    st.title("📊 Drift Monitoring Dashboard")
    st.markdown("*Detecting when the model needs retraining*")
    st.divider()

    # ── Load drift score ─────────────────────────────────────────
    if os.path.exists("drift_score.json"):
        with open("drift_score.json") as f:
            drift = json.load(f)

        # ── Status banner ─────────────────────────────────────────
        if drift["drift_detected"]:
            st.error("🚨 DRIFT DETECTED — Model retraining recommended!")
        else:
            st.success("✅ No Drift Detected — Model is healthy")

        st.divider()

        # ── Metrics row ───────────────────────────────────────────
        col1, col2, col3, col4 = st.columns(4)
        col1.metric(
            "Drift Detected",
            "🚨 YES" if drift["drift_detected"] else "✅ NO"
        )
        col2.metric(
            "Drift Share",
            f"{drift['drift_share']:.0%}",
            delta=f"{drift['drift_share']:.0%} of features drifted",
            delta_color="inverse"
        )
        col3.metric(
            "Reference Avg Rating",
            drift["reference_avg_rating"],
            help="Average rating during training"
        )
        col4.metric(
            "Current Avg Rating",
            drift["current_avg_rating"],
            delta=f"{drift['current_avg_rating'] - drift['reference_avg_rating']:.2f}",
            delta_color="inverse"
        )

        st.divider()

        # ── Rating distribution comparison ────────────────────────
        st.subheader("📈 Rating Distribution — Reference vs Current")

        cols = ["user_id", "item_id", "rating", "timestamp"]
        df = pd.read_csv("data/raw/ml-100k/u.data", sep="\t", names=cols)
        df = df.drop(columns=["timestamp"])

        import numpy as np
        reference = df.head(5000)[["rating"]]
        current = df.tail(5000)[["rating"]].copy()
        current["rating"] = current["rating"] - 1.8
        current["rating"] = current["rating"] + np.random.normal(0, 0.8, len(current))
        current["rating"] = current["rating"].clip(1, 5).round(1)

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**📚 Reference Data (Training Era)**")
            ref_counts = reference["rating"].value_counts().sort_index()
            st.bar_chart(ref_counts)

        with col2:
            st.markdown("**⚠️ Current Data (Production)**")
            cur_counts = current["rating"].value_counts().sort_index()
            st.bar_chart(cur_counts)

        st.divider()

        # ── Drift threshold gauge ─────────────────────────────────
        st.subheader("🎚️ Drift Threshold Monitor")
        threshold = 0.3
        drift_val = drift["drift_share"]

        st.markdown(f"**Threshold:** {threshold:.0%} | **Current:** {drift_val:.0%}")
        st.progress(min(drift_val, 1.0))

        if drift_val > threshold:
            st.warning(f"⚠️ Drift share {drift_val:.0%} exceeds threshold {threshold:.0%} — trigger retraining!")
        else:
            st.success(f"✅ Drift share {drift_val:.0%} is within threshold {threshold:.0%}")

        st.divider()

        # ── Evidently full report ─────────────────────────────────
        st.subheader("📋 Full Evidently Report")
        if os.path.exists("drift_report.html"):
            with open("drift_report.html", "r", encoding="utf-8") as f:
                html_content = f.read()
            st.components.v1.html(html_content, height=600, scrolling=True)
        else:
            st.warning("Run drift_simulation.py first to generate the report")

    else:
        st.warning("No drift data found. Run drift_simulation.py first!")
        st.code("python drift_simulation.py")

# ════════════════════════════════════════════════════════════════
# PAGE 3 — Model Health
# ════════════════════════════════════════════════════════════════
elif page == "⚙️ Model Health":
    st.title("⚙️ Model Health Dashboard")
    st.markdown("*Live system status for the Movie Recommender pipeline*")
    st.divider()

    api_url = "http://localhost:8000"

    # ── API Health ────────────────────────────────────────────────
    st.subheader("🔌 API Status")
    col1, col2 = st.columns(2)

    try:
        health = requests.get(f"{api_url}/health", timeout=3)
        col1.success("✅ FastAPI — Online")
        col1.metric("Status Code", health.status_code)
    except:
        col1.error("❌ FastAPI — Offline")

    try:
        import mlflow
        mlflow.set_tracking_uri("http://localhost:5000")
        client = mlflow.MlflowClient()
        experiments = client.search_experiments()
        col2.success("✅ MLflow — Online")
        col2.metric("Experiments", len(experiments))
    except:
        col2.error("❌ MLflow — Offline")

    st.divider()

    # ── Model info ────────────────────────────────────────────────
    st.subheader("🤖 Model Info")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Model Type", "TruncatedSVD")
    col2.metric("n_factors", "50")
    col3.metric("RMSE", "2.8758")
    col4.metric("Training Rows", "80,000")

    st.divider()

    # ── Drift summary ─────────────────────────────────────────────
    st.subheader("🌊 Latest Drift Summary")
    if os.path.exists("drift_score.json"):
        with open("drift_score.json") as f:
            drift = json.load(f)

        col1, col2 = st.columns(2)
        col1.metric("Drift Detected", "🚨 YES" if drift["drift_detected"] else "✅ NO")
        col2.metric("Drift Share", f"{drift['drift_share']:.0%}")

        if drift["drift_detected"]:
            st.error("🚨 Action Required — Retrain the model!")
            if st.button("🔄 Trigger Retraining Now"):
                import subprocess
                with st.spinner("Retraining model..."):
                    result = subprocess.run(
                        ["python", "train.py"],
                        capture_output=True, text=True
                    )
                    if result.returncode == 0:
                        st.success("✅ Retraining complete!")
                        st.code(result.stdout)
                    else:
                        st.error("❌ Retraining failed!")
                        st.code(result.stderr)
        else:
            st.success("✅ Model is healthy — no action needed")
    else:
        st.info("Run drift_simulation.py to see drift summary")

    st.divider()

    # ── Pipeline status ───────────────────────────────────────────
    st.subheader("🏗️ Pipeline Components")
    components = {
        "📦 Training Data": os.path.exists("data/raw/ml-100k/u.data"),
        "🔢 User-Item Matrix": os.path.exists("data/processed/user_item_matrix.csv"),
        "🤖 Trained Model": os.path.exists("models/recommender.pkl"),
        "📊 Drift Report": os.path.exists("drift_report.html"),
        "📈 Drift Score": os.path.exists("drift_score.json"),
    }
    for component, exists in components.items():
        if exists:
            st.success(f"✅ {component} — Found")
        else:
            st.error(f"❌ {component} — Missing")