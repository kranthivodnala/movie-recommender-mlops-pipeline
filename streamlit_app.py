import streamlit as st
import requests
import pandas as pd

# ── Page config ──────────────────────────────────────────────────
st.set_page_config(
    page_title="🎬 Movie Recommender",
    page_icon="🎬",
    layout="wide"
)

# ── Header ───────────────────────────────────────────────────────
st.title("🎬 Movie Recommender")
st.markdown("*Powered by SVD Collaborative Filtering · MLOps Project by Kranthi*")
st.divider()

# ── Sidebar ──────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Settings")
    api_url = st.text_input("API URL", value="http://localhost:8000")
    user_id = st.number_input("User ID", min_value=1, max_value=943, value=42, step=1)
    top_n = st.slider("Number of Recommendations", min_value=5, max_value=20, value=10)
    get_recs = st.button("🎯 Get Recommendations", use_container_width=True)

    st.divider()
    st.markdown("**About**")
    st.markdown("Built with MovieLens 100K dataset — 100,000 real ratings from 943 users across 1,682 movies.")

# ── Health check ─────────────────────────────────────────────────
try:
    health = requests.get(f"{api_url}/health", timeout=3)
    if health.status_code == 200:
        st.sidebar.success("✅ API Connected")
    else:
        st.sidebar.error("❌ API Error")
except:
    st.sidebar.error("❌ API Unreachable — is FastAPI running?")

# ── Main content ─────────────────────────────────────────────────
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

                # ── Metrics row ──────────────────────────────────
                st.subheader(f"Top {top_n} Recommendations for User {user_id}")
                col1, col2, col3 = st.columns(3)
                col1.metric("User ID", user_id)
                col2.metric("Recommendations", len(recs))
                col3.metric("Top Predicted Rating", f"{recs[0]['predicted_rating']} ⭐")
                st.divider()

                # ── Movie cards grid ─────────────────────────────
                cols = st.columns(2)
                for i, rec in enumerate(recs):
                    with cols[i % 2]:
                        with st.container(border=True):
                            st.markdown(f"### {i+1}. {rec['title']}")
                            
                            # Rating bar
                            rating = rec['predicted_rating']
                            max_rating = 5.0
                            st.progress(
                                min(rating / max_rating, 1.0),
                                text=f"⭐ Predicted Rating: {rating}"
                            )

                            # Genres as badges
                            genres = rec['genres'].split(" | ")
                            genre_html = " ".join([
                                f'<span style="background:#1f4e79;color:white;padding:3px 10px;border-radius:12px;font-size:12px;margin:2px;display:inline-block">{g}</span>'
                                for g in genres
                            ])
                            st.markdown(genre_html, unsafe_allow_html=True)
                            st.caption(f"Movie ID: {rec['movie_id']}")

                st.divider()

                # ── Genre distribution chart ─────────────────────
                st.subheader("📊 Genre Distribution in Your Recommendations")
                all_genres = []
                for rec in recs:
                    all_genres.extend(rec['genres'].split(" | "))
                genre_counts = pd.Series(all_genres).value_counts()
                st.bar_chart(genre_counts)

            else:
                st.error(f"API returned error: {response.status_code}")

        except requests.exceptions.ConnectionError:
            st.error("Cannot connect to FastAPI. Make sure it's running on port 8000!")
        except Exception as e:
            st.error(f"Something went wrong: {e}")

else:
    # ── Default empty state ──────────────────────────────────────
    st.info("👈 Enter a User ID in the sidebar and click **Get Recommendations** to start!")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Users", "943")
    with col2:
        st.metric("Total Movies", "1,682")
    with col3:
        st.metric("Total Ratings", "100,000")