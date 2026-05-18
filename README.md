# 🎬 Movie Recommender MLOps Pipeline

A MLOps pipeline built on the MovieLens 100K dataset — covering experiment tracking, model serving, containerization, and a visual frontend. Built as part of a 20-hour MLOps mastery plan by a DevOps engineer transitioning into ML Platform Engineering.

---

## 🏗️ Architecture

```
MovieLens Data → DVC Versioning → MLflow Experiment Tracking
                                          ↓
                               Best Model (SVD, n_factors=50)
                                          ↓
                              FastAPI REST API (Docker Container)
                                          ↓
                            Streamlit Visual Frontend (localhost:8501)
```

---

## ⚡ Tech Stack

| Layer | Technology |
|---|---|
| Data Versioning | DVC |
| Experiment Tracking | MLflow 3.9 |
| ML Model | TruncatedSVD (scikit-learn) |
| REST API | FastAPI + Uvicorn |
| Containerization | Docker |
| Container Registry | DockerHub |
| Frontend | Streamlit |
| Dataset | MovieLens 100K (100,000 ratings · 943 users · 1,682 movies) |

---

## 📁 Project Structure

```
movie-recommender-mlops-pipeline/
├── app/
│   └── main.py               # FastAPI app — loads model, serves recommendations
├── data/
│   ├── raw/ml-100k/          # MovieLens raw data (DVC tracked)
│   └── processed/
│       └── user_item_matrix.csv  # Processed training matrix
├── models/
│   └── recommender.pkl       # Best trained model (SVD, n_factors=50)
├── train.py                  # Training script — 3 MLflow experiments
├── streamlit_app.py          # Visual frontend
├── Dockerfile                # Container definition
├── requirements.txt          # Python dependencies
├── .gitignore
└── README.md
```

---

## 🚀 Quick Start

### 1. Clone the repo
```bash
git clone https://github.com/kranthivodnala/movie-recommender-mlops-pipeline.git
cd movie-recommender-mlops-pipeline
```

### 2. Set up environment
```bash
python -m venv .venv
source .venv/Scripts/activate  # Windows Git Bash
pip install -r requirements.txt
```

### 3. Download MovieLens 100K dataset
Download from [grouplens.org/datasets/movielens/100k](https://grouplens.org/datasets/movielens/100k) and unzip into `data/raw/ml-100k/`

### 4. Start MLflow server
```bash
mlflow server --host 0.0.0.0 --port 5000
```

### 5. Train the model
```bash
python train.py
```
Runs 3 experiments with n_factors = 50, 100, 150. Best model (n_factors=50, RMSE=2.87) saved to `models/recommender.pkl`.

### 6. Start the API
```bash
uvicorn app.main:app --reload --port 8000
```

### 7. Launch the Streamlit frontend
```bash
streamlit run streamlit_app.py
```
Open [http://localhost:8501](http://localhost:8501) in your browser.

---

## 🐳 Run with Docker

Pull and run the pre-built image directly from DockerHub — no setup needed:

```bash
docker pull <your-dockerhub-username>/movie-recommender:v1
docker run -p 8000:8000 <your-dockerhub-username>/movie-recommender:v1
```

Then launch the Streamlit frontend separately:
```bash
streamlit run streamlit_app.py
```

---

## 🔌 API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/` | API status |
| GET | `/health` | Health check |
| GET | `/recommend/{user_id}` | Top-N movie recommendations for a user |

### Example Response
```json
{
  "user_id": 42,
  "recommendations": [
    {
      "movie_id": 69,
      "title": "Forrest Gump (1994)",
      "genres": "Comedy | Drama | Romance | War",
      "predicted_rating": 3.76
    }
  ]
}
```

Query params: `top_n` (default: 10, max: 20)
```
GET /recommend/42?top_n=15
```

---

## 🧠 How It Works

This pipeline uses **Collaborative Filtering** via Singular Value Decomposition (SVD):

1. Build a **user-item matrix** — 943 users × 1,653 movies, filled with real ratings (0 = not watched)
2. **Decompose** the matrix using TruncatedSVD to find hidden taste patterns (latent factors)
3. **Reconstruct** the matrix — filling in predicted ratings for every unwatched movie
4. For any user, return the **top-N highest predicted ratings** for movies they haven't seen yet

The model learns that users with similar taste histories will likely enjoy similar unwatched movies — the same core idea behind Netflix, Spotify, and YouTube recommendations.

---

## 📊 Experiment Results

| Model | n_factors | RMSE | MAE |
|---|---|---|---|
| SVD_factors_50 ✅ | 50 | **2.8758** | **2.6073** |
| SVD_factors_100 | 100 | 3.1695 | 2.9154 |
| SVD_factors_150 | 150 | 3.3510 | 3.1156 |

Lower RMSE = smaller average prediction error = better model. n_factors=50 wins because fewer factors generalise better on unseen data — simpler models often outperform complex ones on small datasets.

---

## 🗺️ MLOps Pipeline Stages

| Stage | Tool | What it does |
|---|---|---|
| Data versioning | DVC | Tracks dataset versions alongside code in Git |
| Experiment tracking | MLflow | Logs params, metrics, and artifacts for all 3 runs |
| Model registry | MLflow Registry | Registers best model with `production` alias |
| Model serving | FastAPI | REST API loads model from disk, serves predictions |
| Containerization | Docker | Packages API + model into a portable image |
| Registry | DockerHub | Publishes image publicly for anyone to pull |
| Frontend | Streamlit | Visual UI with movie cards, genre badges, rating bars |

---

## 🔜 Coming Soon

- [ ] Phase 3 — Drift monitoring with Evidently AI
- [ ] Grafana dashboard for prediction latency + drift score
- [ ] GitHub Actions CI/CD pipeline (auto retrain on data drift)
- [ ] ECS deployment via Terraform

---

## 👤 Author

**Kranthi Vodnala**
DevOps Engineer → MLOps Engineer

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue)](https://www.linkedin.com/in/kranthi-v-355470157/)
[![DockerHub](https://img.shields.io/badge/DockerHub-Image-blue)](https://hub.docker.com/repository/docker/spidermanintegration/movie-recommender)