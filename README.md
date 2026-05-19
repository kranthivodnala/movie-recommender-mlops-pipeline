# 🎬 Movie Recommender MLOps Pipeline

A production-grade MLOps pipeline built on the MovieLens 100K dataset — covering experiment tracking, model serving, containerization, drift monitoring, and a 3-page visual dashboard. Built as part of a 20-hour MLOps mastery plan by a DevOps engineer transitioning into ML Platform Engineering.

---

## 🧠 How It Works — In Plain English

A user watches 10 movies and gives their ratings (1–5 stars).
The ML model studies that rating pattern and predicts scores
for every other movie they haven't seen yet.
It then returns the top 10 highest predicted ones as recommendations.

The model also looks at other users who rated those same 10 movies
similarly — and borrows their opinions on unseen movies.
That's why it's called Collaborative Filtering — it collaborates
across users, not just your own history.

Same core idea behind Netflix, Spotify, and YouTube recommendations.

---

## 🏗️ Architecture

```
MovieLens Data → DVC Versioning → MLflow Experiment Tracking
                                          ↓
                               Best Model (SVD, n_factors=50)
                                          ↓
                              FastAPI REST API (Docker Container)
                                          ↓
                            Streamlit 3-Page Visual Dashboard
                         ┌──────────┬──────────┬──────────────┐
                         │  Page 1  │  Page 2  │    Page 3    │
                         │  Recs    │  Drift   │ Model Health │
                         └──────────┴──────────┴──────────────┘
                                          ↓
                          Evidently AI Drift Detection
                          (auto-triggers retraining on drift)
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
| Frontend | Streamlit (3-page dashboard) |
| Drift Monitoring | Evidently AI 0.7.x |
| Statistical Testing | SciPy KS Test |
| Dataset | MovieLens 100K (100,000 ratings · 943 users · 1,682 movies) |

---

## 📁 Project Structure

```
movie-recommender-mlops-pipeline/
├── app/
│   └── main.py                   # FastAPI app — loads model, serves recommendations
├── data/
│   ├── raw/ml-100k/              # MovieLens raw data (DVC tracked)
│   └── processed/
│       └── user_item_matrix.csv  # Processed training matrix
├── models/
│   └── recommender.pkl           # Best trained model (SVD, n_factors=50)
├── logs/                         # Service logs (Linux/Mac)
├── train.py                      # Training script — 3 MLflow experiments
├── drift_simulation.py           # Drift detection — Evidently + KS Test
├── streamlit_app.py              # 3-page visual dashboard
├── Dockerfile                    # Container definition
├── requirements.txt              # Python dependencies
├── start.sh                      # One-command startup script (cross-platform)
├── .gitignore
└── README.md
```

---

## 🚀 Quick Start — One Command

The easiest way to run everything:

```bash
git clone https://github.com/kranthivodnala/movie-recommender-mlops-pipeline.git
cd movie-recommender-mlops-pipeline
chmod +x start.sh
./start.sh
```

The startup script will:
- Auto-detect your OS (Windows / Mac / Linux)
- Activate the virtual environment
- Check and install dependencies
- Validate the dataset exists
- Start MLflow, FastAPI, and Streamlit in separate windows
- Run training and drift simulation automatically
- Print all URLs when ready

---

## 🛠️ Manual Setup

### 1. Clone the repo
```bash
git clone https://github.com/kranthivodnala/movie-recommender-mlops-pipeline.git
cd movie-recommender-mlops-pipeline
```

### 2. Set up environment
```bash
python -m venv .venv
source .venv/Scripts/activate   # Windows Git Bash
source .venv/bin/activate       # Mac / Linux
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

### 6. Run drift simulation
```bash
python drift_simulation.py
```
Simulates production data drift and generates `drift_report.html` and `drift_score.json`.

### 7. Start the API
```bash
uvicorn app.main:app --reload --port 8000
```

### 8. Launch the Streamlit dashboard
```bash
streamlit run streamlit_app.py
```
Open [http://localhost:8501](http://localhost:8501) in your browser.

---

## 🐳 Run with Docker

Pull and run the pre-built image directly from DockerHub — no setup needed:

```bash
docker pull spidermanintegration/movie-recommender:v1
docker run -p 8000:8000 spidermanintegration/movie-recommender:v1
```

Then launch the Streamlit dashboard separately:
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

## 📊 Experiment Results

| Model | n_factors | RMSE | MAE |
|---|---|---|---|
| SVD_factors_50 ✅ | 50 | **2.8758** | **2.6073** |
| SVD_factors_100 | 100 | 3.1695 | 2.9154 |
| SVD_factors_150 | 150 | 3.3510 | 3.1156 |

Lower RMSE = smaller average prediction error = better model. n_factors=50 wins because fewer factors generalise better on unseen data — simpler models often outperform complex ones on small datasets.

---

## 🌊 Drift Monitoring

The pipeline monitors for **data drift** — detecting when real-world user behaviour shifts away from what the model was trained on.

### What is Drift?
> The model was trained on historical ratings. Over time, user tastes shift. Drift is the gap between what the model learned and what's happening now.

### How it's detected:
- **KS Test (Kolmogorov-Smirnov)** — statistical test comparing reference vs current rating distributions
- **Drift Share** — percentage of features that have drifted
- **Drift Score** — how different each feature's distribution is (0 = identical, 1 = completely different)

### Current drift results (simulated):
| Metric | Value |
|---|---|
| Drift Detected | 🚨 YES |
| Drift Share | 63% |
| Reference Avg Rating | 3.544 |
| Current Avg Rating | 2.003 |
| Rating Drop | -1.54 stars |

### Drift threshold:
- Threshold set at **30%** drift share
- Above threshold → 🚨 alert + retraining recommended
- Below threshold → ✅ model is healthy

---

## 🎨 Streamlit Dashboard — 3 Pages

| Page | What it shows |
|---|---|
| 🎬 Recommendations | Enter a User ID → get top-N personalised movie recommendations with titles, genres, predicted ratings, and a genre distribution chart |
| 📊 Drift Monitoring | Drift status banner, reference vs current distribution charts, drift threshold gauge, full Evidently HTML report embedded |
| ⚙️ Model Health | API + MLflow status, model info, pipeline component health checks, one-click retraining trigger |

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
| Drift detection | Evidently AI + SciPy | Detects when model needs retraining |
| Dashboard | Streamlit | 3-page visual monitoring and serving UI |
| Startup | start.sh | Cross-platform one-command pipeline launcher |

---

## 🔜 Coming Soon

- [ ] Prometheus + Grafana metrics dashboard
- [ ] GitHub Actions CI/CD pipeline (auto retrain on data drift)
- [ ] ECS deployment via Terraform
- [ ] Project 2 — Weather Drift Detector

---

## 👤 Author

**Kranthi Vodnala**
DevOps Engineer → MLOps Engineer
7+ years at Disney & NBCUniversal

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue)](https://www.linkedin.com/in/kranthi-v-355470157/)
[![DockerHub](https://img.shields.io/badge/DockerHub-Image-blue)](https://hub.docker.com/repository/docker/spidermanintegration/movie-recommender)