#!/bin/bash

echo "🚀 Starting Movie Recommender MLOps Pipeline..."
echo ""

# ── Detect OS ─────────────────────────────────────────────────────
OS="$(uname -s)"
case "${OS}" in
    Linux*)     MACHINE=Linux;;
    Darwin*)    MACHINE=Mac;;
    MINGW*|MSYS*|CYGWIN*) MACHINE=Windows;;
    *)          MACHINE=Unknown;;
esac
echo "🖥️  Detected OS: $MACHINE"
echo ""

# ── Create virtual environment if it doesn't exist ───────────────
if [ ! -d ".venv" ]; then
    echo "📦 Virtual environment not found — creating one..."
    python3 -m venv .venv
    echo "✅ Virtual environment created"
fi

# ── Activate virtual environment ──────────────────────────────────
if [ "$MACHINE" = "Windows" ]; then
    source .venv/Scripts/activate
else
    source .venv/bin/activate
fi
echo "✅ Virtual environment activated"

# ── Activate virtual environment ──────────────────────────────────
if [ "$MACHINE" = "Windows" ]; then
    source .venv/Scripts/activate
else
    source .venv/bin/activate
fi
echo "✅ Virtual environment activated"

# ── Check dependencies ────────────────────────────────────────────
echo "🔍 Checking dependencies..."
python -c "import mlflow, fastapi, sklearn, pandas, evidently, streamlit" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "📦 Installing missing dependencies..."
    pip install -r requirements.txt
fi
echo "✅ Dependencies ready"
echo ""

# ── Check dataset ─────────────────────────────────────────────────
if [ ! -f "data/raw/ml-100k/u.data" ]; then
    echo "❌ MovieLens dataset not found!"
    echo "👉 Download from: https://grouplens.org/datasets/movielens/100k/"
    echo "👉 Unzip into: data/raw/ml-100k/"
    exit 1
fi
echo "✅ Dataset found"

# ── Start MLflow server ───────────────────────────────────────────
echo "📊 Starting MLflow server on port 5000..."
if [ "$MACHINE" = "Mac" ]; then
    osascript -e 'tell app "Terminal" to do script "cd '"$(pwd)"' && source .venv/bin/activate && mlflow server --host 0.0.0.0 --port 5000"'
elif [ "$MACHINE" = "Windows" ]; then
    cmd /c start "MLflow Server" cmd /k "cd $(pwd) && source .venv/Scripts/activate && mlflow server --host 0.0.0.0 --port 5000"
else
    # Linux — run in background
    mlflow server --host 0.0.0.0 --port 5000 > logs/mlflow.log 2>&1 &
    echo "   MLflow PID: $!"
fi

# ── Wait for MLflow ───────────────────────────────────────────────
echo "⏳ Waiting for MLflow to start..."
for i in {1..10}; do
    sleep 2
    curl -s http://localhost:5000/health > /dev/null 2>&1
    if [ $? -eq 0 ]; then
        echo "✅ MLflow is up!"
        break
    fi
    echo "   Waiting... ($i/10)"
done

# ── Train model ───────────────────────────────────────────────────
echo ""
echo "🤖 Training model..."
python train.py
if [ $? -ne 0 ]; then
    echo "❌ Training failed! Check logs above."
    exit 1
fi
echo "✅ Model trained successfully"

# ── Run drift simulation ──────────────────────────────────────────
echo ""
echo "📈 Running drift simulation..."
python drift_simulation.py
if [ $? -ne 0 ]; then
    echo "⚠️  Drift simulation failed — continuing anyway"
else
    echo "✅ Drift simulation complete"
fi

# ── Create logs directory ─────────────────────────────────────────
mkdir -p logs

# ── Start FastAPI ─────────────────────────────────────────────────
echo ""
echo "⚡ Starting FastAPI on port 8000..."
if [ "$MACHINE" = "Mac" ]; then
    osascript -e 'tell app "Terminal" to do script "cd '"$(pwd)"' && source .venv/bin/activate && uvicorn app.main:app --reload --port 8000"'
elif [ "$MACHINE" = "Windows" ]; then
    cmd /c start "FastAPI Server" cmd /k "cd $(pwd) && source .venv/Scripts/activate && uvicorn app.main:app --reload --port 8000"
else
    uvicorn app.main:app --host 0.0.0.0 --port 8000 > logs/fastapi.log 2>&1 &
    echo "   FastAPI PID: $!"
fi

# ── Wait for FastAPI ──────────────────────────────────────────────
echo "⏳ Waiting for FastAPI to start..."
for i in {1..10}; do
    sleep 2
    curl -s http://localhost:8000/health > /dev/null 2>&1
    if [ $? -eq 0 ]; then
        echo "✅ FastAPI is up!"
        break
    fi
    echo "   Waiting... ($i/10)"
done

# ── Start Streamlit ───────────────────────────────────────────────
echo ""
echo "🎨 Starting Streamlit on port 8501..."
if [ "$MACHINE" = "Mac" ]; then
    osascript -e 'tell app "Terminal" to do script "cd '"$(pwd)"' && source .venv/bin/activate && streamlit run streamlit_app.py"'
elif [ "$MACHINE" = "Windows" ]; then
    cmd /c start "Streamlit App" cmd /k "cd $(pwd) && source .venv/Scripts/activate && streamlit run streamlit_app.py"
else
    streamlit run streamlit_app.py > logs/streamlit.log 2>&1 &
    echo "   Streamlit PID: $!"
fi

# ── Done ──────────────────────────────────────────────────────────
echo ""
echo "════════════════════════════════════════"
echo "✅ Everything is running!"
echo "════════════════════════════════════════"
echo ""
echo "📊 MLflow    → http://localhost:5000"
echo "⚡ FastAPI   → http://localhost:8000/docs"
echo "🎬 Streamlit → http://localhost:8501"
echo ""
echo "📁 Logs (Linux) → logs/"
echo ""
echo "To stop all services:"
echo "  Windows/Mac → close the terminal windows"
echo "  Linux       → kill \$(lsof -ti:5000,8000,8501)"
echo "════════════════════════════════════════"
