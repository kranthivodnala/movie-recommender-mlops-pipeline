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

# ── Create required directories ───────────────────────────────────
mkdir -p logs data/processed models
echo "✅ Directories ready"

# ── Install system dependencies (Amazon Linux) ────────────────────
if [ "$MACHINE" = "Linux" ]; then
    if ! command -v unzip &>/dev/null; then
        echo "📦 Installing unzip..."
        sudo yum install unzip -y 2>/dev/null || sudo apt-get install unzip -y 2>/dev/null
    fi
    if ! command -v curl &>/dev/null; then
        echo "📦 Installing curl..."
        sudo yum install curl -y 2>/dev/null || sudo apt-get install curl -y 2>/dev/null
    fi
fi

# ── Check and install Python dependencies ────────────────────────
echo "🔍 Checking dependencies..."
python3 -c "import mlflow, fastapi, sklearn, pandas, evidently, streamlit, scipy" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "📦 Installing dependencies..."
    pip3 install --ignore-installed requests
    pip3 install mlflow fastapi uvicorn scikit-learn pandas numpy boto3 streamlit scipy evidently
    if [ $? -ne 0 ]; then
        echo "❌ Dependency installation failed!"
        exit 1
    fi
fi
echo "✅ Dependencies ready"
echo ""

# ── Check and download dataset ────────────────────────────────────
if [ ! -f "data/raw/ml-100k/u.data" ]; then
    echo "📥 MovieLens dataset not found — downloading automatically..."
    mkdir -p data/raw

    curl -L -o ml-100k.zip http://files.grouplens.org/datasets/movielens/ml-100k.zip
    if [ $? -ne 0 ]; then
        echo "❌ Dataset download failed!"
        exit 1
    fi

    unzip -o ml-100k.zip -d data/raw/
    rm ml-100k.zip
    echo "✅ Dataset downloaded and extracted"
else
    echo "✅ Dataset found"
fi

# ── Verify dataset ────────────────────────────────────────────────
if [ ! -f "data/raw/ml-100k/u.data" ]; then
    echo "❌ u.data still missing after extraction!"
    echo "📁 Contents of data/raw/:"
    ls -la data/raw/
    ls -la data/raw/ml-100k/ 2>/dev/null || echo "   folder not found"
    exit 1
fi
echo "✅ Dataset verified"
echo ""

# ── Start MLflow server ───────────────────────────────────────────
echo "📊 Starting MLflow server on port 5000..."
if [ "$MACHINE" = "Mac" ]; then
    osascript -e 'tell app "Terminal" to do script "cd '"$(pwd)"' && mlflow server --host 0.0.0.0 --port 5000"'
elif [ "$MACHINE" = "Windows" ]; then
    cmd /c start "MLflow Server" cmd /k "cd $(pwd) && mlflow server --host 0.0.0.0 --port 5000"
else
    nohup python3 -m mlflow server --host 0.0.0.0 --port 5000 > logs/mlflow.log 2>&1 &
    MLFLOW_PID=$!
    echo "   MLflow PID: $MLFLOW_PID"
    echo $MLFLOW_PID > logs/mlflow.pid
fi

# ── Wait for MLflow to be ready ───────────────────────────────────
echo "⏳ Waiting for MLflow to start..."
for i in {1..15}; do
    sleep 2
    curl -s http://localhost:5000 > /dev/null 2>&1
    if [ $? -eq 0 ]; then
        echo "✅ MLflow is up!"
        break
    fi
    echo "   Waiting... ($i/15)"
    if [ $i -eq 15 ]; then
        echo "❌ MLflow failed to start!"
        echo "💡 Check logs/mlflow.log for details:"
        cat logs/mlflow.log
        exit 1
    fi
done
echo ""

# ── Train model ───────────────────────────────────────────────────
echo "🤖 Training model..."
python3 train.py
if [ $? -ne 0 ]; then
    echo "❌ Training failed!"
    exit 1
fi
echo "✅ Model trained successfully"
echo ""

# ── Run drift simulation ──────────────────────────────────────────
echo "📈 Running drift simulation..."
python3 drift_simulation.py
if [ $? -ne 0 ]; then
    echo "⚠️  Drift simulation failed — continuing anyway"
else
    echo "✅ Drift simulation complete"
fi
echo ""

# ── Start FastAPI ─────────────────────────────────────────────────
echo "⚡ Starting FastAPI on port 8000..."
if [ "$MACHINE" = "Mac" ]; then
    osascript -e 'tell app "Terminal" to do script "cd '"$(pwd)"' && uvicorn app.main:app --reload --port 8000"'
elif [ "$MACHINE" = "Windows" ]; then
    cmd /c start "FastAPI Server" cmd /k "cd $(pwd) && uvicorn app.main:app --reload --port 8000"
else
    nohup python3 -m uvicorn app.main:app --host 0.0.0.0 --port 8000 > logs/fastapi.log 2>&1 &
    FASTAPI_PID=$!
    echo "   FastAPI PID: $FASTAPI_PID"
    echo $FASTAPI_PID > logs/fastapi.pid
fi

# ── Wait for FastAPI to be ready ──────────────────────────────────
echo "⏳ Waiting for FastAPI to start..."
for i in {1..15}; do
    sleep 2
    curl -s http://localhost:8000/health > /dev/null 2>&1
    if [ $? -eq 0 ]; then
        echo "✅ FastAPI is up!"
        break
    fi
    echo "   Waiting... ($i/15)"
    if [ $i -eq 15 ]; then
        echo "❌ FastAPI failed to start!"
        echo "💡 Check logs/fastapi.log for details:"
        cat logs/fastapi.log
        exit 1
    fi
done
echo ""

# ── Start Streamlit ───────────────────────────────────────────────
echo "🎨 Starting Streamlit on port 8501..."
if [ "$MACHINE" = "Mac" ]; then
    osascript -e 'tell app "Terminal" to do script "cd '"$(pwd)"' && streamlit run streamlit_app.py"'
elif [ "$MACHINE" = "Windows" ]; then
    cmd /c start "Streamlit App" cmd /k "cd $(pwd) && streamlit run streamlit_app.py"
else
    nohup python3 -m streamlit run streamlit_app.py \
        --server.port 8501 \
        --server.address 0.0.0.0 \
        > logs/streamlit.log 2>&1 &
    STREAMLIT_PID=$!
    echo "   Streamlit PID: $STREAMLIT_PID"
    echo $STREAMLIT_PID > logs/streamlit.pid
fi
echo ""

# ── Wait for Streamlit ────────────────────────────────────────────
echo "⏳ Waiting for Streamlit to start..."
for i in {1..15}; do
    sleep 2
    curl -s http://localhost:8501 > /dev/null 2>&1
    if [ $? -eq 0 ]; then
        echo "✅ Streamlit is up!"
        break
    fi
    echo "   Waiting... ($i/15)"
done
echo ""

# ── Done ──────────────────────────────────────────────────────────
echo "════════════════════════════════════════"
echo "✅ Everything is running!"
echo "════════════════════════════════════════"
echo ""
echo "📊 MLflow    → http://localhost:5000"
echo "⚡ FastAPI   → http://localhost:8000/docs"
echo "🎬 Streamlit → http://localhost:8501"
echo ""
if [ "$MACHINE" = "Linux" ]; then
    echo "📁 Logs → logs/"
    echo "   mlflow.log | fastapi.log | streamlit.log"
    echo ""
    echo "🛑 To stop all services:"
    echo "   kill \$(cat logs/mlflow.pid logs/fastapi.pid logs/streamlit.pid)"
fi
echo "════════════════════════════════════════"
