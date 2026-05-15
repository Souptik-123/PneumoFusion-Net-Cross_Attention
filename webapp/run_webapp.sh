#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# run_webapp.sh  –  Start both FastAPI backend and Streamlit frontend locally
#
# Usage
# -----
#   chmod +x webapp/run_webapp.sh
#   ./webapp/run_webapp.sh
#
# Prerequisites
# -------------
#   pip install -r webapp/requirements_webapp.txt
#   Trained checkpoint must exist at outputs/checkpoints/fold0_finetuned.pt
#   (or set CKPT_PATH env var)
#
# OpenAI (optional)
# -----------------
#   export OPENAI_API_KEY=sk-...
#   (or enter it in the sidebar at runtime)
# ─────────────────────────────────────────────────────────────────────────────

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_ROOT"

echo "============================================================"
echo "  PneumoFusion-Net Web Application"
echo "============================================================"

# ── Check checkpoint ──────────────────────────────────────────────────────
CKPT="${CKPT_PATH:-outputs/checkpoints/fold2_finetuned.pt}"
if [ ! -f "$CKPT" ]; then
    echo ""
    echo "  WARNING: checkpoint not found at $CKPT"
    echo "  Set CKPT_PATH to your checkpoint, or train the model first:"
    echo "    python main.py"
    echo ""
fi

# ── Start FastAPI backend (background) ────────────────────────────────────
echo ""
echo "  Starting FastAPI backend on http://localhost:8000 ..."
uvicorn webapp.backend.api:app --host 0.0.0.0 --port 8000 --reload &
BACKEND_PID=$!
echo "  Backend PID: $BACKEND_PID"

# ── Wait for backend ──────────────────────────────────────────────────────
echo "  Waiting for backend to start..."
for i in $(seq 1 20); do
    if curl -s http://localhost:8000/health > /dev/null 2>&1; then
        echo "  Backend is ready."
        break
    fi
    sleep 1
done

# ── Start Streamlit frontend ──────────────────────────────────────────────
echo ""
echo "  Starting Streamlit frontend on http://localhost:8501 ..."
echo ""
echo "  API docs:  http://localhost:8000/docs"
echo "  UI:        http://localhost:8501"
echo ""
echo "  Press Ctrl+C to stop both services."
echo "============================================================"

trap "kill $BACKEND_PID 2>/dev/null; exit" SIGINT SIGTERM

streamlit run webapp/frontend/app.py \
    --server.port 8501 \
    --server.address 0.0.0.0 \
    --browser.gatherUsageStats false

wait $BACKEND_PID
