#!/usr/bin/env bash
# =====================================================
# Spatial-MLLM Environment Bootstrap (Cluster-Safe + Version-Locked)
# Author: Ruiqi
#
# NOTE:
# - This is a *reference* setup script for a specific Linux + CUDA 12.4 environment.
# - Versions are strictly pinned for reproducibility. If your system differs, it may fail by design.
# - You should refer to the official Spatial-MLLM repository for canonical instructions.
# =====================================================

set -euo pipefail
trap 'echo "❌ Failed at line $LINENO"; exit 1' ERR

echo "🚀 Setting up Spatial-MLLM environment (reference, version-locked)"

# -----------------------------------------------------
# Hard prerequisites (fail fast)
# -----------------------------------------------------
if [[ "$(uname -s)" != "Linux" ]]; then
  echo "❌ This reference script is Linux-only."
  exit 1
fi
if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "❌ nvidia-smi not found. CUDA GPU environment required."
  exit 1
fi
nvidia-smi >/dev/null 2>&1 || { echo "❌ nvidia-smi failed; driver/GPU not ready."; exit 1; }

# -----------------------------------------------------
# Base paths (override-friendly; defaults use $HOME so any user can run this)
# -----------------------------------------------------
SCR_ROOT="${SCR_ROOT:-$HOME}"
CONDA_DIR="${CONDA_DIR:-$HOME/anaconda3}"
INSTALLER="${INSTALLER:-Anaconda3-2025.06-0-Linux-x86_64.sh}"
REPO_DIR="${REPO_DIR:-$HOME/Spatial-MLLM}"

PIP_CACHE_DIR="${PIP_CACHE_DIR:-$HOME/.cache/pip}"
TMPDIR="${TMPDIR:-/tmp}"

# HF caches (defaults to standard ~/.cache/huggingface paths; override as needed)
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HOME/.cache/huggingface/transformers}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HOME/.cache/huggingface/datasets}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HOME/.cache/huggingface/hub}"

mkdir -p "$PIP_CACHE_DIR" "$TMPDIR" "$TRANSFORMERS_CACHE" "$HF_HOME" "$HF_DATASETS_CACHE" "$HF_HUB_CACHE"

echo "✅ PIP_CACHE_DIR → $PIP_CACHE_DIR"
echo "✅ TMPDIR       → $TMPDIR"
echo "✅ HF_HOME      → $HF_HOME"
echo "✅ HF caches    → $TRANSFORMERS_CACHE"

# -----------------------------------------------------
# Step 1. Move to SCR_ROOT and prepare base paths
# -----------------------------------------------------
cd "$SCR_ROOT"

# -----------------------------------------------------
# Step 2. Install Anaconda if needed
# -----------------------------------------------------
if [ ! -f "$INSTALLER" ]; then
  echo "⬇️ Downloading Anaconda installer..."
  wget "https://repo.anaconda.com/archive/${INSTALLER}"
fi

if [ ! -d "$CONDA_DIR" ]; then
  echo "📦 Installing Anaconda to $CONDA_DIR..."
  bash "$INSTALLER" -b -p "${CONDA_DIR}"
fi

# -----------------------------------------------------
# Step 3. Initialize conda (non-invasive; DO NOT conda init)
# -----------------------------------------------------
export PATH="${CONDA_DIR}/bin:${PATH}"
# shellcheck disable=SC1091
source "${CONDA_DIR}/etc/profile.d/conda.sh"

# -----------------------------------------------------
# Step 4. Create / Activate environment
# -----------------------------------------------------
if ! conda info --envs | grep -q "spatial-mllm"; then
  echo "🧩 Creating conda environment: spatial-mllm"
  conda create -n spatial-mllm python=3.10 -y
fi
conda activate spatial-mllm

# -----------------------------------------------------
# Step 5. Redirect pip caches/temp dirs
# -----------------------------------------------------
export PIP_CACHE_DIR="$PIP_CACHE_DIR"
export TMPDIR="$TMPDIR"

# -----------------------------------------------------
# Step 6. Clone Spatial-MLLM repo
# -----------------------------------------------------
cd "$SCR_ROOT"
if [ ! -d "$REPO_DIR" ]; then
  echo "📂 Cloning Spatial-MLLM repository..."
  git clone https://github.com/diankun-wu/Spatial-MLLM "$REPO_DIR"
fi
cd "$REPO_DIR"

# -----------------------------------------------------
# Step 7. Install dependencies (version locked)
# -----------------------------------------------------
echo "📦 Installing core dependencies (version-locked)..."
pip install --upgrade pip setuptools wheel

# Torch + CUDA (STRICT: cu124)
TMPDIR="$TMPDIR" PIP_CACHE_DIR="$PIP_CACHE_DIR" \
  pip install torch==2.6.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# Core packages (STRICT versions where you pinned them)
TMPDIR="$TMPDIR" PIP_CACHE_DIR="$PIP_CACHE_DIR" pip install \
  transformers==4.51.3 \
  accelerate==1.5.2 \
  qwen_vl_utils \
  decord \
  ray \
  Levenshtein \
  tyro \
  einops \
  timm \
  av

# flash-attn (NOTE: if you want strict reproducibility, pin a version here)
TMPDIR="$TMPDIR" PIP_CACHE_DIR="$PIP_CACHE_DIR" pip install flash-attn --no-build-isolation

# -----------------------------------------------------
# Step 8. Sanity check
# -----------------------------------------------------
echo ""
echo "🔍 Verifying key packages..."
python - <<'EOF'
import torch, transformers, accelerate, decord, Levenshtein, tyro, einops, timm, av
print("✅ torch:", torch.__version__)
print("✅ torch cuda:", torch.version.cuda)
print("✅ transformers:", transformers.__version__)
print("✅ accelerate:", accelerate.__version__)
print("✅ einops:", einops.__version__)
print("✅ timm:", timm.__version__)
EOF

echo ""
echo "✅ Spatial-MLLM environment setup complete! (reference)"
echo "---------------------------------------------"
echo "Conda env: spatial-mllm"
echo "Python: $(python --version)"
echo "PIP cache: $PIP_CACHE_DIR"
echo "TMPDIR: $TMPDIR"
echo "HF_HOME: $HF_HOME"
echo "Torch CUDA: $(python -c 'import torch; print(torch.version.cuda)')"
echo "Repo: $REPO_DIR"
echo "---------------------------------------------"
