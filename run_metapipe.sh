#!/bin/bash

#==============================================================================
# MetaPipe - Comprehensive Experiment Runner
#==============================================================================
# Paper: MetaPipe: Budget-Constrained Time-Series Routing with Meta-Learning
# Simple wrapper for running MetaPipe experiments
#==============================================================================

hostname

# ===================== GPU SELECTION =====================
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
USE_GPUS="0"  # MetaPipe uses single GPU for routing

DEVICE=()
BACKEND="cpu"
if command -v nvidia-smi &>/dev/null && [[ -n "$CUDA_VISIBLE_DEVICES" ]] && [[ -n "$USE_GPUS" ]]; then
  IFS=',' read -ra SEL <<< "$USE_GPUS"
  for lg in "${SEL[@]}"; do
    DEVICE+=("cuda:${lg}")
  done
  if [[ ${#DEVICE[@]} -gt 0 ]]; then BACKEND="cuda"; fi
fi
if [[ ${#DEVICE[@]} -eq 0 ]]; then DEVICE=("cpu"); BACKEND="cpu"; fi

echo "Visible (physical) GPUs: ${CUDA_VISIBLE_DEVICES}"
echo "Using (logical) devices: ${DEVICE[*]}"
echo "Backend: ${BACKEND}"

# ===================== CONDA ENV =====================
ENV_NAME="llms"
if [[ -z "$CONDA_DEFAULT_ENV" || "$CONDA_DEFAULT_ENV" != "$ENV_NAME" ]]; then
  if ! command -v conda &>/dev/null; then echo "❌ Conda not found."; exit 1; fi
  source activate "$ENV_NAME" || { echo "❌ Could not activate Conda env '$ENV_NAME'."; exit 1; }
fi

# ===================== TEST CONFIGS =====================

# ⚡ QUICK TEST: Uncomment for fast validation (~1 minute)
#MODE="quick"
#EXTRA_ARGS="--quick"
#EPISODES=50

# 🏆 FULL BENCHMARK: Uncomment for comprehensive results (~30-60 minutes)
MODE="full"
EXTRA_ARGS=""
EPISODES=500

# ===================== OUTPUT DIRECTORY =====================
OUTPUT_DIR="RESULTS/raw_data"
mkdir -p "$OUTPUT_DIR"

# ===================== RUN EXPERIMENTS =====================

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🚀 MetaPipe Experiments"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Mode: ${MODE}"
echo "Episodes: ${EPISODES}"
echo "Output: ${OUTPUT_DIR}"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Build command
CMD="python run_experiments.py"
CMD+=" --n_episodes ${EPISODES}"
CMD+=" --output_dir ${OUTPUT_DIR}"
if [[ -n "$EXTRA_ARGS" ]]; then
  CMD+=" ${EXTRA_ARGS}"
fi

echo "⏳ Running: ${CMD}"
echo ""

start_time=$(date +%s)

# Run experiments
if eval "$CMD"; then
  end_time=$(date +%s)
  duration=$((end_time - start_time))

  echo ""
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
  echo "✅ Experiments completed successfully!"
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
  echo "⏱️  Duration: ${duration}s"
  echo "📁 Results: ${OUTPUT_DIR}"
  echo ""
  echo "Generated files:"
  ls -lh "${OUTPUT_DIR}"/*.csv "${OUTPUT_DIR}"/*.png 2>/dev/null | awk '{print "   " $9 " (" $5 ")"}'
  echo ""
  echo "📊 Next steps:"
  echo "   1. View figures: open ${OUTPUT_DIR}/*.png"
  echo "   2. Check data: cat ${OUTPUT_DIR}/*.csv"
  echo "   3. Copy to paper: cp ${OUTPUT_DIR}/*.png ../Latex/figures/"
  echo "   4. Update paper: cd ../Latex && make"
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
  exit 0
else
  end_time=$(date +%s)
  duration=$((end_time - start_time))

  echo ""
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
  echo "❌ Experiments failed after ${duration}s"
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
  echo "Check the output above for errors"
  echo "Common issues:"
  echo "   - Missing dependencies: pip install -r requirements.txt"
  echo "   - Wrong environment: conda activate llms"
  echo "   - Missing data files"
  exit 1
fi
