#!/bin/bash
#SBATCH --job-name=pi0fast-hooks-libero
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --partition=scavenge_gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=256G
#SBATCH --time=32:00:00

set -euo pipefail

OPENPI_DIR="/nfs/roberts/project/pi_tkf6/as4643/projects/openpi-pi0fast-hooks"
SERVER_SIF="/nfs/roberts/project/pi_tkf6/as4643/openpi_server.sif"
LIBERO_SIF="/nfs/roberts/project/pi_tkf6/as4643/libero.sif"
NFS_SCRATCH_ROOT="/nfs/roberts/scratch"

CHECKPOINTS=(
  /nfs/roberts/scratch/pi_tkf6/zs377/checkpoints/pi0_fast_libero/29999
)

NUM_TRIALS=1

PY_PATH="/app/src:/app/third_party/libero:/app/packages/openpi-client/src"

########################################
# Hook configuration
########################################

HOOK_CONFIG="/app/hooks.yaml"

EXP_NAME=$(basename "${CHECKPOINTS[0]}")
LOG_ROOT="logs/${EXP_NAME}"
mkdir -p "$LOG_ROOT"

BASE_PORT=$((20000 + (SLURM_JOB_ID % 10000)))

SERVER_PID=""
MEM_MONITOR_PID=""

cleanup_server() {
  if [[ -n "${SERVER_PID:-}" ]]; then
    kill "$SERVER_PID" 2>/dev/null || true
    wait "$SERVER_PID" 2>/dev/null || true
    SERVER_PID=""
  fi

  if [[ -n "${MEM_MONITOR_PID:-}" ]]; then
    kill "$MEM_MONITOR_PID" 2>/dev/null || true
    wait "$MEM_MONITOR_PID" 2>/dev/null || true
    MEM_MONITOR_PID=""
  fi
}

wait_for_server() {
  echo "Waiting for server on port $PORT..."

  for i in {1..180}; do
    if [[ -f "$SERVER_LOG" ]] && grep -q "DEBUG: entering serve_forever\|server listening" "$SERVER_LOG"; then
      echo "Server is ready"
      sleep 5
      return 0
    fi

    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
      echo "Server exited early"
      cat "$SERVER_LOG"
      return 1
    fi

    sleep 2
  done

  echo "Server failed to become ready"
  cat "$SERVER_LOG"
  return 1
}

wait_for_gpu_to_clear() {
  echo "Waiting for GPU memory to settle..."
  sleep 10
  nvidia-smi || true
  sleep 10
}

run_single() {
  CKPT=$1
  CKPT_IDX=$2

  CKPT_NAME=$(basename "$CKPT")
  PORT=$((BASE_PORT + CKPT_IDX))

  CKPT_LOG_DIR="$LOG_ROOT/$CKPT_NAME"
  mkdir -p "$CKPT_LOG_DIR"

  TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
  RECORD_DIR="/nfs/roberts/scratch/pi_tkf6/as4643/policy_records_pi0fast_libero_${TIMESTAMP}"
  mkdir -p "$RECORD_DIR"

  SERVER_LOG="$CKPT_LOG_DIR/server.log"
  EVAL_LOG="$CKPT_LOG_DIR/eval.log"
  MEM_LOG="$CKPT_LOG_DIR/memory.log"

  echo "========================================"
  echo "Checkpoint: $CKPT"
  echo "Port: $PORT"
  echo "Hook config: $HOOK_CONFIG"
  echo "========================================"

  echo "[1/3] Starting server..."

  touch "$SERVER_LOG"

  apptainer exec --nv --containall \
    --bind "$OPENPI_DIR:/app" \
    --bind "$NFS_SCRATCH_ROOT:$NFS_SCRATCH_ROOT" \
    "$SERVER_SIF" \
    bash -c "
      set -x
      cd /app

      export PYTHONPATH=$PY_PATH
      export CUDA_VISIBLE_DEVICES=0

      export HOME=/tmp/home_\${SLURM_JOB_ID:-\$\$}
      mkdir -p \"\$HOME\"

      export XLA_PYTHON_CLIENT_PREALLOCATE=false
      export XLA_PYTHON_CLIENT_MEM_FRACTION=0.7

      export PYTHONUNBUFFERED=1
      export PYTHONFAULTHANDLER=1

      echo '========================================'
      echo 'Using hook config:'
      cat $HOOK_CONFIG
      echo '========================================'

      exec /.venv/bin/python -u scripts/serve_policy.py \
        --env LIBERO \
        --port $PORT \
        --record \
        --hook-config $HOOK_CONFIG \
        --record-dir "$RECORD_DIR" \
        policy:checkpoint \
        --policy.config pi0_fast_libero \
        --policy.dir $CKPT
    " > "$SERVER_LOG" 2>&1 &

  SERVER_PID=$!

  (
    while true; do
      echo "=== $(date) ===" >> "$MEM_LOG"

      nvidia-smi \
        --query-gpu=timestamp,name,memory.used,memory.free,memory.total,utilization.gpu \
        --format=csv,noheader \
        >> "$MEM_LOG" 2>&1 || true

      echo "CPU/RAM: $(free -h | awk '/^Mem/{print "used="$3" free="$4" total="$2}')" >> "$MEM_LOG"

      echo "" >> "$MEM_LOG"

      sleep 10
    done
  ) &

  MEM_MONITOR_PID=$!

  echo "[2/3] Waiting for server..."
  wait_for_server

  echo "[3/3] Running evaluation..."

  sleep $((RANDOM % 20))

  apptainer exec --nv --containall \
    --bind "$OPENPI_DIR:/app" \
    --bind "$NFS_SCRATCH_ROOT:$NFS_SCRATCH_ROOT" \
    "$LIBERO_SIF" \
    bash -c "
      set -x
      cd /app

      export PYTHONPATH=$PY_PATH
      export CUDA_VISIBLE_DEVICES=0

      export HOME=/tmp/home_\${SLURM_JOB_ID:-\$\$}
      mkdir -p \"\$HOME\"

      export PYTHONUNBUFFERED=1
      export PYTHONFAULTHANDLER=1

      mkdir -p /tmp/libero

      printf '%s\n' \
      'benchmark_root: /app/third_party/libero/libero/libero' \
      'bddl_files: /app/third_party/libero/libero/libero/./bddl_files' \
      'init_states: /app/third_party/libero/libero/libero/./init_files' \
      'datasets: /app/third_party/libero/libero/libero/../datasets' \
      'assets: /app/third_party/libero/libero/libero/./assets' \
      > /tmp/libero/config.yaml

      export MUJOCO_GL=osmesa
      export PYOPENGL_PLATFORM=osmesa

      exec /.venv/bin/python -u examples/libero/main.py \
        --args.task_suite_name libero_10 \
        --args.num_trials_per_task $NUM_TRIALS \
        --args.port $PORT \
        --args.host 127.0.0.1 \
        --args.record_dir "$RECORD_DIR"
    " > "$EVAL_LOG" 2>&1

  echo "Stopping server..."

  cleanup_server
  wait_for_gpu_to_clear
}

trap cleanup_server EXIT

echo "Starting sequential evaluation..."
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-unset}"

for idx in "${!CHECKPOINTS[@]}"; do
  run_single "${CHECKPOINTS[$idx]}" "$idx"
done

echo "========================================"
echo "ALL DONE"
echo "Logs saved to: $LOG_ROOT"
echo "========================================"