#!/bin/bash

#SBATCH --job-name=hllm_gemma3_1b
#SBATCH --output=outputs/%x-%j.out
#SBATCH --error=outputs/%x-%j.err
#SBATCH --partition=general
#SBATCH --mail-type=ALL
#SBATCH --mail-user=lixiangl@andrew.cmu.edu
#SBATCH --gres=gpu:A100_40GB:4
#SBATCH --time=2-00:00:00
#SBATCH --mem=128G
#SBATCH --cpus-per-task 16
#SBATCH --nodes=1

set -euo pipefail

SCRIPT_DIR="$(dirname "${BASH_SOURCE[0]}")"
cd "${SCRIPT_DIR}"
mkdir -p outputs

CONDA_ENV="${CONDA_ENV:-hllm}"
if command -v conda >/dev/null 2>&1; then
    eval "$(conda shell.bash hook)"
    conda activate "${CONDA_ENV}"
fi

make_gpu_ids() {
    local ids=""
    for ((gpu_id = 0; gpu_id < NUM_GPUS; gpu_id++)); do
        ids="${ids}${ids:+,}${gpu_id}"
    done
    echo "${ids}"
}

NUM_GPUS="${NUM_GPUS:-4}"
GPU_IDS="${GPU_IDS:-$(make_gpu_ids)}"
NNODES="${NNODES:-1}"
NODE_RANK="${NODE_RANK:-0}"
NPROC_PER_NODE="${NPROC_PER_NODE:-${NUM_GPUS}}"

export nproc_per_node="${NPROC_PER_NODE}"
export nnodes="${NNODES}"
export node_rank="${NODE_RANK}"

if [[ "${NNODES}" != "1" ]]; then
    : "${MASTER_ADDR:?Set MASTER_ADDR for multi-node training}"
    : "${MASTER_PORT:?Set MASTER_PORT for multi-node training}"
    export master_addr="${MASTER_ADDR}"
    export master_port="${MASTER_PORT}"
elif [[ -n "${MASTER_ADDR:-}" || -n "${MASTER_PORT:-}" ]]; then
    export master_addr="${MASTER_ADDR:-127.0.0.1}"
    export master_port="${MASTER_PORT:-29500}"
fi

RUN_NAME="${RUN_NAME:-model_pixelrec_gemma3_1b_baseline}"
PRETRAIN_DIR="${PRETRAIN_DIR:-../gemma-3-1b-pt}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-../${RUN_NAME}/}"
INFO_PATH="${INFO_PATH:-../information}"
DATA_PATH="${DATA_PATH:-../dataset}"

DATASET="${DATASET:-Pixel200K_5_percent}"
DATASET_FOR_EVAL="${DATASET_FOR_EVAL:-}"
TEXT_KEYS=${TEXT_KEYS:-'[\"title\",\"tag\",\"description\"]'}

EPOCHS="${EPOCHS:-5}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-16}"
MAX_TEXT_LENGTH="${MAX_TEXT_LENGTH:-256}"
MAX_ITEM_LIST_LENGTH="${MAX_ITEM_LIST_LENGTH:-10}"
LEARNING_RATE="${LEARNING_RATE:-1e-4}"
STAGE="${STAGE:-3}"
STRATEGY="${STRATEGY:-deepspeed}"
PRECISION="${PRECISION:-bf16-mixed}"

VAL_ONLY="${VAL_ONLY:-False}"
FINETUNE_CLUEWEB="${FINETUNE_CLUEWEB:-False}"
BASELINE_TRAIN="${BASELINE_TRAIN:-True}"
CLUEWEB_PRETRAIN="${CLUEWEB_PRETRAIN:-False}"
GEN_RELEVANCE_SCORE="${GEN_RELEVANCE_SCORE:-False}"
GRADIENT_CHECKPOINTING="${GRADIENT_CHECKPOINTING:-True}"

python3 - <<'PY'
import os
from packaging.version import parse
import transformers

required = os.environ.get("REQUIRED_TRANSFORMERS_VERSION", "4.50.0")
if parse(transformers.__version__) < parse(required):
    raise SystemExit(
        f"Gemma 3 requires transformers>={required}; found {transformers.__version__}. "
        f"Run: python3 -m pip install -U 'transformers>={required}'"
    )
PY

ARGS=(
    --config_file overall/LLM_deepspeed.yaml HLLM/HLLM.yaml
    --loss nce
    --epochs "${EPOCHS}"
    --dataset "${DATASET}"
    --train_batch_size "${TRAIN_BATCH_SIZE}"
    --MAX_TEXT_LENGTH "${MAX_TEXT_LENGTH}"
    --MAX_ITEM_LIST_LENGTH "${MAX_ITEM_LIST_LENGTH}"
    --checkpoint_dir "${CHECKPOINT_DIR}"
    --optim_args.learning_rate "${LEARNING_RATE}"
    --item_pretrain_dir "${PRETRAIN_DIR}"
    --user_pretrain_dir "${PRETRAIN_DIR}"
    --data_path "${DATA_PATH}"
    --text_path "${INFO_PATH}"
    --text_keys "${TEXT_KEYS}"
    --val_only "${VAL_ONLY}"
    --finetune_clueweb "${FINETUNE_CLUEWEB}"
    --baseline_train "${BASELINE_TRAIN}"
    --clueweb_pretrain "${CLUEWEB_PRETRAIN}"
    --gen_relevance_score "${GEN_RELEVANCE_SCORE}"
    --gradient_checkpointing "${GRADIENT_CHECKPOINTING}"
    --strategy "${STRATEGY}"
    --precision "${PRECISION}"
    --stage "${STAGE}"
    --clueweb_project "${RUN_NAME}"
)

if [[ -n "${DATASET_FOR_EVAL}" ]]; then
    ARGS+=(--dataset_for_eval "${DATASET_FOR_EVAL}")
fi

echo "Running HLLM with Gemma 3 1B"
echo "GPUs: ${GPU_IDS}  nproc_per_node=${nproc_per_node}  nnodes=${nnodes}  node_rank=${node_rank}"
echo "Pretrained model: ${PRETRAIN_DIR}"
echo "Checkpoint dir: ${CHECKPOINT_DIR}"

CUDA_VISIBLE_DEVICES="${GPU_IDS}" python3 main.py "${ARGS[@]}"
