#!/bin/bash

#SBATCH --job-name=amazon_ind_sci_tinyllama
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#SBATCH --partition=general
#SBATCH --mail-type=ALL
#SBATCH --mail-user=lixiangl@andrew.cmu.edu
#SBATCH --gres=gpu:A100_40GB:4
#SBATCH --time=2-00:00:00
#SBATCH --mem=128G
#SBATCH --cpus-per-task=16
#SBATCH --nodes=1

set -euo pipefail

# Slurm runs a temporary copy of this script. Submit from the repository root
# or code directory, or set HLLM_ROOT to the repository path when submitting
# from elsewhere. Slurm logs go to the submission directory.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -n "${HLLM_ROOT:-}" ]]; then
    HLLM_ROOT="$(cd "${HLLM_ROOT}" && pwd)"
elif [[ -f "${SCRIPT_DIR}/run.py" ]]; then
    HLLM_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
elif [[ -n "${SLURM_SUBMIT_DIR:-}" && -f "${SLURM_SUBMIT_DIR}/code/run.py" ]]; then
    HLLM_ROOT="$(cd "${SLURM_SUBMIT_DIR}" && pwd)"
elif [[ -n "${SLURM_SUBMIT_DIR:-}" && -f "${SLURM_SUBMIT_DIR}/run.py" ]]; then
    HLLM_ROOT="$(cd "${SLURM_SUBMIT_DIR}/.." && pwd)"
else
    echo "Cannot locate the HLLM repository. Set HLLM_ROOT to its absolute path." >&2
    exit 1
fi
SCRIPT_DIR="${HLLM_ROOT}/code"
if [[ ! -f "${SCRIPT_DIR}/run.py" ]]; then
    echo "Missing training entry point: ${SCRIPT_DIR}/run.py" >&2
    exit 1
fi
cd "${SCRIPT_DIR}"
mkdir -p outputs

CONDA_ENV="${CONDA_ENV:-hllm}"
if command -v conda >/dev/null 2>&1; then
    eval "$(conda shell.bash hook)"
    conda activate "${CONDA_ENV}"
fi

PYTHON_BIN="${PYTHON_BIN:-python3}"

# Full dataset by default. To run a reduced-density version, submit with, e.g.:
# DATASET=amazon_industrial_and_scientific_25_percent \
#     sbatch code/srun_baseline_amazon_industrial_and_scientific_tinyllama.sh
DATASET="${DATASET:-amazon_industrial_and_scientific}"
RUN_NAME="${RUN_NAME:-model_${DATASET}_tinyllama_baseline}"

DATA_PATH="${DATA_PATH:-${HLLM_ROOT}/dataset}"
INFO_PATH="${INFO_PATH:-${HLLM_ROOT}/information}"
PRETRAIN_DIR="${PRETRAIN_DIR:-${HLLM_ROOT}/TinyLlama-1.1B-intermediate-step-1431k-3T}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-${HLLM_ROOT}/${RUN_NAME}/}"
TEXT_KEYS="${TEXT_KEYS:-[\"title\",\"description\"]}"

EPOCHS="${EPOCHS:-5}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-16}"
MAX_TEXT_LENGTH="${MAX_TEXT_LENGTH:-256}"
MAX_ITEM_LIST_LENGTH="${MAX_ITEM_LIST_LENGTH:-10}"
LEARNING_RATE="${LEARNING_RATE:-1e-4}"
STAGE="${STAGE:-3}"
GPU_IDS="${GPU_IDS:-0,1,2,3}"
NPROC_PER_NODE="${NPROC_PER_NODE:-4}"

interaction_file="${DATA_PATH}/${DATASET}.csv"
information_file="${INFO_PATH}/${DATASET}.csv"
if [[ ! -f "${interaction_file}" ]]; then
    echo "Missing interaction file: ${interaction_file}" >&2
    exit 1
fi
if [[ ! -f "${information_file}" ]]; then
    echo "Missing information file: ${information_file}" >&2
    exit 1
fi

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
    --val_only False
    --finetune_clueweb False
    --gen_relevance_score False
    --baseline_train True
    --clueweb_pretrain False
    --gradient_checkpointing True
    --stage "${STAGE}"
    --clueweb_project "${RUN_NAME}"
)

export nproc_per_node="${NPROC_PER_NODE}"

echo "Running Amazon Industrial and Scientific baseline with TinyLlama"
echo "Dataset: ${DATASET}"
echo "Interaction file: ${interaction_file}"
echo "Information file: ${information_file}"
echo "Pretrained model: ${PRETRAIN_DIR}"
echo "Checkpoint directory: ${CHECKPOINT_DIR}"

CUDA_VISIBLE_DEVICES="${GPU_IDS}" "${HLLM_ROOT}/TORCHRUN" run.py "${ARGS[@]}"
