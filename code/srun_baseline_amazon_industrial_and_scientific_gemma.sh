#!/bin/bash

#SBATCH --job-name=amazon_ind_sci_gemma
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

# Full dataset by default. To run a reduced-density version, submit with, e.g.:
# DATASET=amazon_industrial_and_scientific_25_percent \
#     sbatch code/srun_baseline_amazon_industrial_and_scientific_gemma.sh
export DATASET="${DATASET:-amazon_industrial_and_scientific}"
export RUN_NAME="${RUN_NAME:-model_${DATASET}_gemma3_1b_baseline}"

export DATA_PATH="${DATA_PATH:-${HLLM_ROOT}/dataset}"
export INFO_PATH="${INFO_PATH:-${HLLM_ROOT}/information}"
export PRETRAIN_DIR="${PRETRAIN_DIR:-${HLLM_ROOT}/gemma-3-1b-pt}"
export CHECKPOINT_DIR="${CHECKPOINT_DIR:-${HLLM_ROOT}/${RUN_NAME}/}"
export TEXT_KEYS="${TEXT_KEYS:-[\"title\",\"description\"]}"

export EPOCHS="${EPOCHS:-5}"
export TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-16}"
export MAX_TEXT_LENGTH="${MAX_TEXT_LENGTH:-256}"
export MAX_ITEM_LIST_LENGTH="${MAX_ITEM_LIST_LENGTH:-10}"
export LEARNING_RATE="${LEARNING_RATE:-1e-4}"
export STAGE="${STAGE:-3}"
export GPU_IDS="${GPU_IDS:-0,1,2,3}"
export NUM_GPUS="${NUM_GPUS:-4}"
export NPROC_PER_NODE="${NPROC_PER_NODE:-4}"
export TMUX_SESSION_NAME="${TMUX_SESSION_NAME:-amazon_industrial_scientific_gemma3_1b}"

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

echo "Running Amazon Industrial and Scientific baseline with Gemma 3 1B"
echo "Dataset: ${DATASET}"
echo "Interaction file: ${interaction_file}"
echo "Information file: ${information_file}"
echo "Pretrained model: ${PRETRAIN_DIR}"
echo "Checkpoint directory: ${CHECKPOINT_DIR}"

exec bash "${SCRIPT_DIR}/srun_hllm_gemma3_1b.sh"
