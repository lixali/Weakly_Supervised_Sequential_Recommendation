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

if [[ -z "${PYTORCH_ALLOC_CONF:-}" && -z "${PYTORCH_CUDA_ALLOC_CONF:-}" ]]; then
    export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
fi

SCRIPT_DIR="$(dirname "${BASH_SOURCE[0]}")"
cd "${SCRIPT_DIR}"
mkdir -p outputs

CONDA_ENV="${CONDA_ENV:-hllm}"
TRAIN_VENV_DIR="${TRAIN_VENV_DIR:-../.venv_train}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
GET_PIP_URL="${GET_PIP_URL:-https://bootstrap.pypa.io/get-pip.py}"
BOOTSTRAP_TRAIN_ENV="${BOOTSTRAP_TRAIN_ENV:-True}"

USING_CONDA=0
if command -v conda >/dev/null 2>&1; then
    eval "$(conda shell.bash hook)"
    if conda env list | awk '{print $1}' | grep -qx "${CONDA_ENV}"; then
        conda activate "${CONDA_ENV}"
        PYTHON_BIN="$(command -v python3)"
        USING_CONDA=1
    else
        echo "Conda env ${CONDA_ENV} not found; using a repo-local training virtualenv."
    fi
fi

python_has_module() {
    local python_bin="$1"
    local module_name="$2"
    "${python_bin}" - "${module_name}" <<'PY' >/dev/null 2>&1
import importlib.util
import sys

raise SystemExit(0 if importlib.util.find_spec(sys.argv[1]) else 1)
PY
}

ensure_pip() {
    local python_bin="$1"
    if "${python_bin}" -m pip --version >/dev/null 2>&1; then
        return
    fi

    echo "pip is missing for ${python_bin}; bootstrapping pip locally."
    local get_pip
    get_pip="$(mktemp /tmp/get-pip.XXXXXX.py)"
    "${python_bin}" - "${GET_PIP_URL}" "${get_pip}" <<'PY'
import sys
import urllib.request

url, output_path = sys.argv[1], sys.argv[2]
urllib.request.urlretrieve(url, output_path)
PY
    "${python_bin}" "${get_pip}"
}

ensure_train_python() {
    if [[ "${USING_CONDA}" == "1" ]]; then
        return
    fi

    if [[ ! -x "${TRAIN_VENV_DIR}/bin/python" ]]; then
        echo "Creating repo-local training virtualenv at ${TRAIN_VENV_DIR}"
        python3 -m venv --system-site-packages --without-pip "${TRAIN_VENV_DIR}"
    fi

    PYTHON_BIN="${TRAIN_VENV_DIR}/bin/python"
    ensure_pip "${PYTHON_BIN}"
}

ensure_train_packages() {
    if [[ "${BOOTSTRAP_TRAIN_ENV}" != "True" ]]; then
        return
    fi

    ensure_pip "${PYTHON_BIN}"
    if ! python_has_module "${PYTHON_BIN}" packaging; then
        "${PYTHON_BIN}" -m pip install -U packaging
    fi

    local missing
    missing="$("${PYTHON_BIN}" - <<'PY'
import importlib.util
from packaging.version import parse

checks = [
    ("colorlog", "colorlog"),
    ("colorama", "colorama"),
    ("yaml", "PyYAML"),
    ("pandas", "pandas"),
    ("torch_geometric", "torch_geometric"),
    ("lightning", "lightning"),
    ("deepspeed", "deepspeed"),
    ("tensorboardX", "tensorboardX"),
    ("sentencepiece", "sentencepiece"),
]

missing = []
for module_name, package_name in checks:
    if importlib.util.find_spec(module_name) is None:
        missing.append(package_name)

if importlib.util.find_spec("transformers") is None:
    missing.append("transformers>=4.50.0")
else:
    import transformers
    if parse(transformers.__version__) < parse("4.50.0"):
        missing.append("transformers>=4.50.0")

print(" ".join(missing))
PY
)"

    if [[ -n "${missing}" ]]; then
        echo "Installing missing Gemma training dependencies into $(${PYTHON_BIN} -c 'import sys; print(sys.prefix)'): ${missing}"
        DS_BUILD_OPS=0 "${PYTHON_BIN}" -m pip install -U ${missing}
    fi

    "${PYTHON_BIN}" - <<'PY'
import importlib.util
import sys

required = ["torch", "transformers", "colorlog", "pandas", "torch_geometric", "lightning", "deepspeed"]
missing = [module for module in required if importlib.util.find_spec(module) is None]
if missing:
    raise SystemExit(
        "Still missing required modules after bootstrap: "
        + ", ".join(missing)
        + "\nInstall PyTorch/CUDA for this machine first if torch is missing."
    )
PY
}

ensure_train_python
ensure_train_packages
export PATH="$(dirname "${PYTHON_BIN}"):${PATH}"

make_gpu_ids() {
    local ids=""
    for ((gpu_id = 0; gpu_id < NUM_GPUS; gpu_id++)); do
        ids="${ids}${ids:+,}${gpu_id}"
    done
    echo "${ids}"
}

detect_num_gpus() {
    "${PYTHON_BIN}" - <<'PY'
import torch

print(torch.cuda.device_count())
PY
}

count_gpu_ids() {
    local gpu_ids="$1"
    local -a gpu_id_list
    IFS=',' read -r -a gpu_id_list <<< "${gpu_ids}"
    echo "${#gpu_id_list[@]}"
}

DETECTED_NUM_GPUS="$(detect_num_gpus)"
if [[ -n "${GPU_IDS:-}" ]]; then
    DEFAULT_GPU_IDS="${GPU_IDS}"
    DEFAULT_NUM_GPUS="$(count_gpu_ids "${DEFAULT_GPU_IDS}")"
elif [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    DEFAULT_GPU_IDS="${CUDA_VISIBLE_DEVICES}"
    DEFAULT_NUM_GPUS="$(count_gpu_ids "${DEFAULT_GPU_IDS}")"
else
    DEFAULT_NUM_GPUS="${DETECTED_NUM_GPUS}"
    NUM_GPUS="${NUM_GPUS:-${DEFAULT_NUM_GPUS}}"
    DEFAULT_GPU_IDS="$(make_gpu_ids)"
fi

NUM_GPUS="${NUM_GPUS:-${DEFAULT_NUM_GPUS}}"
GPU_IDS="${GPU_IDS:-${DEFAULT_GPU_IDS}}"
NNODES="${NNODES:-1}"
NODE_RANK="${NODE_RANK:-0}"
NPROC_PER_NODE="${NPROC_PER_NODE:-${NUM_GPUS}}"

SELECTED_GPU_COUNT="$(CUDA_VISIBLE_DEVICES="${GPU_IDS}" detect_num_gpus)"
if ! [[ "${NPROC_PER_NODE}" =~ ^[1-9][0-9]*$ ]]; then
    echo "Error: NPROC_PER_NODE must be a positive integer; got '${NPROC_PER_NODE}'." >&2
    exit 1
fi
if (( SELECTED_GPU_COUNT < NPROC_PER_NODE )); then
    echo "Error: nproc_per_node=${NPROC_PER_NODE}, but only ${SELECTED_GPU_COUNT} GPU(s) are visible with CUDA_VISIBLE_DEVICES=${GPU_IDS}." >&2
    echo "Use a matching allocation, or set NUM_GPUS, GPU_IDS, and NPROC_PER_NODE to the GPUs available on this machine." >&2
    exit 1
fi

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

DATASET="${DATASET:-microlens_15_percent}"
DATASET_FOR_EVAL="${DATASET_FOR_EVAL:-}"
TEXT_KEYS=${TEXT_KEYS:-'["title","tag","description"]'}

EPOCHS="${EPOCHS:-5}"
# Conservative default for an 80GB GPU (~2x the 40GB batch of 16). Larger batch
# also enlarges the in-batch NCE negative pool. Override with TRAIN_BATCH_SIZE=..
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-32}"
# Optional: sample this many random negatives per user instead of using only the
# in-batch/sequence negatives. Leave empty to keep the default in-batch behavior.
NUM_NEGATIVES="${NUM_NEGATIVES:-}"
MAX_TEXT_LENGTH="${MAX_TEXT_LENGTH:-256}"
MAX_ITEM_LIST_LENGTH="${MAX_ITEM_LIST_LENGTH:-10}"
LEARNING_RATE="${LEARNING_RATE:-1e-4}"
STAGE="${STAGE:-3}"
STRATEGY="${STRATEGY:-deepspeed}"
PRECISION="${PRECISION:-bf16-mixed}"
LOG_WANDB="${LOG_WANDB:-False}"

VAL_ONLY="${VAL_ONLY:-False}"
FINETUNE_CLUEWEB="${FINETUNE_CLUEWEB:-False}"
BASELINE_TRAIN="${BASELINE_TRAIN:-True}"
CLUEWEB_PRETRAIN="${CLUEWEB_PRETRAIN:-False}"
GEN_RELEVANCE_SCORE="${GEN_RELEVANCE_SCORE:-False}"
GRADIENT_CHECKPOINTING="${GRADIENT_CHECKPOINTING:-True}"

"${PYTHON_BIN}" - <<'PY'
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
    --log_wandb "${LOG_WANDB}"
)

if [[ -n "${DATASET_FOR_EVAL}" ]]; then
    ARGS+=(--dataset_for_eval "${DATASET_FOR_EVAL}")
fi

if [[ -n "${NUM_NEGATIVES}" ]]; then
    ARGS+=(--num_negatives "${NUM_NEGATIVES}")
fi

echo "Running HLLM with Gemma 3 1B"
echo "GPUs: ${GPU_IDS}  nproc_per_node=${nproc_per_node}  nnodes=${nnodes}  node_rank=${node_rank}"
echo "Pretrained model: ${PRETRAIN_DIR}"
echo "Checkpoint dir: ${CHECKPOINT_DIR}"
echo "Python: ${PYTHON_BIN}"

TORCHRUN_ARGS=(
    --node_rank="${node_rank}"
    --nproc_per_node="${nproc_per_node}"
    --nnodes="${nnodes}"
)

if [[ -n "${master_addr:-}" ]]; then
    TORCHRUN_ARGS+=(--master_addr="${master_addr}")
fi

if [[ -n "${master_port:-}" ]]; then
    TORCHRUN_ARGS+=(--master_port="${master_port}")
else
    TORCHRUN_ARGS+=(--master_port="$((RANDOM % (58999 - 50001 + 1) + 50001))")
fi

CUDA_VISIBLE_DEVICES="${GPU_IDS}" "${PYTHON_BIN}" -m torch.distributed.run "${TORCHRUN_ARGS[@]}" run.py "${ARGS[@]}"
