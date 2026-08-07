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

# ============================================================
# 1. Use tmux on regular Ubuntu/AutoDL.
# 2. Do not use tmux in Google Colab or inside a Slurm job.
# 3. Print output to the terminal and save it to a timestamped log.
# ============================================================

IS_COLAB=0
if [[ -n "${COLAB_RELEASE_TAG:-}" || -n "${COLAB_GPU:-}" || -d "/content" ]]; then
    IS_COLAB=1
fi

# Start this script again inside tmux on Ubuntu/AutoDL.
if (( IS_COLAB == 0 )) \
    && [[ -z "${SLURM_JOB_ID:-}" ]] \
    && [[ -z "${TMUX:-}" ]] \
    && [[ "${HLLM_TMUX_CHILD:-0}" != "1" ]]; then

    if ! command -v tmux >/dev/null 2>&1; then
        echo "tmux is not installed. Install it with:"
        echo "apt-get update && apt-get install -y tmux"
        exit 1
    fi

    CURRENT_SCRIPT="$(readlink -f "${BASH_SOURCE[0]}")"
    CURRENT_SCRIPT_DIR="$(dirname "${CURRENT_SCRIPT}")"
    TMUX_SESSION_NAME="${TMUX_SESSION_NAME:-hllm_gemma3_1b}"

    # Do not create a second training session with the same name.
    if tmux has-session -t "${TMUX_SESSION_NAME}" 2>/dev/null; then
        echo "tmux session '${TMUX_SESSION_NAME}' already exists."
        exec tmux attach-session -t "${TMUX_SESSION_NAME}"
    fi

    TMUX_LAUNCHER="$(mktemp "/tmp/${TMUX_SESSION_NAME}.XXXXXX.sh")"

    {
        echo '#!/bin/bash'
        echo 'set +e'

        # Preserve variables such as TRAIN_BATCH_SIZE and PRECISION.
        export -p

        printf 'export HLLM_TMUX_CHILD=1\n'
        printf 'cd %q\n' "${CURRENT_SCRIPT_DIR}"
        printf 'bash %q\n' "${CURRENT_SCRIPT}"
        printf 'exit_code=$?\n'
        printf 'echo "Training finished with exit code ${exit_code}"\n'
        printf 'rm -f -- "$0"\n'
        printf 'exec bash\n'
    } > "${TMUX_LAUNCHER}"

    chmod 700 "${TMUX_LAUNCHER}"

    tmux new-session \
        -d \
        -s "${TMUX_SESSION_NAME}" \
        "${TMUX_LAUNCHER}"

    echo "Started tmux session: ${TMUX_SESSION_NAME}"

    if [[ -t 0 && -t 1 ]]; then
        exec tmux attach-session -t "${TMUX_SESSION_NAME}"
    else
        echo "Attach later with:"
        echo "tmux attach -t ${TMUX_SESSION_NAME}"
        exit 0
    fi
fi

# ============================================================
# The code below runs:
# - directly in Colab;
# - inside tmux on Ubuntu/AutoDL;
# - directly inside a Slurm job.
# ============================================================

if [[ -z "${PYTORCH_ALLOC_CONF:-}" && -z "${PYTORCH_CUDA_ALLOC_CONF:-}" ]]; then
    export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
fi

SCRIPT_DIR="$(dirname "${BASH_SOURCE[0]}")"
cd "${SCRIPT_DIR}"
mkdir -p outputs

# Example:
# outputs/hllm_gemma3_1b_20260719_183045.log
RUN_TIMESTAMP="${RUN_TIMESTAMP:-$(date '+%Y%m%d_%H%M%S')}"
LOG_FILE="${LOG_FILE:-outputs/hllm_gemma3_1b_${RUN_TIMESTAMP}.log}"

# Print stdout and stderr to the terminal and save both to the log.
exec > >(tee -a "${LOG_FILE}") 2>&1

echo "Log file: ${LOG_FILE}"

log_source_version() {
    local hllm_file="REC/model/HLLM/hllm.py"
    local reference_commit="d129b808afa58cfc854106858954dd567edaa721"
    local source_hash

    if command -v sha256sum >/dev/null 2>&1; then
        source_hash="$(sha256sum "${hllm_file}" | awk '{print $1}')"
    else
        source_hash="unavailable"
    fi

    if command -v git >/dev/null 2>&1 && git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
        local repo_commit
        local hllm_commit
        local hllm_blob
        local hllm_state

        repo_commit="$(git rev-parse HEAD)"
        hllm_commit="$(git log -1 --format=%H -- "${hllm_file}")"
        hllm_blob="$(git hash-object "${hllm_file}")"
        if git diff --quiet HEAD -- "${hllm_file}"; then
            hllm_state="clean"
        else
            hllm_state="modified"
        fi

        echo "Repository commit: ${repo_commit}"
        echo "HLLM source commit: ${hllm_commit}"
        echo "HLLM source state: ${hllm_state}"
        if [[ "${hllm_blob}" == "$(git rev-parse "${reference_commit}:code/${hllm_file}" 2>/dev/null || true)" ]]; then
            echo "HLLM source content matches commit: ${reference_commit}"
        else
            echo "HLLM source content matches commit: no match for ${reference_commit}"
        fi
    else
        echo "Repository commit: unavailable (not a Git checkout)"
        echo "HLLM source commit: unavailable"
        echo "HLLM source state: unknown"
        echo "HLLM source content matches commit: unavailable"
    fi

    echo "HLLM source file: ${hllm_file}"
    echo "HLLM source SHA256: ${source_hash}"
}

log_source_version


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
from importlib.metadata import version
from packaging.version import parse

checks = [
    ("colorlog", "colorlog"),
    ("colorama", "colorama"),
    ("yaml", "PyYAML"),
    ("pandas", "pandas"),
    ("torch_geometric", "torch_geometric"),
    ("lightning", "lightning"),
    ("deepspeed", "deepspeed==0.19.2"),
    ("tensorboardX", "tensorboardX"),
    ("sentencepiece", "sentencepiece"),
]

missing = []
for module_name, package_name in checks:
    if importlib.util.find_spec(module_name) is None:
        missing.append(package_name)

if importlib.util.find_spec("deepspeed") is not None and version("deepspeed") != "0.19.2":
    missing.append("deepspeed==0.19.2")

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
from importlib.metadata import version

required = ["torch", "transformers", "colorlog", "pandas", "torch_geometric", "lightning", "deepspeed"]
missing = [module for module in required if importlib.util.find_spec(module) is None]
if missing:
    raise SystemExit(
        "Still missing required modules after bootstrap: "
        + ", ".join(missing)
        + "\nInstall PyTorch/CUDA for this machine first if torch is missing."
    )
installed_deepspeed = version("deepspeed")
if installed_deepspeed != "0.19.2":
    raise SystemExit(f"DeepSpeed version mismatch: required 0.19.2, found {installed_deepspeed}.")
print(f"DeepSpeed version: {installed_deepspeed}")
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

DATASET="${DATASET:-Pixel200K_5_percent}"
DATASET_FOR_EVAL="${DATASET_FOR_EVAL:-}"
TEXT_KEYS=${TEXT_KEYS:-'["title","tag","description"]'}

EPOCHS="${EPOCHS:-5}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-32}"
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
