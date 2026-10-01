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

CONDA_ENV="${CONDA_ENV:-hllm}"
TRAIN_VENV_DIR="${TRAIN_VENV_DIR:-${HLLM_ROOT}/.venv_train}"
BOOTSTRAP_TRAIN_ENV="${BOOTSTRAP_TRAIN_ENV:-True}"
GET_PIP_URL="${GET_PIP_URL:-https://bootstrap.pypa.io/get-pip.py}"

# Start independently on a fresh Colab runtime. An explicit PYTHON_BIN takes
# precedence; otherwise prefer Conda, then create/reuse the training virtualenv.
# BOOTSTRAP_TRAIN_ENV=False disables creation/installation, but keeps validation.
if [[ -z "${PYTHON_BIN:-}" ]]; then
    if command -v conda >/dev/null 2>&1; then
        eval "$(conda shell.bash hook)"
        if conda env list | awk '{print $1}' | grep -qx "${CONDA_ENV}"; then
            conda activate "${CONDA_ENV}"
            PYTHON_BIN="$(command -v python3)"
        fi
    fi
    if [[ -z "${PYTHON_BIN:-}" ]]; then
        if [[ ! -x "${TRAIN_VENV_DIR}/bin/python" && "${BOOTSTRAP_TRAIN_ENV}" == "True" ]]; then
            echo "Creating repo-local training virtualenv at ${TRAIN_VENV_DIR}"
            python3 -m venv --system-site-packages --without-pip "${TRAIN_VENV_DIR}"
        fi
        if [[ -x "${TRAIN_VENV_DIR}/bin/python" ]]; then
            PYTHON_BIN="${TRAIN_VENV_DIR}/bin/python"
        fi
    fi
fi
PYTHON_BIN="${PYTHON_BIN:-python3}"
PYTHON_BIN="$("${PYTHON_BIN}" -c 'import sys; print(sys.executable)')"
# torchrun otherwise honors a possibly stale PYTHON_EXEC from the notebook.
# Keep installation, validation, and training workers on this same interpreter.
export PYTHON_EXEC="${PYTHON_BIN}"
export PATH="$(dirname "${PYTHON_BIN}"):${PATH}"
echo "Training Python: ${PYTHON_BIN}"

python_has_module() {
    "${PYTHON_BIN}" - "$1" <<'PY' >/dev/null 2>&1
import importlib.util
import sys

raise SystemExit(0 if importlib.util.find_spec(sys.argv[1]) else 1)
PY
}

ensure_pip() {
    if "${PYTHON_BIN}" -m pip --version >/dev/null 2>&1; then
        return
    fi
    if "${PYTHON_BIN}" -m ensurepip --upgrade; then
        return
    fi
    echo "Bootstrapping pip for ${PYTHON_BIN}"
    local get_pip
    get_pip="$(mktemp /tmp/get-pip.XXXXXX.py)"
    "${PYTHON_BIN}" - "${GET_PIP_URL}" "${get_pip}" <<'PY'
import sys
import urllib.request

urllib.request.urlretrieve(sys.argv[1], sys.argv[2])
PY
    "${PYTHON_BIN}" "${get_pip}"
}

check_train_packages() {
    "${PYTHON_BIN}" - "$1" <<'PY'
import importlib
import importlib.util
import sys
from importlib.metadata import PackageNotFoundError, version

requirements = [
    ("packaging", "packaging"),
    ("colorlog", "colorlog"),
    ("colorama", "colorama"),
    ("yaml", "PyYAML"),
    ("numpy", "numpy"),
    ("pandas", "pandas"),
    ("pytz", "pytz"),
    ("sklearn", "scikit-learn"),
    ("tqdm", "tqdm"),
    ("torch_geometric", "torch_geometric"),
    ("lightning", "lightning"),
    ("deepspeed", "deepspeed==0.19.2"),
    ("tensorboardX", "tensorboardX"),
    ("sentencepiece", "sentencepiece"),
    ("transformers", "transformers>=4.50.0"),
    ("wandb", "wandb"),
]
missing = []
for module, requirement in requirements:
    if importlib.util.find_spec(module) is None:
        missing.append(requirement)
        continue
    try:
        if module == "deepspeed" and version(module) != "0.19.2":
            missing.append(requirement)
        elif module == "transformers" and importlib.util.find_spec("packaging") is not None:
            from packaging.version import Version
            if Version(version(module)) < Version("4.50.0"):
                missing.append(requirement)
    except PackageNotFoundError:
        missing.append(requirement)

if sys.argv[1] == "report":
    print(" ".join(missing))
elif missing:
    raise SystemExit(
        f"Missing or incompatible training packages for {sys.executable}: "
        + ", ".join(missing)
        + "\nEnable BOOTSTRAP_TRAIN_ENV=True or install them in this interpreter."
    )
else:
    for module in ["torch"] + [module for module, _ in requirements]:
        try:
            importlib.import_module(module)
        except Exception as exc:
            raise SystemExit(f"Cannot import {module} using {sys.executable}: {exc}") from exc
    print(f"Training dependencies verified; DeepSpeed {version('deepspeed')}")
PY
}

# Require an existing PyTorch installation before bootstrapping dependencies.
if ! python_has_module torch; then
    echo "PyTorch is missing from ${PYTHON_BIN}. Install CUDA-enabled PyTorch for this machine first." >&2
    exit 1
fi
if [[ "${BOOTSTRAP_TRAIN_ENV}" == "True" ]]; then
    ensure_pip
    if ! python_has_module packaging; then
        "${PYTHON_BIN}" -m pip install packaging
    fi
    missing="$(check_train_packages report)"
    if [[ -n "${missing}" ]]; then
        read -r -a missing_packages <<< "${missing}"
        echo "Installing missing TinyLlama training dependencies: ${missing}"
        DS_BUILD_OPS=0 "${PYTHON_BIN}" -m pip install --upgrade "${missing_packages[@]}"
    fi
fi
check_train_packages validate

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
echo "Python: ${PYTHON_BIN}"

TORCHRUN_ARGS=(
    --node_rank="${NODE_RANK:-${node_rank:-0}}"
    --nproc_per_node="${NPROC_PER_NODE}"
    --nnodes="${NNODES:-${nnodes:-1}}"
    --master_addr="${MASTER_ADDR:-${master_addr:-127.0.0.1}}"
    --master_port="${MASTER_PORT:-${master_port:-$((RANDOM % (58999 - 50001 + 1) + 50001))}}"
)

# A system torchrun executable can use a different Python than the selected
# training environment. Invoke its module directly and preserve failure status.
CUDA_VISIBLE_DEVICES="${GPU_IDS}" exec "${PYTHON_BIN}" -m torch.distributed.run "${TORCHRUN_ARGS[@]}" run.py "${ARGS[@]}"
