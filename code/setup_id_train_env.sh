#!/usr/bin/env bash
# Sourced by run_amazon_id_baseline.sh; keeps pip and torchrun on one Python.
# HLLM_ROOT must already point to the repository.
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

with urllib.request.urlopen(sys.argv[1], timeout=60) as response:
    with open(sys.argv[2], "wb") as destination:
        destination.write(response.read())
PY
    "${PYTHON_BIN}" "${get_pip}"
}

check_train_packages() {
    "${PYTHON_BIN}" - "$1" <<'PY'
import importlib
import importlib.util
import sys
from importlib.metadata import PackageNotFoundError, version

# The vendored Llama implementation expects the Transformers 4.x config API
# (including rope_theta and rope_scaling). Transformers 5 is incompatible.
transformers_versions = ">=4.50.0,<5"
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
    ("transformers", f"transformers{transformers_versions}"),
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
            from packaging.specifiers import SpecifierSet
            if version(module) not in SpecifierSet(transformers_versions):
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
    print(
        f"Training dependencies verified; DeepSpeed {version('deepspeed')}; "
        f"Transformers {version('transformers')}"
    )
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
        echo "Installing missing or incompatible ID baseline training dependencies: ${missing}"
        DS_BUILD_OPS=0 "${PYTHON_BIN}" -m pip install --upgrade "${missing_packages[@]}"
    fi
fi
check_train_packages validate
