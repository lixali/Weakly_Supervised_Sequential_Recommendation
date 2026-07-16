#!/bin/bash
set -euo pipefail

MODEL_ID="${MODEL_ID:-google/gemma-3-1b-pt}"
OUT_DIR="${OUT_DIR:-../gemma-3-1b-pt}"
CONDA_ENV="${CONDA_ENV:-hllm}"
HF_VENV_DIR="${HF_VENV_DIR:-../.venv_hf}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
GET_PIP_URL="${GET_PIP_URL:-https://bootstrap.pypa.io/get-pip.py}"
HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
HF_DOWNLOAD_WORKERS="${HF_DOWNLOAD_WORKERS:-1}"
export HF_HUB_DISABLE_XET HF_DOWNLOAD_WORKERS

SCRIPT_DIR="$(dirname "${BASH_SOURCE[0]}")"
cd "${SCRIPT_DIR}"

if command -v conda >/dev/null 2>&1; then
    eval "$(conda shell.bash hook)"
    conda activate "${CONDA_ENV}"
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

ensure_download_python() {
    if python_has_module "${PYTHON_BIN}" huggingface_hub; then
        return
    fi

    echo "huggingface_hub is not installed for ${PYTHON_BIN}."
    echo "Using a repo-local virtualenv at ${HF_VENV_DIR} instead of installing into system Python."

    if [[ ! -x "${HF_VENV_DIR}/bin/python" ]]; then
        python3 -m venv --without-pip "${HF_VENV_DIR}"
    fi

    PYTHON_BIN="${HF_VENV_DIR}/bin/python"
    if python_has_module "${PYTHON_BIN}" huggingface_hub; then
        return
    fi

    ensure_pip "${PYTHON_BIN}"
    "${PYTHON_BIN}" -m pip install -U pip huggingface_hub
}

ensure_download_python

if [[ -z "${HF_TOKEN:-${HUGGING_FACE_HUB_TOKEN:-${HUGGINGFACE_HUB_TOKEN:-}}}" ]]; then
    if ! "${PYTHON_BIN}" - <<'PY' >/dev/null 2>&1
from huggingface_hub import get_token

raise SystemExit(0 if get_token() else 1)
PY
    then
        echo "No Hugging Face token found."
        echo "Gemma is gated: accept the license at https://huggingface.co/google/gemma-3-1b-pt"
        echo "Then paste a read token from https://huggingface.co/settings/tokens"
        if [[ -t 0 ]]; then
            read -r -s -p "HF token (input hidden): " HF_TOKEN
            echo
            if [[ -z "${HF_TOKEN}" ]]; then
                echo "No token entered; cannot download gated Gemma weights."
                exit 2
            fi
            export HF_TOKEN
        else
            echo "Non-interactive shell: set HF_TOKEN before running this script."
            exit 2
        fi
    fi
fi

mkdir -p "${OUT_DIR}"
export MODEL_ID OUT_DIR

"${PYTHON_BIN}" - <<'PY'
import os
import sys

try:
    from huggingface_hub import snapshot_download
    from huggingface_hub.errors import GatedRepoError, HfHubHTTPError, LocalTokenNotFoundError
except ImportError as exc:
    raise SystemExit("huggingface_hub is required but could not be imported.") from exc

model_id = os.environ["MODEL_ID"]
out_dir = os.environ["OUT_DIR"]
max_workers = int(os.environ.get("HF_DOWNLOAD_WORKERS", "1"))
token = (
    os.environ.get("HF_TOKEN")
    or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    or os.environ.get("HUGGINGFACE_HUB_TOKEN")
)

print(f"Downloading {model_id} to {out_dir}")
print("If access fails, accept the Gemma license on Hugging Face and export HF_TOKEN.")
if os.environ.get("HF_HUB_DISABLE_XET") == "1":
    print("HF_HUB_DISABLE_XET=1, using the standard Hugging Face download path.")

try:
    snapshot_download(
        repo_id=model_id,
        local_dir=out_dir,
        token=token,
        max_workers=max_workers,
    )
except (GatedRepoError, LocalTokenNotFoundError) as exc:
    raise SystemExit(
        "\nGemma download needs Hugging Face access.\n"
        "1. Open https://huggingface.co/google/gemma-3-1b-pt and accept the license.\n"
        "2. Create a read token at https://huggingface.co/settings/tokens.\n"
        "3. Re-run with: HF_TOKEN=hf_... ./download_gemma3_1b.sh\n"
    ) from exc
except HfHubHTTPError as exc:
    raise SystemExit(
        f"\nHugging Face download failed: {exc}\n"
        "If this is a 401/403 error, accept the Gemma license and set HF_TOKEN.\n"
    ) from exc

print(f"Done: {out_dir}")
PY
