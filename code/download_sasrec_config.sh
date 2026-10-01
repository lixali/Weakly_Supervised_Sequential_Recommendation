#!/usr/bin/env bash
set -euo pipefail
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HLLM_ROOT="${HLLM_ROOT:-$(cd "${script_dir}/.." && pwd)}"
PRETRAIN_DIR="${PRETRAIN_DIR:-${HLLM_ROOT}/TinyLlama-1.1B-intermediate-step-1431k-3T}"
echo "Preparing SASRec paper baseline architecture (random initialization; no weights needed)."
"${PYTHON_BIN:-python3}" - "${PRETRAIN_DIR}" <<'PY'
import json
import os
from pathlib import Path
import sys
import tempfile
import urllib.request

destination = Path(sys.argv[1]) / 'config.json'
url = ('https://huggingface.co/TinyLlama/'
       'TinyLlama-1.1B-intermediate-step-1431k-3T/resolve/main/config.json')
expected = {'model_type': 'llama', 'hidden_size': 2048, 'num_hidden_layers': 22,
            'intermediate_size': 5632, 'num_attention_heads': 32,
            'num_key_value_heads': 4, 'vocab_size': 32000}
if destination.is_file():
    content = destination.read_bytes()
else:
    print(f'Downloading {url}', flush=True)
    with urllib.request.urlopen(url, timeout=60) as response:
        content = response.read()
config = json.loads(content)
for name, value in expected.items():
    if config.get(name) != value:
        raise SystemExit(f'{destination}: expected {name}={value}, got {config.get(name)!r}')
if not destination.is_file():
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=destination.parent, delete=False) as temporary:
        temporary.write(content)
        temporary_path = temporary.name
    os.replace(temporary_path, destination)
print(f'Architecture ready: {destination}')
print('SASRec paper baseline uses user_llm_init=False; existing pretrained weights are not loaded.')
PY
