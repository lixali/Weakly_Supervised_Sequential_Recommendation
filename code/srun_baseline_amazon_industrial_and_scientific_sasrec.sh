#!/usr/bin/env bash
#SBATCH --job-name=amazon_sasrec
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=2-00:00:00
set -euo pipefail

# Slurm executes a temporary copy; accept root/code submission directories.
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -n "${HLLM_ROOT:-}" ]]; then
    script_dir="${HLLM_ROOT}/code"
elif [[ ! -f "${script_dir}/run_amazon_id_baseline.sh" ]]; then
    if [[ -f "${SLURM_SUBMIT_DIR:-}/code/run_amazon_id_baseline.sh" ]]; then
        script_dir="${SLURM_SUBMIT_DIR}/code"
    elif [[ -f "${SLURM_SUBMIT_DIR:-}/run_amazon_id_baseline.sh" ]]; then
        script_dir="${SLURM_SUBMIT_DIR}"
    else
        echo "Cannot locate repository. Set HLLM_ROOT to its absolute path." >&2
        exit 1
    fi
fi
exec bash "${script_dir}/run_amazon_id_baseline.sh" sasrec "$@"
