#!/usr/bin/env bash
# Shared Colab/single-node launcher. Use the sasrec or hstu wrapper.
set -euo pipefail
MODEL="${1:?Usage: run_amazon_id_baseline.sh sasrec|hstu}"
shift
if (( $# )); then
    echo "Use environment variables for overrides; see README_amazon_id_baselines.md." >&2
    exit 2
fi
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HLLM_ROOT="$(cd "${HLLM_ROOT:-${SCRIPT_DIR}/..}" && pwd)"
cd "${HLLM_ROOT}/code"
export HLLM_ROOT PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false

DATASET="${DATASET:-amazon_industrial_and_scientific_9251_users_52568_interactions}"
DATA_PATH="${DATA_PATH:-${HLLM_ROOT}/dataset}"
interaction_file="${DATA_PATH}/${DATASET}.csv"
GPU_IDS="${GPU_IDS:-0}"
NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
export CUDA_VISIBLE_DEVICES="${GPU_IDS}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
EPOCHS="${EPOCHS:-201}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-64}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-128}"
MAX_ITEM_LIST_LENGTH="${MAX_ITEM_LIST_LENGTH:-10}"
NUM_WORKERS="${NUM_WORKERS:-2}"
NUM_NEGATIVES="${NUM_NEGATIVES:-0}"
PRECISION="${PRECISION:-bf16-mixed}"
STAGE="${STAGE:-2}"
if [[ "${STAGE}" != 2 ]]; then
    echo "These ID baseline launchers require STAGE=2; stage-3 evaluation is not supported." >&2
    exit 2
fi
export PRETRAIN_DIR="${PRETRAIN_DIR:-${HLLM_ROOT}/TinyLlama-1.1B-intermediate-step-1431k-3T}"

case "${MODEL}" in
    sasrec)
        SASREC_VARIANT="${SASREC_VARIANT:-paper}"
        case "${SASREC_VARIANT}" in
            paper)
                config_file=IDNet/llama_id.yaml
                MODEL_ARGS=(--user_pretrain_dir "${PRETRAIN_DIR}" --user_llm_init False
                    --item_embed_dim 512 --use_ft_flash_attn False --gradient_checkpointing True)
                ;;
            standard)
                config_file=IDNet/sasrec.yaml
                MODEL_ARGS=()
                ;;
            *) echo "SASREC_VARIANT must be paper or standard." >&2; exit 2 ;;
        esac
        model_label="sasrec_${SASREC_VARIANT}"
        ;;
    hstu)
        config_file=IDNet/hstu.yaml
        model_label=hstu
        MODEL_ARGS=(--n_layers "${HSTU_LAYERS:-22}" --n_heads "${HSTU_HEADS:-32}"
            --item_embedding_size "${HSTU_ITEM_DIM:-2048}" --hstu_embedding_size "${HSTU_DIM:-2048}"
            --hidden_dropout_prob "${HSTU_DROPOUT:-0.5}" --attn_dropout_prob "${HSTU_DROPOUT:-0.5}")
        ;;
    *) echo "Unknown model: ${MODEL}; choose sasrec or hstu." >&2; exit 2 ;;
esac
RUN_NAME="${RUN_NAME:-model_${DATASET}_${model_label}_baseline}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-${HLLM_ROOT}/${RUN_NAME}/}"
echo "Starting Amazon ${model_label} launcher..."
echo "Interaction file: ${interaction_file}"
if [[ ! -f "${interaction_file}" ]]; then
    echo "Missing interaction file: ${interaction_file}" >&2
    echo "Upload the interaction CSV to dataset/ or set DATA_PATH and DATASET." >&2
    exit 1
fi

source "${SCRIPT_DIR}/setup_id_train_env.sh"
export PYTHON_BIN
if [[ "${model_label}" == sasrec_paper ]]; then
    bash "${SCRIPT_DIR}/download_sasrec_config.sh"
fi
echo "Checking training imports, CSV, and CUDA..."
"${PYTHON_BIN}" check_amazon_id_inputs.py "${interaction_file}" "${config_file}" \
    "${NPROC_PER_NODE}" "${PRECISION}"

ARGS=(
    --config_file overall/ID_deepspeed.yaml "${config_file}"
    --dataset "${DATASET}" --data_path "${DATA_PATH}"
    --epochs "${EPOCHS}" --stopping_step "${STOPPING_STEP:-10}"
    --train_batch_size "${TRAIN_BATCH_SIZE}" --eval_batch_size "${EVAL_BATCH_SIZE}"
    --MAX_ITEM_LIST_LENGTH "${MAX_ITEM_LIST_LENGTH}" --num_workers "${NUM_WORKERS}"
    --loss nce --num_negatives "${NUM_NEGATIVES}" --fix_temp True
    --optim_args.learning_rate "${LEARNING_RATE:-1e-3}" --optim_args.weight_decay "${WEIGHT_DECAY:-0.1}"
    --use_text False --use_gpu True --log_wandb False --show_progress True --update_interval 1
    --val_only False --finetune_clueweb False --gen_relevance_score False
    --baseline_train True --clueweb_pretrain False
    --checkpoint_dir "${CHECKPOINT_DIR}" --clueweb_project "${RUN_NAME}"
    --stage "${STAGE}" --precision "${PRECISION}"
    "${MODEL_ARGS[@]}"
)
echo "Checkpoint directory: ${CHECKPOINT_DIR}"
echo "GPUs: ${GPU_IDS}; processes: ${NPROC_PER_NODE}; precision: ${PRECISION}; stage: ${STAGE}"
echo "Batch size: ${TRAIN_BATCH_SIZE}; max sequence: ${MAX_ITEM_LIST_LENGTH}; epochs: ${EPOCHS}"
if [[ "${PREPARE_ONLY:-False}" == True ]]; then
    echo "Preparation complete. Run again without PREPARE_ONLY=True to train."
    exit 0
fi
mkdir -p "${CHECKPOINT_DIR}"
printf 'Launching:'
printf ' %q' "${PYTHON_BIN}" -m torch.distributed.run --standalone --nnodes=1 \
    --nproc_per_node="${NPROC_PER_NODE}" run.py "${ARGS[@]}"
printf '\n'
exec "${PYTHON_BIN}" -m torch.distributed.run --standalone --nnodes=1 \
    --nproc_per_node="${NPROC_PER_NODE}" run.py "${ARGS[@]}"
