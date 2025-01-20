#!/bin/bash

# checkpoint_dir="/data/user_data/lixiangl/HLLM_2/HLLM/model_clueweb_sbatch_pretrain_script_batchszie_64_deepspeed_3/HLLM-0.pth"

# checkpoint_dir="/data/datasets/hf_cache/sample//TinyLlama-1.1B-intermediate-step-1431k-3T/"
checkpoint_dir="/data/user_data/lixiangl/HLLM_2/HLLM/model_benchmark_batchszie_64_deepspeed_3_Jan_16_2025_need_training_loss_vs_iteration/HLLM-0.pth"
# pretrain_dir="/data/user_data/lixiangl/HLLM_2/HLLM/tinyllama"
pretrain_dir="/data/datasets/hf_cache/sample/TinyLlama_redownload_Jan_9_2025/TinyLlama-1.1B-intermediate-step-1431k-3T/"

info_path="/data/user_data/lixiangl/HLLM_2/HLLM/information"
# info_path="best_model_epoch_13/sorted_pixelrec_only/information/"
data_path="/data/user_data/lixiangl/HLLM_2/HLLM/dataset"
# data_path="best_model_epoch_13/sorted_pixelrec_only/dataset/"
file_prefix="/data/user_data/lixiangl/HLLM_2/HLLM/code"

relevance_score_file=${file_prefix}/epoch_6_clueweb90k_relevance_score.jsonl

CUDA_VISIBLE_DEVICES=0 python3 ${file_prefix}/main.py \
    --config_file ${file_prefix}/overall/LLM_deepspeed.yaml HLLM/HLLM.yaml \
    --loss nce \
    --epochs 5 \
    --dataset epoch_6_clueweb90k_train \
    --train_batch_size 16 \
    --MAX_TEXT_LENGTH 256 \
    --MAX_ITEM_LIST_LENGTH 10 \
    --checkpoint_dir $checkpoint_dir \
    --optim_args.learning_rate 1e-4 \
    --text_keys '[\"description\"]' \
    --text_path $info_path \
    --val_only True \
    --gen_relevance_score True \
    --relevance_score_file $relevance_score_file \
    --finetune_only False \
    --item_pretrain_dir $pretrain_dir \
    --user_pretrain_dir $pretrain_dir \
    --gradient_checkpointing True \
    --stage 3 \
    # --text_keys '[\"title\",\"tag\",\"description\"]' \
