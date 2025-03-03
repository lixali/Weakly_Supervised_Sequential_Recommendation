#!/bin/bash

#SBATCH --job-name=clueweb_pretrain
#SBATCH --output=outputs/%x-%j.out
#SBATCH --error=outputs/%x-%j.err # I put this in directory `outputs`, if the directory doesn't exists, job will fail immediately

#SBATCH --partition=general # check the partitions available and switch if you need a longer job/ different resources 

#SBATCH --mail-type=ALL
#SBATCH --mail-user=lixiangl@andrew.cmu.edu

#SBATCH --gres=gpu:A100_40GB:1
#SBATCH --time=2-00:00:00
#SBATCH --mem=128G
#SBATCH --cpus-per-task 16
#SBATCH --nodes=1

# enter a config env
eval "$(conda shell.bash hook)"
conda activate hllm

# checkpoint_dir="/data/user_data/lixiangl/HLLM_2/HLLM/model_clueweb_sbatch_pretrain_script_batchszie_64_deepspeed_3/HLLM-0.pth"

# checkpoint_dir="/data/datasets/hf_cache/sample//TinyLlama-1.1B-intermediate-step-1431k-3T/"
checkpoint_dir="/data/user_data/lixiangl/HLLM_2/HLLM/model_amazon_book_benchmark_batchszie_32_deepspeed_3_Feb_12_2025_1_epoch_30_percent_data/HLLM-0.pth"

# pretrain_dir="/data/user_data/lixiangl/HLLM_2/HLLM/tinyllama"
pretrain_dir="/data/datasets/hf_cache/sample/TinyLlama_redownload_Jan_9_2025/TinyLlama-1.1B-intermediate-step-1431k-3T/"

info_path="/data/user_data/lixiangl/HLLM_2/HLLM/information"
# info_path="best_model_epoch_13/sorted_pixelrec_only/information/"
data_path="/data/user_data/lixiangl/HLLM_2/HLLM/dataset"
# data_path="best_model_epoch_13/sorted_pixelrec_only/dataset/"
file_prefix="/data/user_data/lixiangl/HLLM_2/HLLM/code"

relevance_score_file=/data/user_data/lixiangl/fastText/amazon_clueweb_440k_save_best_epoch/sorted_amazon_only/score_ranking/amazon200k_relevance_score2.jsonl

CUDA_VISIBLE_DEVICES=0,1 python3 ${file_prefix}/main.py \
    --config_file ${file_prefix}/overall/LLM_deepspeed.yaml HLLM/HLLM.yaml \
    --loss nce \
    --epochs 5 \
    --dataset clueweb200k_fasttext440k_train \
    --train_batch_size 16 \
    --MAX_TEXT_LENGTH 256 \
    --MAX_ITEM_LIST_LENGTH 10 \
    --checkpoint_dir $checkpoint_dir \
    --optim_args.learning_rate 1e-4 \
    --text_keys '[\"title\",\"tag\",\"description\"]' \
    --text_path $info_path \
    --val_only False \
    --gen_relevance_score True \
    --relevance_score_file $relevance_score_file \
    --finetune_clueweb False \
    --baseline_train False \
    --clueweb_pretrain False \
    --item_pretrain_dir $pretrain_dir \
    --user_pretrain_dir $pretrain_dir \
    --gradient_checkpointing True \
    --stage 3 \
    # --text_keys '[\"title\",\"tag\",\"description\"]' \
