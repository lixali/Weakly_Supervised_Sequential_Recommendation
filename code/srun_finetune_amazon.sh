#!/bin/bash

#SBATCH --job-name=amazon_finetune
#SBATCH --output=outputs/%x-%j.out
#SBATCH --error=outputs/%x-%j.err # I put this in directory `outputs`, if the directory doesn't exists, job will fail immediately

#SBATCH --partition=general # check the partitions available and switch if you need a longer job/ different resources 

#SBATCH --mail-type=ALL
#SBATCH --mail-user=lixiangl@andrew.cmu.edu

#SBATCH --gres=gpu:A100_40GB:4
#SBATCH --time=2-00:00:00
#SBATCH --mem=256G
#SBATCH --cpus-per-task 16
#SBATCH --nodes=1

# enter a config env
eval "$(conda shell.bash hook)"
conda activate hllm

checkpoint_dir="/data/user_data/lixiangl/HLLM_2/HLLM/model_200k_clueweb_amazon_pretrain_script_batchszie_128_deepspeed_3_pretrained_fasttext440k/HLLM-0.pth/pretrained/10_percent/"
# checkpoint_dir="/data/user_data/lixiangl/HLLM_2/HLLM/model_200k_clueweb_amazon_pretrain_script_batchszie_128_deepspeed_3/HLLM-0.pth/pretrained/"

# pretrain_dir="/data/user_data/lixiangl/HLLM_2/HLLM/tinyllama"
# pretrain_dir="/data/user_data/lixiangl/HLLM_2/HLLM/model_clueweb_sbatch_pretrain_script_batchszie_64_deepspeed_3/HLLM-0.pth/"
pretrain_dir="/data/datasets/hf_cache/sample/TinyLlama_redownload_Jan_9_2025/TinyLlama-1.1B-intermediate-step-1431k-3T/"
info_path="/data/user_data/lixiangl/HLLM_2/HLLM/information"
data_path="/data/user_data/lixiangl/HLLM_2/HLLM/dataset"

file_prefix="/data/user_data/lixiangl/HLLM_2/HLLM/code"
CUDA_VISIBLE_DEVICES=0,1,2,3 python3 ${file_prefix}/main.py \
    --config_file ${file_prefix}/overall/LLM_deepspeed.yaml HLLM/HLLM.yaml \
    --loss nce \
    --epochs 5 \
    --dataset amazon_books_10_percent \
    --train_batch_size 16 \
    --MAX_TEXT_LENGTH 256 \
    --MAX_ITEM_LIST_LENGTH 10 \
    --checkpoint_dir $checkpoint_dir \
    --optim_args.learning_rate 1e-4 \
    --item_pretrain_dir $pretrain_dir \
    --user_pretrain_dir $pretrain_dir \
    --data_path $data_path \
    --text_path $info_path \
    --text_keys '[\"title\",\"description\"]' \
    --val_only False \
    --finetune_clueweb True \
    --gen_relevance_score False \
    --baseline_train False \
    --clueweb_pretrain False \
    --gradient_checkpointing True \
    --stage 3 \
    # --text_keys '[\"title\",\"tag\",\"description\"]' \

