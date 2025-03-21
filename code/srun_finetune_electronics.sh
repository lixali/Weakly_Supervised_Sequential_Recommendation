#!/bin/bash

#SBATCH --job-name=finetune_electronics
#SBATCH --output=outputs/%x-%j.out
#SBATCH --error=outputs/%x-%j.err # I put this in directory `outputs`, if the directory doesn't exists, job will fail immediately

#SBATCH --partition=general # check the partitions available and switch if you need a longer job/ different resources 

#SBATCH --mail-type=ALL
#SBATCH --mail-user=lixiangl@andrew.cmu.edu

#SBATCH --gres=gpu:A100_40GB:4
#SBATCH --time=2-00:00:00
#SBATCH --mem=128G
#SBATCH --cpus-per-task 16
#SBATCH --nodes=1

# enter a config env
eval "$(conda shell.bash hook)"
conda activate hllm

run_name="model_electronics_proposed_batchszie_64_deepspeed_3_small2_pretrain_2_epoch_march_6_2025_HLLM-0.pth_25_percent_SFT"

sed -i "s/^clueweb_project: .*/clueweb_project: '$run_name'/" overall/LLM_deepspeed.yaml


checkpoint_dir="/data/user_data/lixiangl/HLLM_2/HLLM/model_electronics_continual_pretrain_batchszie_64_deepspeed_3_small2_first_100_words_2_epochs_March_6_2025/HLLM-0.pth/pretrained/25_percent_SFT/"
pretrain_dir="/data/user_data/lixiangl/HLLM_2/HLLM/TinyLlama-1.1B-intermediate-step-1431k-3T/"

info_path="/data/user_data/lixiangl/HLLM_2/HLLM/information"
data_path="/data/user_data/lixiangl/HLLM_2/HLLM/dataset"

file_prefix="/data/user_data/lixiangl/HLLM_2/HLLM/code"

CUDA_VISIBLE_DEVICES=0,1,2,3 python3 ${file_prefix}/main.py \
    --config_file ${file_prefix}/overall/LLM_deepspeed.yaml HLLM/HLLM.yaml \
    --loss nce \
    --epochs 5 \
    --dataset amazon_electronics_25_percent \
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
    --baseline_train False \
    --clueweb_pretrain False \
    --gen_relevance_score False \
    --gradient_checkpointing True \
    --stage 3 \
    --gen_relevance_score False \

