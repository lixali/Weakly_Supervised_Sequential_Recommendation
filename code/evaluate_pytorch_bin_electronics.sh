#!/bin/bash

#SBATCH --job-name=eval_electronics
#SBATCH --output=outputs/%x-%j.out
#SBATCH --error=outputs/%x-%j.err # I put this in directory `outputs`, if the directory doesn't exists, job will fail immediately

#SBATCH --partition=preempt # check the partitions available and switch if you need a longer job/ different resources 

#SBATCH --mail-type=ALL
#SBATCH --mail-user=lixiangl@andrew.cmu.edu

#SBATCH --gres=gpu:A100_40GB:4
#SBATCH --time=2-00:00:00
#SBATCH --mem=128G
#SBATCH --cpus-per-task 16
#SBATCH --nodes=1

# enter a config env
checkpoint_dir="/data/user_data/lixiangl/HLLM_2/HLLM/model_pixelrec_continual_pretrain_batchszie_64_deepspeed_3_25_percent_model_filtered_threhold_0p35_march_5_2025/HLLM-0.pth/pretrained/25_percent_SFT/HLLM-0.pth/"
# checkpoint_dir="../model_clueweb_sbatch_epoch_6_pretrain_script_batchszie_64_deepspeed_3/HLLM-0.pth/pretrained/"

pretrain_dir="/data/user_data/lixiangl/HLLM_2/HLLM/TinyLlama-1.1B-intermediate-step-1431k-3T/"

info_path="/data/user_data/lixiangl/HLLM_2/HLLM/information"
# info_path="best_model_epoch_13/sorted_pixelrec_only/information/"
data_path="/data/user_data/lixiangl/HLLM_2/HLLM/dataset"
# data_path="best_model_epoch_13/sorted_pixelrec_only/dataset/"
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
    --text_keys '[\"title\",\"description\"]' \
    --text_path $info_path \
    --val_only True \
    --finetune_clueweb False \
    --clueweb_pretrain False \
    --baseline_train False \
    --item_pretrain_dir $pretrain_dir \
    --user_pretrain_dir $pretrain_dir \
    --gradient_checkpointing True \
    --stage 3 \
    --gen_relevance_score False \
