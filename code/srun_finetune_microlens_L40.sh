#!/bin/bash

#SBATCH --job-name=finetune_microlens
#SBATCH --output=outputs/%x-%j.out
#SBATCH --error=outputs/%x-%j.err # I put this in directory `outputs`, if the directory doesn't exists, job will fail immediately

#SBATCH --partition=general # check the partitions available and switch if you need a longer job/ different resources 

#SBATCH --mail-type=ALL
#SBATCH --mail-user=lixiangl@andrew.cmu.edu

#SBATCH --gres=gpu:L40:4
#SBATCH --nodelist=babel-9-7,babel-13-37

#SBATCH --time=2-00:00:00
#SBATCH --mem=256G
#SBATCH --cpus-per-task 16
#SBATCH --nodes=1

# enter a config env
eval "$(conda shell.bash hook)"
conda activate hllm

run_name=model_microlens_proposed_batchszie_64_deepspeed_3_pretrain_5_epoch_April_11_2025_eval_SFT_data_HLLM-0.pth_20_percent_SFT
# run_name=model_microlens_proposed_batchszie_64_deepspeed_3_pretrain_5_epoch_march_25_2025_more_curated_eval_SFT_data_filtered_threshold_0p65_HLLM-0.pth_50_percent_SFT

sed -i "s/^clueweb_project: .*/clueweb_project: '$run_name'/" overall/LLM_deepspeed.yaml


checkpoint_dir=/data/user_data/lixiangl/HLLM_2/HLLM/model_microlens_continual_pretrain_batchszie_128_deepspeed_3_epochs_5_eval_different_data_further_curate_corpus_March_13_2025/HLLM-0.pth/pretrained/20_percent_SFT
pretrain_dir="/data/user_data/lixiangl/HLLM_2/HLLM/TinyLlama-1.1B-intermediate-step-1431k-3T/"

info_path="/data/user_data/lixiangl/HLLM_2/HLLM/information"
data_path="/data/user_data/lixiangl/HLLM_2/HLLM/dataset"

file_prefix="/data/user_data/lixiangl/HLLM_2/HLLM/code"

CUDA_VISIBLE_DEVICES=0,1,2,3 python3 ${file_prefix}/main.py \
    --config_file ${file_prefix}/overall/LLM_deepspeed.yaml HLLM/HLLM.yaml \
    --loss nce \
    --epochs 5 \
    --dataset microlens_20_percent \
    --train_batch_size 16 \
    --MAX_TEXT_LENGTH 256 \
    --MAX_ITEM_LIST_LENGTH 10 \
    --checkpoint_dir $checkpoint_dir \
    --optim_args.learning_rate 1e-4 \
    --item_pretrain_dir $pretrain_dir \
    --user_pretrain_dir $pretrain_dir \
    --data_path $data_path \
    --text_path $info_path \
    --text_keys '[\"title\",\"tag\",\"description\"]' \
    --val_only False \
    --finetune_clueweb True \
    --baseline_train False \
    --clueweb_pretrain False \
    --gen_relevance_score False \
    --gradient_checkpointing True \
    --stage 3 \
    --gen_relevance_score False \

