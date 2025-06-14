#!/bin/bash


#SBATCH --job-name=SASRec_pixelrec200k_5_percent_data
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


file_prefix="/data/user_data/lixiangl/HLLM_2/HLLM/code"
checkpoint_dir=/data/user_data/lixiangl/HLLM_2/HLLM/model_SASRec_baseline_5_percent_data_batchszie_32_x_4_deepspeed_3_epochs_50_May_15_2025/
saved_folder="saved_5_percent_pixel200k_batchsize_16"
sed -i "s/^checkpoint_dir: .*/checkpoint_dir: '$checkpoint_dir'/" overall/ID.yaml

CUDA_VISIBLE_DEVICES=0,1,2,3 python3 main.py \
    --config_file ${file_prefix}/IDNet/sasrec.yaml overall/ID.yaml \
    --optim_args.learning_rate 1e-4 \
    --loss nce \
    --train_batch_size 32 \
    --MAX_ITEM_LIST_LENGTH 10 \
    --epochs 50 \
    --dataset Pixel200K_5_percent \
    --stopping_step 5 \
    --show_progress True \
    --update_interval 100 \
    --text_keys '[\"title\",\"tag\",\"description\"]' \
    --val_only False \
    --finetune_clueweb False \
    --gen_relevance_score False \
    --baseline_train True \
    --clueweb_pretrain False \
    --gradient_checkpointing True \
    --stage 3 \