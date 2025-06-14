#!/bin/bash


#SBATCH --job-name=SASRec_1B_pixelrec_5_percent_data
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
checkpoint_dir=/data/user_data/lixiangl/HLLM_2/HLLM/model_SASRec_1B_pixelrec200k_baseline_5_percent_data_batchszie_32_x_4_deepspeed_3_epochs_50_learning_rate_e3_removing_negative_num_May_15_2025/
# sed -i "s#^checkpoint_dir: .*#checkpoint_dir: '$checkpoint_dir'#" overall/ID.yaml
pretrain_dir="/data/user_data/lixiangl/HLLM_2/HLLM/TinyLlama-1.1B-intermediate-step-1431k-3T/"

CUDA_VISIBLE_DEVICES=0,1,2,3 python3 main.py \
    --config_file IDNet/llama_id.yaml overall/ID_deepspeed.yaml \
    --optim_args.learning_rate 1e-3 \
    --loss nce \
    --train_batch_size 32 \
    --MAX_ITEM_LIST_LENGTH 50 \
    --epochs 201 \
    --dataset Pixel200K_5_percent \
    --item_embed_dim 512 \
    --show_progress True \
    --update_interval 100 \
    --text_keys '[\"title\",\"tag\",\"description\"]' \
    --fix_temp True \
    --optim_args.weight_decay 0.1 \
    --user_pretrain_dir ${pretrain_dir} \
    --checkpoint_dir ${checkpoint_dir} \
    --stopping_step 10 \
    --val_only False \
    --finetune_clueweb False \
    --gen_relevance_score False \
    --baseline_train True \
    --clueweb_pretrain False \
    # --num_negatives 512 \
