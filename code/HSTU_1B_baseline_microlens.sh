#!/bin/bash


#SBATCH --job-name=HSTU_1B_microlens_15_percent
#SBATCH --output=outputs/%x-%j.out
#SBATCH --error=outputs/%x-%j.err # I put this in directory `outputs`, if the directory doesn't exists, job will fail immediately

#SBATCH --partition=general # check the partitions available and switch if you need a longer job/ different resources 

#SBATCH --mail-type=ALL
#SBATCH --mail-user=lixiangl@andrew.cmu.edu
#SBATCH --gres=gpu:4
#SBATCH --constraint=A100_40GB


#SBATCH --time=2-00:00:00
#SBATCH --mem=128G
#SBATCH --cpus-per-task 16
#SBATCH --nodes=1


file_prefix="/data/user_data/lixiangl/HLLM_2/HLLM/code"
checkpoint_dir=/data/user_data/lixiangl/HLLM_2/HLLM/model_HSTU_1B_microlens_baseline_15_percent_data_batchszie_32_x_4_deepspeed_3_epochs_50_May_10_2025_second_run/


# escaped_dir=$(printf '%s\n' "$checkpoint_dir" | sed 's/[\/&]/\\&/g')
# sed -i "s/^checkpoint_dir: .*/checkpoint_dir: '$escaped_dir'/" overall/ID.yaml

# CUDA_VISIBLE_DEVICES=0 python3 -m pdb main.py \
CUDA_VISIBLE_DEVICES=0,1,2,3 python3 main.py \
    --config_file IDNet/hstu.yaml overall/ID_deepspeed.yaml \
    --optim_args.learning_rate 1e-3 \
    --loss nce \
    --train_batch_size 32 \
    --MAX_ITEM_LIST_LENGTH 50 \
    --epochs 201 \
    --dataset microlens_15_percent \
    --hidden_dropout_prob 0.5 \
    --attn_dropout_prob 0.5 \
    --n_layers 22 \
    --n_heads 32 \
    --item_embedding_size 2048 \
    --hstu_embedding_size 2048 \
    --fix_temp True \
    --text_keys '[\"title\",\"tag\",\"description\"]' \
    --show_progress True \
    --update_interval 100 \
    --checkpoint_dir ${checkpoint_dir} \
    --stopping_step 10 \
    --val_only False \
    --finetune_clueweb False \
    --gen_relevance_score False \
    --baseline_train True \
    --clueweb_pretrain False \
    --gradient_checkpointing True \
    --num_negatives 512 \
    # --stage 3 \
