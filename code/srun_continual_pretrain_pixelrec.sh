#!/bin/bash

#SBATCH --job-name=clueweb_pretrain
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


# checkpoint_dir="/data/user_data/lixiangl/HLLM_2/HLLM/model_clueweb_sbatch_epoch_6_pretrain_script_batchszie_64_deepspeed_3_200k_seed_wo_nltk_out_of_bounds_first_run/"
# checkpoint_dir="/data/user_data/lixiangl/HLLM_2/HLLM/model_clueweb_sbatch_epoch_6_pretrain_script_batchszie_64_deepspeed_3_400k_seed_wo_nltk/"
# checkpoint_dir="/data/user_data/lixiangl/HLLM_2/HLLM/model_clueweb_sbatch_epoch_6_pretrain_script_batchszie_64_deepspeed_3_second_run/"


run_name="model_pixelrec_continual_pretrain_batchszie_64_deepspeed_3_25_percent_model_filtered_threhold_0p35_5_epochs_march_8_2025_correct_hllm_data_pool"
sed -i "s/^clueweb_project: .*/clueweb_project: '$run_name'/" overall/LLM_deepspeed.yaml

checkpoint_dir="/data/user_data/lixiangl/HLLM_2/HLLM/${run_name}"

pretrain_dir="/data/user_data/lixiangl/HLLM_2/HLLM/TinyLlama-1.1B-intermediate-step-1431k-3T/"



info_path="/data/user_data/lixiangl/HLLM_2/HLLM/information"
data_path="/data/user_data/lixiangl/HLLM_2/HLLM/dataset"

file_prefix="/data/user_data/lixiangl/HLLM_2/HLLM/code"

CUDA_VISIBLE_DEVICES=0,1,2,3 python3 ${file_prefix}/main.py \
    --config_file ${file_prefix}/overall/LLM_deepspeed.yaml HLLM/HLLM.yaml \
    --loss nce \
    --epochs 5 \
    --dataset epoch_6_clueweb300k_train_25_percent_benchmark_filtered_p35_hllm_correct_data_pool \
    --dataset_for_eval Pixel200K_filtered_25_percent \
    --train_batch_size 32 \
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
    --finetune_clueweb False \
    --baseline_train False \
    --clueweb_pretrain True \
    --gen_relevance_score False \
    --gradient_checkpointing True \
    --stage 3 \
    # --text_keys '[\"title\",\"tag\",\"description\"]' \

