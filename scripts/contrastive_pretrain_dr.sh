#!/bin/bash

#SBATCH --job-name=build_pretrain_data

# The line below writes to a logs dir inside the one where sbatch was called
# %x will be replaced by the job name, and %j by the job id

#SBATCH --output=logs/%x-%j.out
#SBATCH -e logs/%x-%j.err
#SBATCH -n 1 # Number of tasks
#SBATCH --cpus-per-task 12 # number cpus (threads) per task

# 327680
#SBATCH --mem=200000 # Memory - Use up to 2GB per requested CPU as a rule of thumb
#SBATCH --time=0 # No time limit

#SBATCH --gres=gpu:nvidia_a100-sxm4-40gb:4

eval "$(conda shell.bash hook)"
conda activate openmatch

split=documents
text_length=2048
n_gpus=4

initial_model=./models/t5-base-marco-2048-v3-scaled #-marco-pretrain

first_trained_model_name=t5-base-marco-2048-v3-scaled-dr-pretrain-v3-lq

corpus=./marco/$split/corpus_firstp_2048.tsv

initial_data_save_folder=./marco/$split/pretrain_data/smaller/
mkdir -p $initial_data_save_folder

python OpenMatch/scripts/msmarco/build_pretrain.py \
   --tokenizer_name $initial_model \
   --collection $corpus \
   --truncate $text_length \
   --save_to $initial_data_save_folder 

cat $initial_data_save_folder/split*.jsonl > $initial_data_save_folder/full.jsonl
rm $initial_data_save_folder/split*.jsonl

line_count=$(wc -l $initial_data_save_folder/full.jsonl | awk '{print $1}')
n_val=30000
n_train=$((line_count - n_val))

echo $n_train

tail -n $n_val $initial_data_save_folder/full.jsonl > $initial_data_save_folder/val.jsonl
head -n $n_train $initial_data_save_folder/full.jsonl > $initial_data_save_folder/train.jsonl

rm $initial_data_save_folder/full.jsonl

first_model_output_path=./models/marco/$first_trained_model_name
train_data=$initial_data_save_folder/train.jsonl

/home/jcoelho/.conda/envs/openmatch/bin/accelerate launch --num_processes $n_gpus --multi_gpu OpenMatch/src/openmatch/driver/pretrain_dr.py  \
    --output_dir $first_model_output_path \
    --model_name_or_path $initial_model \
    --do_train \
    --save_steps 2000 \
    --fp16 \
    --train_path $train_data  \
    --per_device_train_batch_size 128 \
    --per_device_eval_batch_size 4 \
    --learning_rate 5e-6  \
    --q_max_len $text_length \
    --p_max_len $text_length \
    --num_train_epochs 1  \
    --report_to wandb \
    --logging_steps 1 \
    --run_name $first_trained_model_name \
    --dataloader_num_workers 4 \
    --rope True \
    --grad_cache True \
    --use_mapping_dataset True \
    --gc_p_chunk_size 24 \
    --gc_q_chunk_size 24 \
    --negatives_x_device True \
    --pretrain_strategies crop 


