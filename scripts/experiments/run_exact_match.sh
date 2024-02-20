#!/bin/bash

#SBATCH --job-name=a_exact_match_encode

# The line below writes to a logs dir inside the one where sbatch was called
# %x will be replaced by the job name, and %j by the job id

#SBATCH --output=logs/%x-%j.out
#SBATCH -e logs/%x-%j.err
#SBATCH -n 1 # Number of tasks
#SBATCH --cpus-per-task 12 # number cpus (threads) per task

# 327680
#SBATCH --mem=200000 # Memory - Use up to 2GB per requested CPU as a rule of thumb
#SBATCH --time=0 # No time limit

# You can also change the number of requested GPUs
# replace the XXX with nvidia_a100-pcie-40gb or nvidia_a100-sxm4-40gb
# replace the YYY with the number of GPUs that you need, 1 to 8 PCIe or 1 to 4 SXM4

#SBATCH --gres=gpu:nvidia_a100-sxm4-40gb:1

eval "$(conda shell.bash hook)"
conda activate openmatch

model_to_eval=/user/home/jcoelho/clueweb-structured-retrieval/models/marco/t5-base-marco-2048-v3-scaled-dr-pretrain-v2-meanpool

split=documents


text_length=2048
n_gpus=1


for i in {0..10}; do
    mkdir /user/home/jcoelho/clueweb-structured-retrieval/marco/documents/small_eval_set/exact_match/embeddings/10-evaluation-dots/mean-pool/$i
    
    /home/jcoelho/.conda/envs/openmatch/bin/accelerate launch --num_processes $n_gpus OpenMatch/src/openmatch/driver/build_index.py  \
        --output_dir /user/home/jcoelho/clueweb-structured-retrieval/marco/documents/small_eval_set/exact_match/embeddings/10-evaluation-dots/mean-pool/$i \
        --model_name_or_path $model_to_eval \
        --per_device_eval_batch_size 450  \
        --corpus_path /user/home/jcoelho/clueweb-structured-retrieval/marco/documents/small_eval_set/exact_match/10-evaluation-dots/$i.tsv \
        --doc_template "Title: <title> Text: <text>"  \
        --doc_column_names id,title,text  \
        --p_max_len $text_length  \
        --fp16  \
        --dataloader_num_workers 1
done
