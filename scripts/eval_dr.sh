#!/bin/bash
#SBATCH --job-name=eval_dr_openmatch
#SBATCH --output=logs/%x-%j.out
#SBATCH -e logs/%x-%j.err
#SBATCH --partition=general
#SBATCH --gres=gpu:A6000:2
#SBATCH --cpus-per-task=4
#SBATCH --mem=200G
#SBATCH --time=2-00:00:00

eval "$(conda shell.bash hook)"
conda activate openmatch

model_to_eval=$1
model_name=$(basename "$model_to_eval")

text_length=2048
n_gpus=2

embeddings_out="./data/embeddings/dev/$model_name"
run_save="./results/$model_name"

dev_queries=./data/marco_documents_processed/dev.query.txt
dev_qrels=./data/marco_documents_processed/qrels.dev.tsv
corpus=./data/marco_documents_processed/corpus_firstp_$text_length.tsv

mkdir -p $run_save
mkdir -p $embeddings_out


accelerate launch --num_processes $n_gpus --multi_gpu --main_process_port 29777 OpenMatch/src/openmatch/driver/build_index.py  \
    --output_dir $embeddings_out \
    --model_name_or_path $model_to_eval \
    --per_device_eval_batch_size 430  \
    --corpus_path $corpus \
    --doc_template "Title: <title> Text: <text>"  \
    --doc_column_names id,title,text  \
    --p_max_len $text_length  \
    --fp16  \
    --dataloader_num_workers 1 \
    --rope True

accelerate launch --num_processes $n_gpus --multi_gpu --main_process_port 29777 OpenMatch/src/openmatch/driver/retrieve.py  \
    --output_dir $embeddings_out  \
    --model_name_or_path $model_to_eval \
    --per_device_eval_batch_size 600  \
    --query_path $dev_queries \
    --query_template "<text>"  \
    --query_column_names id,text  \
    --q_max_len 32  \
    --fp16  \
    --trec_save_path $run_save/dev.trec \
    --dataloader_num_workers 1 \
    --use_gpu \
    --retrieve_depth 100 \
    --rope True

python OpenMatch/scripts/evaluate.py $dev_qrels $run_save/dev.trec > $run_save/dev.results