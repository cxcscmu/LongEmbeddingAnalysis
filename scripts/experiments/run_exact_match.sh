#!/bin/bash
#SBATCH --job-name=exact_match
#SBATCH --output=logs/%x-%j.out
#SBATCH -e logs/%x-%j.err
#SBATCH --partition=general
#SBATCH --cpus-per-task=4
#SBATCH --mem=200G
#SBATCH --gres=gpu:A6000:1
#SBATCH --time=2-00:00:00

eval "$(conda shell.bash hook)"
conda activate openmatch

model_to_eval=$1
model_name=$(basename "$model_to_eval")

echo "Evaluating model $model_name on exact match task"

outpath=./data/exact_match_experiment/corpus/

mkdir -p $outpath

# tail -30000 ./data/marco_documents_processed/corpus_firstp_2048.tsv > $outpath/full.tsv

# python ./data/process_corpus_exact_match.py $outpath $model_to_eval

text_length=2048
n_gpus=1

for i in {0..9}; do
    embeddings_path=./data/exact_match_experiment/embeddings/$model_name/$i
    mkdir -p $embeddings_path
    
    accelerate launch --num_processes $n_gpus OpenMatch/src/openmatch/driver/build_index.py  \
        --output_dir $embeddings_path \
        --model_name_or_path $model_to_eval \
        --per_device_eval_batch_size 450  \
        --corpus_path $outpath/$i.tsv \
        --doc_template "Title: <title> Text: <text>"  \
        --doc_column_names id,title,text  \
        --p_max_len $text_length  \
        --fp16  \
        --dataloader_num_workers 1 \
        --rope True
done

embeddings_path=./data/exact_match_experiment/embeddings/$model_name/full
mkdir $embeddings_path
accelerate launch --num_processes $n_gpus OpenMatch/src/openmatch/driver/build_index.py  \
    --output_dir $embeddings_path \
    --model_name_or_path $model_to_eval \
    --per_device_eval_batch_size 450  \
    --corpus_path $outpath/full.tsv \
    --doc_template "Title: <title> Text: <text>"  \
    --doc_column_names id,title,text  \
    --p_max_len $text_length  \
    --fp16  \
    --dataloader_num_workers 1 \
    --rope True

mkdir -p ./data/exact_match_experiment/plots/$model_name

python ./data/exact_match_experiment/plot_exact_match_violin.py ./data/exact_match_experiment/embeddings/$model_name ./data/exact_match_experiment/plots/$model_name