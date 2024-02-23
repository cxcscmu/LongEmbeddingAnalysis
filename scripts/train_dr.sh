#!/bin/bash
#SBATCH --job-name=train_dr_openmatch
#SBATCH --output=logs/%x-%j.out
#SBATCH -e logs/%x-%j.err
#SBATCH --partition=long
#SBATCH --cpus-per-task=4
#SBATCH --mem=200G
#SBATCH --gres=gpu:A6000:4
#SBATCH --time=7-00:00:00

eval "$(conda shell.bash hook)"
conda activate openmatch

DATA_PATH="."

split=documents
text_length=2048
n_gpus=4

num_episodes=4

train_qrels=$DATA_PATH/data/marco_documents_processed/qrels.train.tsv
train_queries=$DATA_PATH/data/marco_documents_processed/train.query.txt
corpus=$DATA_PATH/data/marco_documents_processed/corpus_firstp_2048.tsv
negatives=$DATA_PATH/data/marco_documents_processed/train.negatives.tsv

initial_model=$1
trained_model_name=t5-base-marco-$split-$text_length

train_data_folder=$DATA_PATH/data/training_data/$trained_model_name
mkdir -p $train_data_folder

echo "########################################"
echo "Building initial data"
echo "########################################"

python OpenMatch/scripts/msmarco/build_train.py \
   --tokenizer_name $initial_model \
   --negative_file $negatives  \
   --qrels $train_qrels  \
   --queries $train_queries  \
   --collection $corpus \
   --truncate $text_length \
   --save_to $train_data_folder  \
   --doc_template "Title: <title> Text: <text>" \
   --n_sample 9


cat $train_data_folder/split*.jsonl > $train_data_folder/full.jsonl
rm $train_data_folder/split*.jsonl

line_count=$(wc -l $train_data_folder/full.jsonl | awk '{print $1}')
n_val=500
n_train=$((line_count - n_val))

echo $n_train

tail -n $n_val $train_data_folder/full.jsonl > $train_data_folder/val.jsonl
head -n $n_train $train_data_folder/full.jsonl > $train_data_folder/train.jsonl

rm $train_data_folder/full.jsonl

echo "########################################"
echo "Train + HN sampling loop - 4 episodes"
echo "########################################"

train_data=$train_data_folder/train.jsonl
valid_data=$train_data_folder/val.jsonl
output_path=$DATA_PATH/models/$trained_model_name

for ((i = 1; i <= num_episodes; i++)); do

    accelerate launch --num_processes $n_gpus --multi_gpu --main_process_port 29777 OpenMatch/src/openmatch/driver/train_dr.py  \
        --output_dir $output_path \
        --model_name_or_path $initial_model \
        --do_train \
        --save_steps 125  \
        --eval_steps 125  \
        --fp16 \
        --train_path $train_data  \
        --eval_path $valid_data  \
        --per_device_train_batch_size 128 \
        --per_device_eval_batch_size 4 \
        --train_n_passages 10  \
        --learning_rate 5e-6  \
        --q_max_len 32  \
        --p_max_len $text_length  \
        --num_train_epochs 2  \
        --report_to wandb \
        --logging_steps 10 \
        --run_name $trained_model_name \
        --evaluation_strategy steps \
        --dataloader_num_workers 4 \
        --rope True \
        --grad_cache True \
        --use_mapping_dataset True \
        --gc_p_chunk_size 24 \
        --gc_q_chunk_size 24 \
        --negatives_x_device True 
    
    # Hard negative sampling - ANCE negative refresh
    # set variables for next training episode

    initial_model=$output_path

    embeddings_out=$DATA_PATH/data/embeddings/train/$trained_model_name
    run_save=$DATA_PATH/data/negatives/$trained_model_name

    trained_model_name=$trained_model_name-self-hn-$i
    train_data_folder=$DATA_PATH/data/training_data/$trained_model_name

    mkdir -p $run_save
    mkdir -p $embeddings_out

    accelerate launch --num_processes $n_gpus --multi_gpu --main_process_port 29777 OpenMatch/src/openmatch/driver/build_index.py  \
        --output_dir $embeddings_out \
        --model_name_or_path $output_path  \
        --per_device_eval_batch_size 430 \
        --corpus_path $corpus  \
        --doc_template "Title: <title> Text: <text>"  \
        --doc_column_names id,title,text  \
        --p_max_len $text_length  \
        --fp16  \
        --dataloader_num_workers 1

    accelerate launch --num_processes $n_gpus --multi_gpu --main_process_port 29777 OpenMatch/src/openmatch/driver/retrieve.py  \
        --output_dir $embeddings_out  \
        --model_name_or_path $output_path  \
        --per_device_eval_batch_size 600  \
        --query_path $train_queries  \
        --query_template "<text>"  \
        --query_column_names id,text  \
        --q_max_len 32  \
        --fp16  \
        --trec_save_path $run_save/train.trec \
        --dataloader_num_workers 1 \
        --use_gpu

    python OpenMatch/scripts/msmarco/build_hn.py  \
        --tokenizer_name $output_path \
        --hn_file $run_save/train.trec \
        --qrels $train_qrels \
        --queries $train_queries  \
        --collection $corpus  \
        --save_to $train_data_folder  \
        --doc_template "Title: <title> Text: <text>" \
        --n_sample 9 \
        --truncate $text_length

    cat $train_data_folder/*.hn.jsonl > $train_data_folder/full.jsonl
    rm $train_data_folder/*.hn.jsonl

    line_count=$(wc -l $train_data_folder/full.jsonl | awk '{print $1}')
    n_val=500
    n_train=$((line_count - n_val))

    echo $n_train

    tail -n $n_val $train_data_folder/full.jsonl > $train_data_folder/val.jsonl
    head -n $n_train $train_data_folder/full.jsonl > $train_data_folder/train.jsonl

    rm $train_data_folder/full.jsonl

    # Train again with the hard negatives - don't set the variable for the last one
    if [ $i -ne $num_episodes ]; then
        train_data=$train_data_folder/train.jsonl
        valid_data=$train_data_folder/val.jsonl
        output_path=$DATA_PATH/models/marco/$trained_model_name
    fi

done 
## Eval

model_to_eval=$output_path
model_name=$(basename "$model_to_eval")

embeddings_out="$DATA_PATH/data/embeddings/dev/$model_name"
run_save="$DATA_PATH/results/$model_name"

dev_queries=$DATA_PATH/data/marco_documents_processed/dev.query.txt
dev_qrels=$DATA_PATH/data/marco_documents_processed/qrels.dev.tsv

mkdir -p $run_save
mkdir -p $embeddings_out

# already have corpus embeddings from hard negatives
cp $DATA_PATH/data/embeddings/train/$model_name/embeddings.corpus.* $DATA_PATH/data/embeddings/dev/$model_name

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