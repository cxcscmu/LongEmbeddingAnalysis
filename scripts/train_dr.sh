#!/bin/bash

#SBATCH --job-name=marco-docs-firstp

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

#SBATCH --gres=gpu:nvidia_a100-pcie-40gb:4

eval "$(conda shell.bash hook)"
conda activate openmatch

split=documents
text_length=2048
n_gpus=4

initial_model=./models/t5-base-marco-2048-v3-scaled/ 

first_trained_model_name=t5-base-no-pretrain-$split-firstp-$text_length

negatives=./marco/$split/train.negatives.tsv

train_qrels=./marco/$split/qrels.train.tsv
train_queries=./marco/$split/train.query.filtered.txt
corpus=./marco/$split/corpus_firstp_2048.tsv

#initial_data_save_folder=./marco/$split/processed_data/$first_trained_model_name
#skip data prep, re-use 2k
#initial_data_save_folder=./marco/$split/processed_data/t5-base-documents-firstp-marco-pretrain-v2-2048

mkdir -p $initial_data_save_folder

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
   --save_to $initial_data_save_folder  \
   --doc_template "Title: <title> Text: <text>" \
   --n_sample 9


cat $initial_data_save_folder/split*.jsonl > $initial_data_save_folder/full.jsonl
rm $initial_data_save_folder/split*.jsonl

line_count=$(wc -l $initial_data_save_folder/full.jsonl | awk '{print $1}')
n_val=500
n_train=$((line_count - n_val))

echo $n_train

tail -n $n_val $initial_data_save_folder/full.jsonl > $initial_data_save_folder/val.jsonl
head -n $n_train $initial_data_save_folder/full.jsonl > $initial_data_save_folder/train.jsonl

rm $initial_data_save_folder/full.jsonl

# Train model with initial negatives
first_model_output_path=./models/marco/$first_trained_model_name
train_data=$initial_data_save_folder/train.jsonl
valid_data=$initial_data_save_folder/val.jsonl

/home/jcoelho/.conda/envs/openmatch/bin/accelerate launch --num_processes $n_gpus --multi_gpu OpenMatch/src/openmatch/driver/train_dr.py  \
    --output_dir $first_model_output_path \
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
    --run_name $first_trained_model_name \
    --evaluation_strategy steps \
    --dataloader_num_workers 4 \
    --rope True \
    --grad_cache True \
    --use_mapping_dataset True \
    --gc_p_chunk_size 24 \
    --gc_q_chunk_size 24 \
    --negatives_x_device True 
   
# HN Sampling

model=$first_model_output_path
embeddings_out=./marco/$split/embeddings/$first_trained_model_name
run_save=./marco/$split/negatives/$first_trained_model_name

second_trained_model_name=$first_trained_model_name-self-hn-1
second_trained_model_data_path=./marco/$split/processed_data/$second_trained_model_name

mkdir $run_save

/home/jcoelho/.conda/envs/openmatch/bin/accelerate launch --num_processes $n_gpus --multi_gpu OpenMatch/src/openmatch/driver/build_index.py  \
    --output_dir $embeddings_out \
    --model_name_or_path $model  \
    --per_device_eval_batch_size 430 \
    --corpus_path $corpus  \
    --doc_template "Title: <title> Text: <text>"  \
    --doc_column_names id,title,text  \
    --q_max_len 32  \
    --p_max_len $text_length  \
    --fp16  \
    --dataloader_num_workers 1

/home/jcoelho/.conda/envs/openmatch/bin/accelerate launch --num_processes $n_gpus --multi_gpu OpenMatch/src/openmatch/driver/retrieve.py  \
    --output_dir $embeddings_out  \
    --model_name_or_path $model  \
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
    --tokenizer_name $model  \
    --hn_file $run_save/train.trec \
    --qrels $train_qrels \
    --queries $train_queries  \
    --collection $corpus  \
    --save_to $second_trained_model_data_path  \
    --doc_template "Title: <title> Text: <text>" \
    --n_sample 9 \
    --truncate $text_length

cat $second_trained_model_data_path/*.hn.jsonl > $second_trained_model_data_path/full.jsonl
rm $second_trained_model_data_path/*.hn.jsonl

line_count=$(wc -l $second_trained_model_data_path/full.jsonl | awk '{print $1}')
n_val=500
n_train=$((line_count - n_val))

echo $n_train

tail -n $n_val $second_trained_model_data_path/full.jsonl > $second_trained_model_data_path/val.jsonl
head -n $n_train $second_trained_model_data_path/full.jsonl > $second_trained_model_data_path/train.jsonl

rm $second_trained_model_data_path/full.jsonl

# Train again with the hard negatives

model_to_train=./models/marco/$first_trained_model_name
train_data=$second_trained_model_data_path/train.jsonl
valid_data=$second_trained_model_data_path/val.jsonl

second_nodel_output_path=./models/marco/$second_trained_model_name

/home/jcoelho/.conda/envs/openmatch/bin/accelerate launch --num_processes $n_gpus --multi_gpu OpenMatch/src/openmatch/driver/train_dr.py  \
    --output_dir $second_nodel_output_path \
    --model_name_or_path $model_to_train \
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
    --run_name $second_trained_model_name \
    --evaluation_strategy steps \
    --dataloader_num_workers 4 \
    --rope True \
    --grad_cache True \
    --use_mapping_dataset True \
    --gc_p_chunk_size 24 \
    --gc_q_chunk_size 24 \
    --negatives_x_device True 
    

# HN Sampling

model=$second_nodel_output_path
embeddings_out=./marco/$split/embeddings/$second_trained_model_name
run_save=./marco/$split/negatives/$second_trained_model_name

third_trained_model_name=$second_trained_model_name-self-hn-2
third_trained_model_data_path=./marco/$split/processed_data/$third_trained_model_name

mkdir $run_save

/home/jcoelho/.conda/envs/openmatch/bin/accelerate launch --num_processes $n_gpus --multi_gpu OpenMatch/src/openmatch/driver/build_index.py  \
    --output_dir $embeddings_out \
    --model_name_or_path $model  \
    --per_device_eval_batch_size 430 \
    --corpus_path $corpus  \
    --doc_template "Title: <title> Text: <text>"  \
    --doc_column_names id,title,text  \
    --q_max_len 32  \
    --p_max_len $text_length  \
    --fp16  \
    --dataloader_num_workers 1

/home/jcoelho/.conda/envs/openmatch/bin/accelerate launch --num_processes $n_gpus --multi_gpu OpenMatch/src/openmatch/driver/retrieve.py  \
    --output_dir $embeddings_out  \
    --model_name_or_path $model  \
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
    --tokenizer_name $model  \
    --hn_file $run_save/train.trec \
    --qrels $train_qrels \
    --queries $train_queries  \
    --collection $corpus  \
    --save_to $third_trained_model_data_path  \
    --doc_template "Title: <title> Text: <text>" \
    --n_sample 9 \
    --truncate $text_length

cat $third_trained_model_data_path/*.hn.jsonl > $third_trained_model_data_path/full.jsonl
rm $third_trained_model_data_path/*.hn.jsonl

line_count=$(wc -l $third_trained_model_data_path/full.jsonl | awk '{print $1}')
n_val=500
n_train=$((line_count - n_val))

echo $n_train

tail -n $n_val $third_trained_model_data_path/full.jsonl > $third_trained_model_data_path/val.jsonl
head -n $n_train $third_trained_model_data_path/full.jsonl > $third_trained_model_data_path/train.jsonl

rm $third_trained_model_data_path/full.jsonl


# Train again with the hard negatives

model_to_train=./models/marco/$second_trained_model_name
train_data=$third_trained_model_data_path/train.jsonl
valid_data=$third_trained_model_data_path/val.jsonl

third_nodel_output_path=./models/marco/$third_trained_model_name

/home/jcoelho/.conda/envs/openmatch/bin/accelerate launch --num_processes $n_gpus --multi_gpu OpenMatch/src/openmatch/driver/train_dr.py  \
    --output_dir $third_nodel_output_path \
    --model_name_or_path $model_to_train \
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
    --num_train_epochs 2 \
    --report_to wandb \
    --logging_steps 10 \
    --run_name $third_trained_model_name \
    --evaluation_strategy steps \
    --dataloader_num_workers 4 \
    --rope True \
    --grad_cache True \
    --use_mapping_dataset True \
    --gc_p_chunk_size 24 \
    --gc_q_chunk_size 24 \
    --negatives_x_device True



# HN Sampling

model=$third_nodel_output_path
embeddings_out=./marco/$split/embeddings/$third_trained_model_name
run_save=./marco/$split/negatives/$third_trained_model_name

fourth_trained_model_name=$third_trained_model_name-self-hn-3
fourth_trained_model_data_path=./marco/$split/processed_data/$fourth_trained_model_name

mkdir $run_save

/home/jcoelho/.conda/envs/openmatch/bin/accelerate launch --num_processes $n_gpus --multi_gpu OpenMatch/src/openmatch/driver/build_index.py  \
    --output_dir $embeddings_out \
    --model_name_or_path $model  \
    --per_device_eval_batch_size 430 \
    --corpus_path $corpus  \
    --doc_template "Title: <title> Text: <text>"  \
    --doc_column_names id,title,text  \
    --q_max_len 32  \
    --p_max_len $text_length  \
    --fp16  \
    --dataloader_num_workers 1

/home/jcoelho/.conda/envs/openmatch/bin/accelerate launch --num_processes $n_gpus --multi_gpu OpenMatch/src/openmatch/driver/retrieve.py  \
    --output_dir $embeddings_out  \
    --model_name_or_path $model  \
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
    --tokenizer_name $model  \
    --hn_file $run_save/train.trec \
    --qrels $train_qrels \
    --queries $train_queries  \
    --collection $corpus  \
    --save_to $fourth_trained_model_data_path  \
    --doc_template "Title: <title> Text: <text>" \
    --n_sample 9 \
    --truncate $text_length

cat $fourth_trained_model_data_path/*.hn.jsonl > $fourth_trained_model_data_path/full.jsonl
rm $fourth_trained_model_data_path/*.hn.jsonl

line_count=$(wc -l $fourth_trained_model_data_path/full.jsonl | awk '{print $1}')
n_val=500
n_train=$((line_count - n_val))

echo $n_train

tail -n $n_val $fourth_trained_model_data_path/full.jsonl > $fourth_trained_model_data_path/val.jsonl
head -n $n_train $fourth_trained_model_data_path/full.jsonl > $fourth_trained_model_data_path/train.jsonl

rm $fourth_trained_model_data_path/full.jsonl


# Train again with the hard negatives

model_to_train=./models/marco/$third_trained_model_name
train_data=$fourth_trained_model_data_path/train.jsonl
valid_data=$fourth_trained_model_data_path/val.jsonl

fourth_nodel_output_path=./models/marco/$fourth_trained_model_name

/home/jcoelho/.conda/envs/openmatch/bin/accelerate launch --num_processes $n_gpus --multi_gpu OpenMatch/src/openmatch/driver/train_dr.py  \
    --output_dir $fourth_nodel_output_path \
    --model_name_or_path $model_to_train \
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
    --num_train_epochs 2 \
    --report_to wandb \
    --logging_steps 10 \
    --run_name $fourth_trained_model_name \
    --evaluation_strategy steps \
    --dataloader_num_workers 4 \
    --rope True \
    --grad_cache True \
    --use_mapping_dataset True \
    --gc_p_chunk_size 24 \
    --gc_q_chunk_size 24 \
    --negatives_x_device True

# HN Sampling

## Eval
model_to_eval=$fourth_nodel_output_path
embeddings_out=./marco/$split/embeddings/$fourth_nodel_output_path
run_save=./marco/$split/results/$fourth_nodel_output_path

dev_queries=./marco/$split/dev.query.txt
dev_qrels=./marco/$split/qrels.dev.tsv

mkdir $run_save

/home/jcoelho/.conda/envs/openmatch/bin/accelerate launch --num_processes $n_gpus --multi_gpu OpenMatch/src/openmatch/driver/build_index.py  \
    --output_dir $embeddings_out \
    --model_name_or_path $model_to_eval \
    --per_device_eval_batch_size 430  \
    --corpus_path $corpus \
    --doc_template "Title: <title> Text: <text>"  \
    --doc_column_names id,title,text  \
    --q_max_len 32  \
    --p_max_len $text_length  \
    --fp16  \
    --dataloader_num_workers 1

/home/jcoelho/.conda/envs/openmatch/bin/accelerate launch --num_processes $n_gpus --multi_gpu OpenMatch/src/openmatch/driver/retrieve.py  \
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
    --retrieve_depth 100

python OpenMatch/scripts/evaluate.py $dev_qrels $run_save/dev.trec > $run_save/dev.results