#!/bin/bash

#SBATCH --job-name=a_moving_passage_encode

# The line below writes to a logs dir inside the one where sbatch was called
# %x will be replaced by the job name, and %j by the job id

#SBATCH --output=logs/%x-%j.out
#SBATCH -e logs/%x-%j.err
#SBATCH -n 1 # Number of tasks
#SBATCH --cpus-per-task 12 # number cpus (threads) per task

#SBATCH --mem=32000 # Memory - Use up to 2GB per requested CPU as a rule of thumb
#SBATCH --time=0 # No time limit

# You can also change the number of requested GPUs
# replace the XXX with nvidia_a100-pcie-40gb or nvidia_a100-sxm4-40gb
# replace the YYY with the number of GPUs that you need, 1 to 8 PCIe or 1 to 4 SXM4

#SBATCH --gres=gpu:nvidia_a100-sxm4-40gb:1

eval "$(conda shell.bash hook)"
conda activate openmatch

split=documents
text_length=2048
n_gpus=1

# this assumes that this model already has a full corpus index.
model_name=t5-base-documents-firstp-marco-pretrain-v3-dr-pt-2e-2048
#model_name=t5-base-documents-firstp-marco-pretrain-v3-dr-pt-2e-2048-self-hn-1-self-hn-2-self-hn-3
#model_name=t5-base-marco-2048-v3-scaled-dr-pretrain-v2

model_to_eval=/user/home/jcoelho/clueweb-structured-retrieval/models/marco/$model_name

dev_queries=./marco/$split/dev.query.txt
dev_qrels=/user/home/jcoelho/clueweb-structured-retrieval/marco/documents/small_eval_set/qrels.dev.tsv.small.v3


corpus_path=/user/home/jcoelho/clueweb-structured-retrieval/marco/documents/small_eval_set/passage_moving/10-eval-dots/
embeddings_out_path=./marco/$split/embeddings/
run_save_path=./marco/$split/results/


run_save_default=./marco/$split/results/MOVING-default-$model_name

mkdir $run_save_default

echo $model_name
for i in {0..9}; do
    /home/jcoelho/.conda/envs/openmatch/bin/accelerate launch --num_processes $n_gpus OpenMatch/src/openmatch/driver/build_index.py  \
        --output_dir $embeddings_out_path/MOVING-$i-$model_name \
        --model_name_or_path $model_to_eval \
        --per_device_eval_batch_size 430  \
        --corpus_path $corpus_path/corpus_$i.tsv \
        --doc_template "Title: <title> Text: <text>"  \
        --doc_column_names id,title,text  \
        --p_max_len $text_length  \
        --fp16  \
        --dataloader_num_workers 1
done

python /user/home/jcoelho/clueweb-structured-retrieval/marco/documents/small_eval_set/passage_moving/merge_indexes.py $model_name

for i in {0..9}; do
    /home/jcoelho/.conda/envs/openmatch/bin/accelerate launch --num_processes $n_gpus OpenMatch/src/openmatch/driver/retrieve.py  \
        --output_dir $embeddings_out_path/MOVING-$i-$model_name  \
        --model_name_or_path $model_to_eval \
        --per_device_eval_batch_size 600  \
        --query_path $dev_queries \
        --query_template "<text>"  \
        --query_column_names id,text  \
        --q_max_len 32  \
        --fp16  \
        --trec_save_path $run_save_path/MOVING-$i-$model_name/dev.trec \
        --dataloader_num_workers 1 \
        --use_gpu \
        --retrieve_depth 100

    python OpenMatch/scripts/evaluate.py $dev_qrels $run_save_path/MOVING-$i-$model_name/dev.trec > $run_save_path/MOVING-$i-$model_name/dev.results
    echo $( python OpenMatch/scripts/evaluate.py -m recip_rank $dev_qrels $run_save_path/MOVING-$i-$model_name/dev.trec )
done

echo "Default"
echo $(python OpenMatch/scripts/evaluate.py -m recip_rank $dev_qrels ./marco/$split/results/$model_name/dev.trec)
