# Adapted from Tevatron (https://github.com/texttron/tevatron)

from argparse import ArgumentParser
from transformers import AutoTokenizer
import os
import random
from tqdm import tqdm
from datetime import datetime
from multiprocessing import Pool
from openmatch.utils import SimpleCollectionPreProcessor as PreTrainPreProcessor

random.seed(datetime.now())
parser = ArgumentParser()
parser.add_argument('--tokenizer_name', required=True)
parser.add_argument('--collection', required=True)
parser.add_argument('--save_to', required=True)

parser.add_argument('--truncate', type=int, default=128)
parser.add_argument('--mp_chunk_size', type=int, default=500)
parser.add_argument('--shard_size', type=int, default=450000)

args = parser.parse_args()


tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name, use_fast=True)
processor = PreTrainPreProcessor(
    tokenizer=tokenizer,
    max_length=args.truncate,
)

print("MAX LEN:", processor.max_length)

counter = 0
shard_id = 0
f = None
os.makedirs(args.save_to, exist_ok=True)

def read_line(l):
    return l

with open(args.collection) as corpus:
    pbar = tqdm(map(read_line, corpus))
    with Pool() as p:
        for x in p.imap(processor.process_line, pbar, chunksize=args.mp_chunk_size):
            counter += 1
            if f is None:
                f = open(os.path.join(args.save_to, f'split{shard_id:02d}.jsonl'), 'w')
                pbar.set_description(f'split - {shard_id:02d}')
            f.write(x + '\n')

            if counter == args.shard_size:
                f.close()
                f = None
                shard_id += 1
            counter = 0

if f is not None:
    f.close()