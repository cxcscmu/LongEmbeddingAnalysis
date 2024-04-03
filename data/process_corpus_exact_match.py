# process a corpus for the exact match task
# given a corpus, divide each document in N chunks.
# writes N files, i.tsv, each with the ith chunk of each document.

from transformers import AutoTokenizer
from tqdm import tqdm
import sys

# path where the full.tsv corpus is, and where each span
path = sys.argv[1]
model = sys.argv[2]

N = 10

tokenizer = AutoTokenizer.from_pretrained(model)

def tokenize_and_divide(input_string, n):
    tokens = input_string.split()

    sublist_size = len(tokens) // n
    remaining_tokens = len(tokens) % n

    divided_lists = []

    start_index = 0
    for i in range(n):
        sublist_end_index = start_index + sublist_size
        if i < remaining_tokens:
            sublist_end_index += 1
        sublist = tokens[start_index:sublist_end_index]
        divided_lists.append(sublist)
        start_index = sublist_end_index

    return [' '.join(i) for i in divided_lists]

outs = []
for i in range(N):
    file_path = f"{path}/{i}.tsv"
    outs.append(open(file_path, 'w'))

with open(f"{path}/full.tsv", 'r') as h:

    for line in tqdm(h):
        did, title, body = line.strip().split("\t")

        len_text = len(tokenizer.encode(f"{title} {body}"))

        if len_text <= 2048:
            substrings = tokenize_and_divide(body, N)

            for i in range(len(outs)):
                outs[i].write(f"{did}\t.\t{substrings[i]}\n")

for out_file in outs:
    out_file.close()