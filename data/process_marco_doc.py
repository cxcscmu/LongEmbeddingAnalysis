# Given the data from here: https://github.com/thunlp/OpenMatch/blob/fbf198ad5353e35b8b67ea7539783db61b32d1f2/v1/docs/experiments-msmarco-doc.md
# build files files in normalized format to be used as input for OpenMatch

import re
import pickle
import sys

MAX_DOC_LEN = int(sys.argv[1])
data_target_in = sys.argv[2]
data_target_path = sys.argv[3]


#MAX_DOC_LEN = 2048

def is_valid_id(did):
    pattern = re.compile(r'^D\d+$')
    return bool(pattern.match(did))

ID_MAPPER_DOCS = {}

corpus_only = False

# this script prepares marco data to be compatible with openmatch

# 1- corpus with sequential integer ids.
with open(f"{data_target_in}/msmarco-docs.tsv", "r") as f, \
    open(f"{data_target_path}/corpus_firstp_{MAX_DOC_LEN}.tsv", "w") as fout:
    start_id = 0
    for line in f:
        try:
            did, url, title, text = line.strip().split("\t")
        except Exception:
            if len(line.strip().split("\t")) == 3:
                did = line.strip().split("\t")[0]
                title = "-"
                text = line.strip().split("\t")[1]
            if len(line.strip().split("\t")) == 2:
                did = line.strip().split("\t")[0]
                title = "-"
                text = line.strip().split("\t")[-1]
            if len(line.strip().split("\t")) == 1:
                raise Exception("this should not happen if initial dataset is correct, check it")
        if is_valid_id(did):
            did = did[1:]
        else:
            raise Exception("problem with ids: corpus")
        
        ID_MAPPER_DOCS[did] = str(start_id)

        text = text.replace("\"", "").replace("\'", "").replace("\n", "").replace("\t", "").replace("\r", "")
        text = " ".join(text.split(" ")[:MAX_DOC_LEN])
        fout.write(f"{ID_MAPPER_DOCS[did]}\t{title}\t{text}\n")
        start_id += 1

if corpus_only:
    exit()


# Read the negatives and process them to openmatch format
# train.negatives.tsv <qid>\t<neg1,neg2,neg3...>
negatives = {}
with open(f"{data_target_in}/bids_marco-doc_ance-maxp-10.tsv", 'r') as f:
    for line in f:
        qid, did, label = line.strip().split()
        if label == "1":
            continue
        else:
            if is_valid_id(did):
                did = ID_MAPPER_DOCS[did[1:]]
            else:
                raise Exception("problem with ids")

            if qid not in negatives:
                negatives[qid] = [did] 
                
            else: 
                negatives[qid].append(did)

with open(f"{data_target_path}/train.negatives.tsv", 'w') as fout:
    for qid in negatives:
        negs = ",".join(negatives[qid])

        fout.write(f"{qid}\t{negs}\n")


# ensure queries have qrels

train_queries_with_qrels = []
with open(f"{data_target_in}/msmarco-doctrain-qrels.tsv", 'r') as f, \
    open(f"{data_target_path}/qrels.train.tsv", 'w') as fout:

    for line in f:
        qid, z, did, rel = line.strip().split()

        if qid in negatives:
            if is_valid_id(did):
                did = ID_MAPPER_DOCS[did[1:]]
            else:
                raise Exception("problem with ids: train qrels")

            train_queries_with_qrels.append(qid)
        
            fout.write(f"{qid}\t{z}\t{did}\t{rel}\n")

train_queries_with_qrels = set(train_queries_with_qrels)


dev_queries_with_qrels = []
with open(f"{data_target_in}/msmarco-docdev-qrels.tsv", 'r') as f, \
    open(f"{data_target_path}/qrels.dev.tsv", 'w') as fout:

    for line in f:
        qid, z, did, rel = line.strip().split()

        if is_valid_id(did):
            did = ID_MAPPER_DOCS[did[1:]]
        else:
            raise Exception("problem with ids: dev qrels")

        dev_queries_with_qrels.append(qid)
        fout.write(f"{qid}\t{z}\t{did}\t{rel}\n")

dev_queries_with_qrels = set(dev_queries_with_qrels)

with open(f"{data_target_in}/msmarco-doctrain-queries.tsv", 'r') as f, \
    open(f"{data_target_path}/train.query.txt", 'w') as fout:
    for line in f:
        qid, text = line.strip().split("\t")
        if qid in train_queries_with_qrels:
            fout.write(f"{qid}\t{text}\n")

with open(f"{data_target_in}/msmarco-docdev-queries.tsv", 'r') as f, \
    open(f"{data_target_path}/dev.query.txt", 'w') as fout:
    for line in f:
        qid, text = line.strip().split("\t")
        if qid in dev_queries_with_qrels:
            fout.write(f"{qid}\t{text}\n")

# save the id mapping in case it is necessary
with open(f"{data_target_path}/id2seqint_mapper.pkl", 'wb') as h:
    pickle.dump(ID_MAPPER_DOCS, h, protocol=pickle.HIGHEST_PROTOCOL)