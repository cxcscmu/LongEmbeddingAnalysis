from transformers import AutoTokenizer
from tqdm import tqdm 

doc_corpus = "/user/home/jcoelho/clueweb-structured-retrieval/marco/documents/corpus.tsv"
dev_doc_qrels_path = "/user/home/jcoelho/clueweb-structured-retrieval/marco/documents/qrels.dev.tsv"
dev_docs_q_path = "/user/home/jcoelho/clueweb-structured-retrieval/marco/documents/dev.query.txt"

pass_corpus = "/user/home/jcoelho/clueweb-structured-retrieval/marco/passage/corpus.tsv"
dev_pass_qrels_path = "/user/home/jcoelho/clueweb-structured-retrieval/marco/passage/qrels.dev.tsv"
dev_pass_q_path = "/user/home/jcoelho/clueweb-structured-retrieval/marco/passage/dev.query.txt"

tokenizer = AutoTokenizer.from_pretrained("/user/home/jcoelho/clueweb-structured-retrieval/models/t5-base-marco-2048-v3+dr-pretrain-scaled")

def create_strings_10(passage, doc):
    # String C
    
    if len(tokenizer.encode(doc)) <= 2048:
        assert passage in doc

        without_passage = doc.replace(passage, '', 1)

        num_intervals = 8
        interval_length = len(without_passage) // (num_intervals + 1)

        # List to store all generated strings
        documents = []

        # String at the beginning
        document_start = passage + " " + without_passage
        documents.append(document_start)

        # Strings in uniform intervals
        for i in range(1, num_intervals + 1):
            start_index = interval_length * i
            interval_doc = without_passage[:start_index] + " " + passage + " " + without_passage[start_index:]
            documents.append(interval_doc)

        # String at the end
        document_end = without_passage + " " + passage
        documents.append(document_end)

        return documents
    else:
        without_passage = doc.replace(passage, '', 1)

        doc_tok = tokenizer.encode(without_passage)
        passage_tok = tokenizer.encode(passage)

        doc_tok = doc_tok[:2048]
        doc_tok = doc_tok[:-len(passage_tok)]

        assert len(doc_tok) + len(passage_tok) <= 2048

        documents = []

        doc = tokenizer.decode(doc_tok)

        num_intervals = 8
        interval_length = len(doc) // (num_intervals + 1)

        document_start = passage + " " + doc

        documents.append(document_start)

        for i in range(1, num_intervals + 1):
            start_index = interval_length * i
            interval_doc = doc[:start_index] + " " + passage + " " + doc[start_index:]
            documents.append(interval_doc)

        document_end = doc + " " + passage
        documents.append(document_end)

        return documents



corpus = {}
with open(pass_corpus, 'r') as h:
    for line in h:
        pid, title, text = line.strip().split("\t")
        corpus[pid] = text.strip()

pass_qrel = {}
with open(dev_pass_qrels_path, 'r') as h:
    for line in h:
        qid,_,pid,_ = line.strip().split("\t")
        pass_qrel[qid] = pid


doc_qrels = {}
with open(dev_doc_qrels_path, 'r') as h:
    for line in h:
        qid,_,pid,_ = line.strip().split("\t")
        doc_qrels[qid] = pid

docs = {}
with open(dev_docs_q_path, 'r') as h:
    for line in h:
        qid, text = line.strip().split("\t")
        text = text.strip().lower()
        docs[text] = qid

passs = {}
with open(dev_pass_q_path, 'r') as h:
    for line in h:
        qid, text = line.strip().split("\t")
        text = text.strip().lower()
        passs[text] = qid

k = 0
for q in docs:
    if q in passs:
        k += 1
    
assert k == len(docs)

# doc_id -> passage text

did2passage = {}
for q in docs:
    pass_qid = passs[q]
    rel_pass = corpus[pass_qrel[pass_qid]]
    did2passage[doc_qrels[docs[q]]] = rel_pass

# doc_id -> doc text
d_corpus = {}
with open(doc_corpus, 'r') as h:
    for line in h:
        did, title, text = line.strip().split("\t")

        if did in did2passage:
            d_corpus[did] = text.strip()

assert d_corpus == did2passage

did2qid = {}

with open("/user/home/jcoelho/clueweb-structured-retrieval/marco/documents/small_eval_set/qrels.dev.tsv.small.v3") as h:
    for line in h:
        qid, _, did, _ = line.strip().split("\t")
        did2qid[did] = qid
    

with open("/user/home/jcoelho/clueweb-structured-retrieval/marco/documents/small_eval_set/positive_passage_index_v3.tsv", 'w') as h:
    for k in did2passage:
        if did2passage[k] in d_corpus[k]:
            h.write(f"{did2qid[k]}\t{d_corpus[k].index(did2passage[k])}\n") 

    

doc_corpus = "/user/home/jcoelho/clueweb-structured-retrieval/marco/documents/corpus.tsv"

corpus_0 = "/user/home/jcoelho/clueweb-structured-retrieval/marco/documents/small_eval_set/10-evaluation-dots/corpus_0.tsv"
corpus_1 = "/user/home/jcoelho/clueweb-structured-retrieval/marco/documents/small_eval_set/10-evaluation-dots/corpus_1.tsv"
corpus_2 = "/user/home/jcoelho/clueweb-structured-retrieval/marco/documents/small_eval_set/10-evaluation-dots/corpus_2.tsv"
corpus_3 = "/user/home/jcoelho/clueweb-structured-retrieval/marco/documents/small_eval_set/10-evaluation-dots/corpus_3.tsv"
corpus_4 = "/user/home/jcoelho/clueweb-structured-retrieval/marco/documents/small_eval_set/10-evaluation-dots/corpus_4.tsv"
corpus_5 = "/user/home/jcoelho/clueweb-structured-retrieval/marco/documents/small_eval_set/10-evaluation-dots/corpus_5.tsv"
corpus_6 = "/user/home/jcoelho/clueweb-structured-retrieval/marco/documents/small_eval_set/10-evaluation-dots/corpus_6.tsv"
corpus_7 = "/user/home/jcoelho/clueweb-structured-retrieval/marco/documents/small_eval_set/10-evaluation-dots/corpus_7.tsv"
corpus_8 = "/user/home/jcoelho/clueweb-structured-retrieval/marco/documents/small_eval_set/10-evaluation-dots/corpus_8.tsv"
corpus_9 = "/user/home/jcoelho/clueweb-structured-retrieval/marco/documents/small_eval_set/10-evaluation-dots/corpus_9.tsv"

did2passage_small = {}
for k in did2passage:
    if did2passage[k] in d_corpus[k]:
        did2passage_small[k] = did2passage[k]

with open(doc_corpus, 'r') as h, \
    open(corpus_0, 'w') as out1, \
    open(corpus_1, 'w') as out2, \
    open(corpus_2, 'w') as out3, \
    open(corpus_3, 'w') as out4, \
    open(corpus_4, 'w') as out5, \
    open(corpus_5, 'w') as out6, \
    open(corpus_6, 'w') as out7, \
    open(corpus_7, 'w') as out8, \
    open(corpus_8, 'w') as out9, \
    open(corpus_9, 'w') as out10:

    outs = [out1, out2, out3, out4, out5, out6, out7, out8, out9, out10]
    
    for line in tqdm(h):
        did, title, text = line.strip().split("\t")
    
        if did not in dids:
            for i in range(len(outs)):
                outs[i].write(f"{did}\t{title}\t{text}\n")

        else:
            docs = create_strings_10(did2passage_small[did].lower(), text.lower())

            for i in range(len(docs)):
                title = title.replace("\"", "").replace("\'", "").replace("\n", "").replace("\t", "").replace("\r", "")
                text = docs[i].replace("\"", "").replace("\'", "").replace("\n", "").replace("\t", "").replace("\r", "")
                outs[i].write(f"{did}\t{title}\t{text}\n")
       
            

