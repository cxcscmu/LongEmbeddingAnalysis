from transformers import AutoTokenizer
from tqdm import tqdm 

doc_corpus = "./data/marco_documents_processed/corpus_firstp_full.tsv"
dev_doc_qrels_path = "./data/marco_documents_processed/qrels.dev.tsv"
dev_docs_q_path = "./data/marco_documents_processed/dev.query.txt"

pass_corpus = "./data/marco_passage/corpus.tsv"
dev_pass_qrels_path = "./data/marco_passage/qrels.dev.tsv"
dev_pass_q_path = "./data/marco_passage/dev.query.txt"

MAX_SEQ_LEN = 2048
NUM_DATA_POINTS = 10 

tokenizer = AutoTokenizer.from_pretrained("jmvcoelho/t5-base-marco-2048") # experiments are model-specific, will need to change this

def create_strings_10(passage, doc):
    # String C
    assert passage in doc # make sure we can exact match the passage in the document!
    if len(tokenizer.encode(doc)) <= MAX_SEQ_LEN: # if doc has less than 2048 just move it around

        without_passage = doc.replace(passage, '', 1)

        num_intervals = NUM_DATA_POINTS-2
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
    else: # if doc has more than 2048, remove passage, truncate doc to doc-passage, then move passahe around
        without_passage = doc.replace(passage, '', 1)

        doc_tok = tokenizer.encode(without_passage)
        passage_tok = tokenizer.encode(passage)

        doc_tok = doc_tok[:MAX_SEQ_LEN]
        doc_tok = doc_tok[:-len(passage_tok)]

        assert len(doc_tok) + len(passage_tok) <= MAX_SEQ_LEN

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
        qid, pid = line.strip().split("\t")
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

did2qid = {}

with open("./data/qrels.dev.tsv.moving.passage") as h:
    for line in h:
        qid, _, did, _ = line.strip().split("\t")
        did2qid[did] = qid
        


did2passage_small = {}
for k in did2passage:
    if did2passage[k] in d_corpus[k]:
        did2passage_small[k] = did2passage[k]

corpus_files = ["./data/moving_passage_experiment/corpus_{}.tsv".format(i) for i in range(NUM_DATA_POINTS)]

with open(doc_corpus, 'r') as h :    
        for line in tqdm(h):
            did, title, text = line.strip().split("\t")
        
            if did in did2qid:
                
                docs = create_strings_10(did2passage_small[did].lower(), text.lower())

                for i in range(len(docs)):
                    with open(corpus_files[i], 'a') as out:
                        title = title.replace("\"", "").replace("\'", "").replace("\n", "").replace("\t", "").replace("\r", "")
                        text = docs[i].replace("\"", "").replace("\'", "").replace("\n", "").replace("\t", "").replace("\r", "")
                        out.write(f"{did}\t{title}\t{text}\n")
