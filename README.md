
## Installation

First, start by building the conda environment, downloading and processing the data.
```bash 
./prepare_environment.sh env.yaml

./prepare_data.sh
```
Note: you may need to install torch versions that are supported by your machine. The provided environment defaults 2.0

### Data formats
Exemplifying the MS-MARCO corpus in OpenMatch format.

Corpus: ``<doc_id>\t<title>\t<text>``
Example: 
```bash 
head -1 ./data/marco_documents_processed/corpus_firstp_2048.tsv
>>> "0       The hot glowing surfaces of stars emit energy in the form of electromagnetic radiation.?        Science & Mathematics Physics The hot glowing surfaces of stars emit energy ...."
```

Queries: ``<query_id>\t<text>``
```bash 
head -1 ./data/marco_documents_processed/train.query.txt
>>> "1185869 )what was the immediate impact of the success of the manhattan project?"
```

Qrels: ``<query_id>\t<Q0>\t<doc_id>\t<relevancy>``
```bash 
head -1 ./data/marco_documents_processed/qrels.train.tsv
>>> "3       0       895028  1"
```

Negatives: ``<query_id>\t<neg_doc_id_1, ..., neg_doc_id_N>``
```bash 
head -1 ./data/marco_documents_processed/train.negatives.tsv
>>> "3       1896535,3054358,527877,2312708,1606470,1680606,2574509,2591560,712805,1678190"
```

Note: OpenMatch expects sequential integers on document ids. Hence the need for the processing step.
## Model inference

```bash
./scripts/eval_dr.sh jmvcoelho/t5-base-marco-2048
```

This will create the results folder, with the trec-formatted run and trec-eval output for the model.

## Model training

TODO

## Other experiments

TODO
