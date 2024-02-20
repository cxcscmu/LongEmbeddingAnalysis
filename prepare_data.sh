downloads_directory="./data/marco_documents"

processed_data_directory="./data/marco_documents_processed"

mkdir -p $downloads_directory
mkdir -p $processed_data_directory

wget -P $downloads_directory https://msmarco.z22.web.core.windows.net/msmarcoranking/msmarco-docs.tsv.gz

wget -P $downloads_directory https://msmarco.z22.web.core.windows.net/msmarcoranking/msmarco-doctrain-queries.tsv.gz

wget -P $downloads_directory https://msmarco.z22.web.core.windows.net/msmarcoranking/msmarco-doctrain-qrels.tsv.gz

wget -P $downloads_directory https://msmarco.z22.web.core.windows.net/msmarcoranking/msmarco-docdev-queries.tsv.gz

wget -P $downloads_directory https://msmarco.z22.web.core.windows.net/msmarcoranking/msmarco-docdev-qrels.tsv.gz

wget -P $downloads_directory https://thunlp.oss-cn-qingdao.aliyuncs.com/OpenMatch/MSMARCO/document_ranking/bids_marco-doc_ance-maxp-10.tsv.zip

for file in "$downloads_directory"/*.gz; do
    gzip -d "$file"
done

for file in "$downloads_directory"/*.zip; do
    unzip "$file" -d "$downloads_directory"
done

rm "$downloads_directory"/*.zip

python ./data/process_marco_doc.py 2048 $downloads_directory $processed_data_directory

