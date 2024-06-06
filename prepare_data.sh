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
python ./data/process_marco_doc.py full $downloads_directory $processed_data_directory

downloads_directory="./data/marco_passage"

mkdir -p $downloads_directory

wget -P $downloads_directory --no-check-certificate https://rocketqa.bj.bcebos.com/corpus/marco.tar.gz
tar -zxf $downloads_directory/marco.tar.gz -C $downloads_directory
rm -rf $downloads_directory/marco.tar.gz
mv $downloads_directory/marco/* $downloads_directory
rm -r $downloads_directory/marco
join  -t "$(echo -en '\t')"  -e '' -a 1  -o 1.1 2.2 1.2  <(sort -k1,1 $downloads_directory/para.txt) <(sort -k1,1 $downloads_directory/para.title.txt) | sort -k1,1 -n > $downloads_directory/corpus.tsv
