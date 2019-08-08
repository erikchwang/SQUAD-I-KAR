ROOT=$(dirname $(realpath $0))
mkdir $ROOT/nltk $ROOT/glove $ROOT/bert $ROOT/data $ROOT/model

python -m nltk.downloader -d $ROOT/nltk stopwords wordnet

wget http://nlp.stanford.edu/data/glove.840B.300d.zip -O $ROOT/glove/glove.zip
unzip -j $ROOT/glove/glove.zip -d $ROOT/glove
rm $ROOT/glove/glove.zip

wget https://storage.googleapis.com/bert_models/2019_05_30/wwm_uncased_L-24_H-1024_A-16.zip -O $ROOT/bert/bert.zip
unzip -j $ROOT/bert/bert.zip -d $ROOT/bert
rm $ROOT/bert/bert.zip

wget https://worksheets.codalab.org/rest/bundles/0x7e0a0a21057c4d989aa68da42886ceb9/contents/blob/ -O $ROOT/data/train_dataset
wget https://worksheets.codalab.org/rest/bundles/0x8f29fe78ffe545128caccab74eb06c57/contents/blob/ -O $ROOT/data/develop_dataset
wget https://worksheets.codalab.org/rest/bundles/0xb765680b60c64d088f5daccac08b3905/contents/blob/ -O $ROOT/data/addsent_dataset
wget https://worksheets.codalab.org/rest/bundles/0x3ac9349d16ba4e7bb9b5920e3b1af393/contents/blob/ -O $ROOT/data/addonesent_dataset
wget https://worksheets.codalab.org/rest/bundles/0xbcd57bee090b421c982906709c8c27e1/contents/blob/ -O $ROOT/data/evaluate_script