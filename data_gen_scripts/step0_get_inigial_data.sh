# data comes from http://mattmahoney.net/
PATH_DATA_STEP0=../data/step0_get_initial_data
PATH_FASTTEXT=../fastText-0.1.0

echo "Downloading data..."
wget -c http://mattmahoney.net/dc/enwik9.zip -P $PATH_DATA_STEP0

echo "Unzipping data..."
unzip $PATH_DATA_STEP0/enwik9.zip -d $PATH_DATA_STEP0

echo "Preprocessing data..."
perl $PATH_FASTTEXT/wikifil.pl $PATH_DATA_STEP0/enwik9 > $PATH_DATA_STEP0/fil9

echo "Data is ready for step 1"
