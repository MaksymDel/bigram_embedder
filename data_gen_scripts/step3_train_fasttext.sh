#!/usr/bin/env bash
PATH_TO_FASTTEXT=../fastText-0.1.0/fasttext
# BIGRAMS_TYPE=likelihood_ratio
DIM=300
LR=0.05
BIGRAMS_TYPE=raw_freq
$PATH_TO_FASTTEXT skipgram -input ../data/corpus_transformed_$BIGRAMS_TYPE.txt -output ../data/vectors_$BIGRAMS_TYPE-$DIM-$LR -dim $DIM -lr $LR -thread 30
