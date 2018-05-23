# Unsupervised Compositional Modeling Framework for Phrases 

## Results replication guide
### Prepare data for training
* `cd data_gen_scripts`
* run `step0` script to download and preprocess wiki data
* run `step1` script to extract potential bigrams we will work with
* run `step2` script to glue some words to form bigrams in corpus from step0
* run `step3` script to train fasttext skip-gram model on words & bigrams corpus from step2 (get fasttext from https://fasttext.cc/))
* run `step4` script to extract actual wordpair_vec -> bigram_vec corpus needed to train prediction model
* run `step5` script to create train, dev, and test splits
### Train model
So by now you have the right data in `data/step5` folder. <br>
Next, set up your development environment by installing what is in `requirements.txt`, and `spacy>=2.0`.
(as after step5 data generation step) and run:
```bash
python -m allennlp.run train experiments/bigram_embedder.json -s models/bigram_embedder_feedforw --include-package bigram_embedder
```

This project relies on `python-3.6`.

