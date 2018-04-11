# Phrase representations via association module inside predefined skip-gram embedding space 

## Results replication guide
### Prepare data for training
* `cd data_gen_scripts`
* run `step0` script to download and preprocess wiki data
* run `step1` script to extract potential bigrams we will work with
* run `step2` script to glue some words to form bigrams in corpus from step0
* run `step3` script to train fasttext skip-gram model on words & bigrams corpus from step2
* run `step4` script to extract actual wordpair_vec -> bigram_vec corpus needed to train prediction model
### Train model
To train this model, after setting up your development environment by installing what is in `requirements.txt`, 
fastText-0.1.0(https://fasttext.cc/) and `spacy>=2.0`, you run:
```bash
python -m allennlp.run train experiments/venue_classifier.json -s /models --include-package towards_machine_reasoning
```

This project relies on `python-3.6`.

