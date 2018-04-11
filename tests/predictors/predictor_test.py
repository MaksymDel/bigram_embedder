# pylint: disable=no-self-use,invalid-name,unused-import
from unittest import TestCase

from allennlp.models.archival import load_archive
from allennlp.service.predictors import Predictor

# required so that our custom model + predictor + dataset reader
# will be registered by name
import bigram_embedder

class TestBigramEmbedderPredictor(TestCase):
    def test_uses_named_inputs(self):
        with open('tests/fixtures/bigram_vecs.tsv') as f:
            inputs = f.read()
            inputs = inputs.split("\n")[0]

        archive = load_archive('tests/fixtures/tiny/model.tar.gz')
        predictor = Predictor.from_archive(archive, 'bigram-embedder')

        inputs = predictor.load_line(inputs)
        result = predictor.predict_json(inputs)

        bigram_vecs_hat = result.get("bigram_vecs_hat")
        DIM = 100
        assert len(bigram_vecs_hat) == len(inputs['w1_vec_str'].split(' ')) == DIM

