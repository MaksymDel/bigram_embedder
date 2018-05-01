# pylint: disable=invalid-name,protected-access
from allennlp.common.testing import ModelTestCase


class BigramEmbedderSeq2Vec(ModelTestCase):
    def setUp(self):
        super(BigramEmbedderSeq2Vec, self).setUp()
        self.set_up_model('tests/fixtures/lstm.json',
                          'tests/fixtures/bigram_vecs.tsv')

    def test_model_can_train_save_and_load(self):
        self.ensure_model_can_train_save_and_load(self.param_file)
