# pylint: disable=invalid-name,protected-access
from allennlp.common.testing import ModelTestCase


class BigramEmbedderTest(ModelTestCase):
    def setUp(self):
        super(BigramEmbedderTest, self).setUp()
        self.set_up_model('tests/fixtures/bigram_embedder.json',
                          'tests/fixtures/bigram_vecs.tsv')

    def test_model_can_train_save_and_load(self):
        self.ensure_model_can_train_save_and_load(self.param_file)
