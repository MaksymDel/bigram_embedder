# pylint: disable=no-self-use,invalid-name
from allennlp.common.testing import AllenNlpTestCase
from allennlp.common.util import ensure_list

from bigram_embedder.dataset_readers import BigramEmbedderDatasetReader


class TestSemanticScholarDatasetReader(AllenNlpTestCase):
    def test_read_from_file(self):

        reader = BigramEmbedderDatasetReader()
        instances = ensure_list(reader.read('tests/fixtures/bigram_vecs.tsv'))

        assert len(instances) == 5

        fields = instances[1].fields
        D = 100
        self.assertTrue(len(fields['word1_vector'].array) == len(fields['word2_vector'].array)
                        == len(fields['bigram_vector'].array) == D)