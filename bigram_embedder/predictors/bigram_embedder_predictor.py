from typing import Tuple

from overrides import overrides

from allennlp.common.util import JsonDict
from allennlp.data import Instance
from allennlp.service.predictors.predictor import Predictor

@Predictor.register('bigram-embedder')
class BigramEmbedderPredictor(Predictor):
    """"Predictor wrapper for the BigramEmbedder"""
    @overrides
    def load_line(self, line: str) -> JsonDict:
        _, _, word1_vector_str, word2_vector_str = line.split('\t')

        return {'w1_vec_str': word1_vector_str, 'w2_vec_str': word2_vector_str}

    @overrides
    def _json_to_instance(self, json_dict: JsonDict) -> Tuple[Instance, JsonDict]:
        w1_vec_str = json_dict['w1_vec_str']
        w2_vec_str = json_dict['w2_vec_str']
        instance = self._dataset_reader.text_to_instance(w1_vec_str, w2_vec_str)

        return instance, {}
