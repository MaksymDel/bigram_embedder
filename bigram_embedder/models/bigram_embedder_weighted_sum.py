from typing import Dict, Optional

import torch
from allennlp.common import Params
from allennlp.data import Vocabulary
from allennlp.models.model import Model
from allennlp.modules import FeedForward, Seq2VecEncoder
from allennlp.nn import InitializerApplicator, RegularizerApplicator
from overrides import overrides


@Model.register("bigram_embedder_ws")
class BigramEmbedderWeightedSum(Model):
    """
    This ``Model`` performs text classification for an academic paper.  We assume we're given a
    title and an abstract, and we predict some output label.

    The basic model structure: we'll embed the title and the abstract, and encode each of them with
    separate Seq2VecEncoders, getting a single vector representing the content of each.  We'll then
    concatenate those two vectors, and pass the result through a feedforward network, the output of
    which we'll use as our scores for each label.

    Parameters
    ----------
    vocab : ``Vocabulary``, required
        A Vocabulary, required in order to compute sizes for input/output projections.
    mapping_layer : ``FeedForward``
        Generates scores (2 scalars) that we will use to weight word vectors before sum
    initializer : ``InitializerApplicator``, optional (default=``InitializerApplicator()``)
        Used to initialize the model parameters.
    regularizer : ``RegularizerApplicator``, optional (default=``None``)
        If provided, will be used to calculate the regularization penalty during training.
    """
    def __init__(self, vocab: Vocabulary,
                 mapping_layer: FeedForward,
                 initializer: InitializerApplicator = InitializerApplicator(),
                 regularizer: Optional[RegularizerApplicator] = None) -> None:
        super(BigramEmbedderWeightedSum, self).__init__(vocab, regularizer)

        self.mapping_layer = mapping_layer

        self.loss = torch.nn.SmoothL1Loss(size_average=True, reduce=True)

        initializer(self)

    @overrides
    def forward(self,  # type: ignore
                word1_vector: torch.LongTensor,
                word2_vector: torch.LongTensor,
                bigram_vector: torch.LongTensor = None) -> Dict[str, torch.Tensor]:
        # pylint: disable=arguments-differ
        """
        Parameters
        ----------
        word1_vector : Variable, required
            Represents vector for the 1st bigram word.
            The output of ``TextField.as_array()``.
        word2_vector : Variable, required
            Represents vector for the 2nd bigram word.
        bigram_vector : Variable, optional (default = None)
            Represents vector for the target bigram.
            The output of ``TextField.as_array()``.

        Returns
        -------
        An output dictionary consisting of:
        bigram_vecs_hat : [torch.FloatTensor]
            A tensor of shape ``(batch_size, emb_dim)`` representing a predicted bigram vectors
        loss : torch.FloatTensor, optional
            A scalar loss to be optimised.
        """

        sum_weights = self.mapping_layer(torch.cat([word1_vector, word2_vector], dim=-1))
        sum_weight_1, sum_weight_2 = sum_weights[:, 0], sum_weights[:, 1]

        bigram_vecs_hat = sum_weight_1.unsqueeze(1) * word1_vector + sum_weight_2.unsqueeze(1) * word2_vector

        output_dict = {'bigram_vecs_hat': bigram_vecs_hat}
        if bigram_vector is not None:
            loss = self.loss(bigram_vecs_hat, bigram_vector)
            output_dict["loss"] = loss

        return output_dict

    @classmethod
    def from_params(cls, vocab: Vocabulary, params: Params) -> 'BigramEmbedderWeightedSum':
        mapping_layer_params = params.pop("mapping_layer")
        mapping_layer = FeedForward.from_params(mapping_layer_params)

        initializer = InitializerApplicator.from_params(params.pop('initializer', []))
        regularizer = RegularizerApplicator.from_params(params.pop('regularizer', []))

        return cls(vocab=vocab,
                   mapping_layer=mapping_layer,
                   initializer=initializer,
                   regularizer=regularizer)
