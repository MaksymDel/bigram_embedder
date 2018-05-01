from typing import Dict, Optional

import torch
from allennlp.common import Params
from allennlp.data import Vocabulary
from allennlp.models.model import Model
from allennlp.modules import FeedForward, Seq2VecEncoder
from allennlp.nn import InitializerApplicator, RegularizerApplicator
from allennlp.nn.util import get_text_field_mask
from overrides import overrides


@Model.register("bigram_embedder_seq2vec")
class BigramEmbedderSeq2Vec(Model):
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
    seq2vec_encoder: 'Seq2VecEncoder",
        It is used to encoder vectors as a sequence
    initializer : ``InitializerApplicator``, optional (default=``InitializerApplicator()``)
        Used to initialize the model parameters.
    regularizer : ``RegularizerApplicator``, optional (default=``None``)
        If provided, will be used to calculate the regularization penalty during training.
    """
    def __init__(self, vocab: Vocabulary,
                 seq2vec_encoder: Seq2VecEncoder,
                 initializer: InitializerApplicator = InitializerApplicator(),
                 regularizer: Optional[RegularizerApplicator] = None) -> None:
        super(BigramEmbedderSeq2Vec, self).__init__(vocab, regularizer)

        self.seq2vec_encoder = seq2vec_encoder

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

        seq_tensor = torch.stack([word1_vector, word2_vector], dim=1)
        #mask_stub = torch.ones(seq_tensor.size()[0:2]).long()
        mask_stub = torch.autograd.Variable(word1_vector.data.new(seq_tensor.size()[0:2]).fill_(1))
        bigram_vecs_hat = self.seq2vec_encoder(seq_tensor, mask_stub.long())
        output_dict = {'bigram_vecs_hat': bigram_vecs_hat}
        if bigram_vector is not None:
            loss = self.loss(bigram_vecs_hat, bigram_vector)
            output_dict["loss"] = loss

        return output_dict

    @classmethod
    def from_params(cls, vocab: Vocabulary, params: Params) -> 'BigramEmbedderSeq2Vec':
        seq2vec_params = params.pop("seq2vec_encoder")
        seq2vec_encoder = Seq2VecEncoder.from_params(seq2vec_params)

        initializer = InitializerApplicator.from_params(params.pop('initializer', []))
        regularizer = RegularizerApplicator.from_params(params.pop('regularizer', []))

        return cls(vocab=vocab,
                   seq2vec_encoder=seq2vec_encoder,
                   initializer=initializer,
                   regularizer=regularizer)
