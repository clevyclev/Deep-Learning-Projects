import edugrad.nn as nn
import numpy as np

from edugrad.tensor import Tensor
import ops


class Embedding(nn.Module):
    def __init__(self, vocab_size: int, embedding_dim: int):
        super(Embedding, self).__init__()
        scale = 1 / np.sqrt(embedding_dim)
        self.weight = Tensor(
            np.random.standard_normal((vocab_size, embedding_dim)), name="E"
        )

    def forward(self, indices: Tensor) -> Tensor:
        return ops.lookup_rows(self.weight, indices)


class FeedForwardLanguageModel(nn.Module):
    def __init__(
        self, window_size: int, vocab_size: int, embedding_size: int, hidden_size: int
    ):
        super(FeedForwardLanguageModel, self).__init__()
        self.embedding = Embedding(vocab_size, embedding_size)
        self.fc = nn.Linear(embedding_size * window_size, hidden_size)
        self.output = nn.Linear(hidden_size, vocab_size)

    def forward(self, word_indices: list[Tensor]) -> Tensor:
        """Executes the forward pass of a FeedForwardLanguageModel.

        Args:
            word_indices: list of [batch_size] tensors
                length = number of previous characters / n-gram length
                each one contains indices of chars at that position

        Returns:
            [batch_size, vocab_size] Tensor
            containing logits (not full probabilities, i.e. pre-softmax)
            over the vocab for each example in the batch
        """
        # TODO: (~7 lines) implement the forward pass of FFNN LM here
        # HINT: use ops.concat to concatenate word embeddings together
        # It takes a variable-length list of Tensors as its input, so you can
        # call it using as ops.concat(*embeddings), where embeddings is a list
        # of Tensors, corresponding to the relevant embeddings
        # [batch_size, num_words * embedding_size]
        embs = ops.concat(*[self.embedding(index) for index in word_indices])
        return self.output(ops.tanh(self.fc(embs)))
