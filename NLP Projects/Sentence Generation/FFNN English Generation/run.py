import argparse
import random

import numpy as np
from tqdm import tqdm

from edugrad.tensor import Tensor

from data import SSTLanguageModelingDataset
from model import FeedForwardLanguageModel
from ops import softmax_rows, cross_entropy_loss
from optim import Adagrad
from vocabulary import Vocabulary


def sample_next_char(probabilities: np.ndarray) -> np.ndarray:
    """Sample the next characters from an array of probability distributions.

    Args:
        probabilities: [batch_size, vocab_size]
            each row of this array is a probability distribution over the vocabulary
            the method samples one character (index from the vocab) for each row, according
            to that probability distribution

    Returns:
        [batch_size] shaped numpy array of integer indices, corresponding to the samples
    """
    # TODO: (~2-3 lines) implement this method
    # Hint: np.random.choice is helpful
    return np.array([np.random.choice(len(dist), p = dist) for dist in probabilities])


def generate(
    model: FeedForwardLanguageModel,
    bos_index: int,
    num_prev_chars: int,
    batch_size: int,
    max_len: int,
    vocab: Vocabulary,
    temp: float = 3.0,
) -> list[str]:
    """Generate character strings from a FeedForwardLanguageModel.

    Arguments:
        model: the model to generate from
        bos_index: index of BOS symbol in vocabulary
        num_prev_chars: character n-gram size for input to the model
        batch_size: how many strings to generate
        max_len: length of generations
        vocab: the vocabulary of the model
        temp: softmax temperature; larger values make the samples closer to argmax

    Returns:
        list, batch_size long, of max_len character strings generated from the model
    """
    # [batch_size, num_prev_chars], filled with BOS tokens
    generated = [np.zeros((batch_size)).astype(int) + bos_index] * num_prev_chars
    for _ in range(max_len):
        batch = generated[-num_prev_chars:]
        # get logits from model
        logits = model([Tensor(idx) for idx in batch]).value * temp
        # use softmax temp to generate probabilities
        probabilities = np.exp(logits) / np.sum(np.exp(logits), axis=1, keepdims=True)
        # sample next characters
        next_chars = sample_next_char(probabilities)
        # add to list of generated characters
        generated = generated + [next_chars]
    # [seq_len, batch_size] --> [batch_size, seq_len]
    texts = np.array(generated).transpose()
    # throw away the initial BOS chunk
    texts = texts[:, num_prev_chars:]
    texts = ["".join(vocab.indices_to_tokens(text)) for text in texts]
    return texts


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--random_seed", type=int, default=575)
    parser.add_argument("--num_epochs", type=int, default=16)
    parser.add_argument("--num_prev_chars", type=int, default=16)
    parser.add_argument("--embedding_size", type=int, default=60)
    parser.add_argument("--hidden_size", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument(
        "--training_data",
        type=str,
        default="/dropbox/20-21/575k/data/sst/train-reviews.txt",
    )
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--temp", type=float, default=2.5)
    parser.add_argument("--generate_every", type=int, default=4)
    parser.add_argument("--generate_length", type=int, default=50)
    parser.add_argument("--num_generate", type=int, default=10)
    args = parser.parse_args()

    random.seed(args.random_seed)
    np.random.seed(args.random_seed)

    training_data = SSTLanguageModelingDataset.from_file(
        args.training_data, args.num_prev_chars
    )
    vocab_size = len(training_data.vocab)
    data_size = len(training_data)
    starts = list(range(0, data_size, args.batch_size))

    model = FeedForwardLanguageModel(
        args.num_prev_chars, vocab_size, args.embedding_size, args.hidden_size
    )
    optimizer = Adagrad(model.parameters(), lr=args.lr)

    for epoch in range(args.num_epochs):
        # shuffle batches
        random.shuffle(starts)
        epoch_loss = 0.0
        for start in tqdm(starts):
            # last batch might not be full size
            end = min(data_size, start + args.batch_size)
            batch = training_data.batch_as_tensors(start, end)
            logits = model([Tensor(array) for array in batch["text"]])
            probabilities = softmax_rows(logits)
            loss = cross_entropy_loss(probabilities, Tensor(batch["target"]))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.value
        print(f"Epoch {epoch} avg train loss: {epoch_loss / len(starts)}")
        # generate some text every N epochs
        if (epoch + 1) % args.generate_every == 0:
            print(
                generate(
                    model,
                    training_data.vocab[SSTLanguageModelingDataset.BOS],
                    args.num_prev_chars,
                    args.num_generate,
                    args.generate_length,
                    training_data.vocab,
                    temp=args.temp,
                )
            )
