from typing import Any, Callable

import numpy as np
from vocabulary import Vocabulary

# type aliases
Example = dict[str, Any]


class Dataset:
    def __init__(self, examples: list[Example], vocab: Vocabulary) -> None:
        self.examples = examples
        self.vocab = vocab

    def example_to_tensors(self, index: int) -> dict[str, np.ndarray]:
        raise NotImplementedError

    def batch_as_tensors(self, start: int, end: int) -> dict[str, np.ndarray]:
        raise NotImplementedError

    def __getitem__(self, idx: int) -> dict:
        return self.examples[idx]

    def __len__(self) -> int:
        return len(self.examples)


def pad_batch(sequences: list[np.ndarray], padding_index: int) -> np.ndarray:
    """Pad a list of sequences, so that that they all have the same length.
    Return as one [batch_size, max_seq_len] numpy array.

    Example usage:
    >>> pad_batch([np.array([2, 4]), np.array([1, 3, 6]), np.array([2])], 0)
    >>> np.array([[2, 4, 0], [1, 3, 6], [2, 0, 0]])

    Arguments:
        sequences: list of arrays, each containing integer indices, to pad and combine
            Each array will be 1-dimensional, but they might have different lengths.
        padding_index: integer index of PAD symbol, used to fill in to make sequences longer

    Returns:
        [batch_size, max_seq_len] numpy array, where each row is the corresponding
        sequence from the sequences argument, but padded out to the maximum length with
        padding_index
    """
    # TODO: implement here! ~6-7 lines
    # NB: you should _not_ directly modify the `sequences` argument, or any of the arrays
    # contained in that list.  (This may cause tests, and some aspects of training, to fail.)
    # Rather, you can use them without modification to build your new array to be returned.
    longest = max([len(sequence) for sequence in sequences])
    padded_sequences = []
    for sequence in sequences:
        temp_seq = [idx for idx in sequence] + [padding_index] * (longest - len(sequence))
        padded_sequences.append(temp_seq)
    return np.array(padded_sequences) 


class SSTClassificationDataset(Dataset):

    labels_to_string = {0: "terrible", 1: "bad", 2: "so-so", 3: "good", 4: "excellent"}
    label_one_hots = np.eye(len(labels_to_string))
    PAD = "<PAD>"

    def example_to_tensors(self, index: int) -> dict[str, np.ndarray]:
        example = self.__getitem__(index)
        return {
            "review": np.array(self.vocab.tokens_to_indices(example["review"])),
            "label": example["label"],
        }

    def batch_as_tensors(self, start: int, end: int) -> dict[str, np.ndarray]:
        examples = [self.example_to_tensors(index) for index in range(start, end)]
        padding_index = self.vocab[SSTClassificationDataset.PAD]
        return {
            "review": pad_batch(
                [example["review"] for example in examples], padding_index
            ),
            "label": np.array([example["label"] for example in examples]),
            "lengths": np.array([len(example["review"]) for example in examples]),
        }

    @classmethod
    def from_files(cls, reviews_file: str, labels_file: str, vocab: Vocabulary = None):
        with open(reviews_file, "r") as reviews, open(labels_file, "r") as labels:
            review_lines = reviews.readlines()
            label_lines = labels.readlines()
        examples = [
            {
                # review is text, stored as a list of tokens
                "review": review_lines[line].strip("\n").split(" "),
                "label": int(label_lines[line].strip("\n")),
            }
            for line in range(len(review_lines))
        ]
        # initialize a vocabulary from the reviews, if none is given
        if not vocab:
            vocab = Vocabulary.from_text_files(
                [reviews_file],
                special_tokens=(Vocabulary.UNK, SSTClassificationDataset.PAD),
            )
        return cls(examples, vocab)
