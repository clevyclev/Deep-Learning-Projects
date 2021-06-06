from collections import Counter
from typing import Any, Callable

import numpy as np
from vocabulary import Vocabulary

# type aliases
Example = dict[str, Any]


class Dataset:
    def __init__(self, examples: list[Example], vocab: Vocabulary) -> None:
        self.examples = examples
        self.vocab = vocab
        self.num_labels = len(self.vocab)
        self._label_one_hots = np.eye(self.num_labels)

    def example_to_tensors(self, index: int) -> dict[str, np.ndarray]:
        raise NotImplementedError

    def batch_as_tensors(self, start: int, end: int) -> dict[str, np.ndarray]:
        raise NotImplementedError

    def __getitem__(self, idx: int) -> Example:
        return self.examples[idx]

    def __len__(self) -> int:
        return len(self.examples)


def examples_from_characters(chars: list[str], num_prev_chars: int) -> list[Example]:
    """Get a list of examples for character-level language modeling from a list of characters.

    Each Example is a dictionary, with two keys:
        "text" is a list of characters of length num_prev_chars
        "target" is a single character, the next one in the sequence

    An example usage:
        >>> examples_from_characters(['a', 'b', 'c', 'd'], 2)
        >>> [{"text": ['a', 'b'], "target": 'c'}, {"text": ['b', 'c'], "target": 'd'}]

    Arguments:
        chars: list of characters
        num_prev_chars: how many prevous characters to use in each Example

    Returns:
        list of Example dictionaries, as described above
    """
    # TODO: (~6-7 lines) implement here :)
    examples = []
    for i in range(len(chars) - num_prev_chars):
        examples.append({"text": chars[i:i + num_prev_chars],"target": chars[i + num_prev_chars]})
    return examples


class SSTLanguageModelingDataset(Dataset):

    BOS = "<s>"
    EOS = "</s>"

    def example_to_indices(self, index: int) -> dict[str, Any]:
        example = self.__getitem__(index)
        characters = self.vocab.tokens_to_indices(example["text"])
        return {"text": characters, "target": self.vocab[example["target"]]}

    def batch_as_tensors(self, start: int, end: int) -> dict[str, Any]:
        examples = [self.example_to_indices(index) for index in range(start, end)]
        example_length = len(examples[0]["text"])
        return {
            # text will be a list of [batch_size] arrays
            # list length = example length = num_prev_words
            # each element is the word at that position for each example in the batch
            "text": [
                np.array([example["text"][idx] for example in examples])
                for idx in range(example_length)
            ],
            # target: [batch_size, vocab_size], one hots for target character
            "target": np.stack(
                [self._label_one_hots[example["target"]] for example in examples]
            ),
        }

    @classmethod
    def from_file(cls, text_file: str, num_prev_chars: int, vocab: Vocabulary = None):
        examples = []
        counter: Counter = Counter()
        with open(text_file, "r") as reviews:
            for line in reviews:
                string = line.strip("\n")
                counter.update(string)
                # prepend BOS (num_prev_chars times) and EOS to each line
                chars = (
                    [SSTLanguageModelingDataset.BOS] * num_prev_chars
                    + list(string)
                    + [SSTLanguageModelingDataset.EOS]
                )
                examples.extend(examples_from_characters(chars, num_prev_chars))
        if not vocab:
            vocab = Vocabulary(
                counter,
                special_tokens=(
                    Vocabulary.UNK,
                    SSTLanguageModelingDataset.BOS,
                    SSTLanguageModelingDataset.EOS,
                ),
            )
        return cls(examples, vocab)
