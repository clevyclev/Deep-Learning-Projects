from typing import Any

import torch
import numpy as np
from transformers import PreTrainedTokenizer

# type aliases
Example = dict[str, Any]


class Dataset:
    def __init__(self, examples: list[Example], tokenizer: PreTrainedTokenizer) -> None:
        self.examples = examples
        self.tokenizer = tokenizer

    def batch_as_tensors(self, start: int, end: int) -> dict[str, torch.Tensor]:
        raise NotImplementedError

    def __getitem__(self, idx: int) -> dict:
        return self.examples[idx]

    def __len__(self) -> int:
        return len(self.examples)


class SSTClassificationDataset(Dataset):

    labels_to_string = {0: "terrible", 1: "bad", 2: "so-so", 3: "good", 4: "excellent"}

    def batch_as_tensors(self, start: int, end: int) -> dict[str, torch.Tensor]:
        """Gets a batch of examples as Tensors, using the dataset's tokenizer.

        Args:
            start: index of start example
            end: index of end example

        Returns:
            dictionary, with four keys as follows:
            - "review": [batch_size, max_seq_len] integer Tensor,
                with token indices of the batch of reviews
            - "token_type_ids": [batch_size, max_seq_len] integer Tensor,
                all 0s, indicating that we only have one input sequence, not two
            - "attention_mask": [batch_size, max_seq_len] Tensor,
                mask for padding tokens
            - "label": [batch_size] integer Tensor, the gold labels
        """
        examples = self.examples[start:end]
        examples_inputs = self.tokenizer(
            [example["review"] for example in examples], padding=True
        )
        return {
            "review": torch.LongTensor(examples_inputs["input_ids"]),
            "token_type_ids": torch.LongTensor(examples_inputs["token_type_ids"]),
            "attention_mask": torch.Tensor(examples_inputs["attention_mask"]),
            "label": torch.LongTensor([example["label"] for example in examples]),
        }

    @classmethod
    def from_files(
        cls, reviews_file: str, labels_file: str, tokenizer: PreTrainedTokenizer
    ):
        """Build a new SSTClassificationDataset from files, with a specified tokenizer.

        Args:
            reviews_file: file with reviews, one per line
            labels_file: file with labels, one per line (matching the reviews)
            tokenizer: the tokenizer to use for this text

        Returns:
            a new SSTClassificationDataset
        """
        with open(reviews_file, "r") as reviews, open(labels_file, "r") as labels:
            review_lines = reviews.readlines()
            label_lines = labels.readlines()
        examples = [
            {
                # review is raw text
                "review": review_lines[line].strip("\n"),
                "label": int(label_lines[line].strip("\n")),
            }
            for line in range(len(review_lines))
        ]
        return cls(examples, tokenizer)