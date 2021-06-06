import argparse
import copy
import random

import torch
import numpy as np
from tqdm import tqdm
from transformers import BertModel, BertTokenizer

from data import SSTClassificationDataset
from model import PretrainedClassifier


def accuracy(logits: torch.Tensor, labels: torch.LongTensor) -> float:
    """Computes accuracy of a set of predictions.

    Args:
        logits: [batch_size, num_labels], model predictions
        labels: [batch_size], indices for gold labels

    Returns:
        percentage of correct predictions, where model prediction is the
        argmax of its probabilities
    """
    # [batch_size]
    predictions = logits.argmax(axis=1)
    return (predictions == labels).float().mean()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=575)
    parser.add_argument("--num_epochs", type=int, default=12)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--l2", type=float, default=0.0)
    parser.add_argument(
        "--model", type=str, default="google/bert_uncased_L-2_H-128_A-2"
    )
    parser.add_argument(
        "--train_reviews",
        type=str,
        default="/dropbox/20-21/575k/data/sst/train-reviews.txt",
    )
    parser.add_argument(
        "--train_labels",
        type=str,
        default="/dropbox/20-21/575k/data/sst/train-labels.txt",
    )
    parser.add_argument(
        "--dev_reviews",
        type=str,
        default="/dropbox/20-21/575k/data/sst/dev-reviews.txt",
    )
    parser.add_argument(
        "--dev_labels", type=str, default="/dropbox/20-21/575k/data/sst/dev-labels.txt"
    )
    parser.add_argument(
        "--num_dev_examples",
        type=int,
        default=10,
        help="How many random dev examples to inspect at the end of training",
    )
    args = parser.parse_args()

    # set seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # get tokenizer and datasets
    tokenizer = BertTokenizer.from_pretrained(args.model)

    # build datasets
    sst_train = SSTClassificationDataset.from_files(
        args.train_reviews, args.train_labels, tokenizer
    )
    sst_dev = SSTClassificationDataset.from_files(
        args.dev_reviews, args.dev_labels, tokenizer
    )
    # dev data as np arrays
    dev_data = sst_dev.batch_as_tensors(0, len(sst_dev))

    # get model
    encoder = BertModel.from_pretrained(args.model)
    model = PretrainedClassifier(
        encoder, len(SSTClassificationDataset.labels_to_string)
    )

    # get training things set up
    data_size = len(sst_train)
    batch_size = args.batch_size
    starts = list(range(0, data_size, batch_size))
    optimizer = torch.optim.AdamW(model.parameters(), weight_decay=args.l2, lr=args.lr)
    best_loss = float("inf")
    best_model = None

    for epoch in range(args.num_epochs):
        running_loss = 0.0
        # shuffle batches
        random.shuffle(starts)
        for start in tqdm(starts):
            batch = sst_train.batch_as_tensors(
                start, min(start + batch_size, data_size)
            )
            # get probabilities and loss
            # [batch_size, num_labels]
            logits = model(
                batch["review"], batch["attention_mask"], batch["token_type_ids"]
            )
            # scalar
            loss = torch.nn.functional.cross_entropy(logits, batch["label"])
            running_loss += loss.item()

            # get gradients and update weights
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch} train loss: {running_loss / len(starts)}")

        # get dev loss every epoch
        logits = model(
            dev_data["review"], dev_data["attention_mask"], dev_data["token_type_ids"]
        )
        epoch_loss = torch.nn.functional.cross_entropy(logits, dev_data["label"]).item()
        print(f"Epoch {epoch} dev loss: {epoch_loss}")

        if epoch_loss < best_loss:
            best_loss = epoch_loss
            print("New best loss; saving current model")
            best_model = copy.deepcopy(model)

    # get dev accuracy at the very end
    with torch.no_grad():
        logits = best_model(
            dev_data["review"], dev_data["attention_mask"], dev_data["token_type_ids"]
        )
    dev_accuracy = accuracy(logits, dev_data["label"])
    print(f"\nBest model dev accuracy: {dev_accuracy}")

    # look at some examples
    if args.num_dev_examples:
        print("\nSome dev examples, with model predictions:\n")
        data_indices = np.random.randint(len(sst_dev), size=args.num_dev_examples)
        for index in data_indices:
            datum = sst_dev[index]
            prediction = logits[index].argmax()
            print(
                f"Review:\t{datum['review']}\nGold label:\t{datum['label']}\nModel prediction:\t{prediction}\n"
            )