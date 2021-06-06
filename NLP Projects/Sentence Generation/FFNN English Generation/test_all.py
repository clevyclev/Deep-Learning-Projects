import torch
import numpy as np

from edugrad.ops import reduce_mean
from edugrad.tensor import Tensor

import data
import model
import ops
import run


def test_data():
    characters = ["a", "b", "c", "d", "e", "f"]
    # NOTE: this test assumes that you process the character list in left-to-right order
    examples = data.examples_from_characters(characters, 2)
    assert examples == [
        {"text": ["a", "b"], "target": "c"},
        {"text": ["b", "c"], "target": "d"},
        {"text": ["c", "d"], "target": "e"},
        {"text": ["d", "e"], "target": "f"},
    ]
    examples = data.examples_from_characters(characters, 3)
    assert examples == [
        {"text": ["a", "b", "c"], "target": "d"},
        {"text": ["b", "c", "d"], "target": "e"},
        {"text": ["c", "d", "e"], "target": "f"},
    ]


def test_tanh():
    batch = np.array([[1.0, 2.0], [3.0, 4.0]])
    edugrad_batch = Tensor(batch)
    torch_batch = torch.tensor(batch, requires_grad=True)
    # forward
    outputs = ops.tanh(edugrad_batch)
    torch_outputs = torch.tanh(torch_batch)
    np.testing.assert_allclose(outputs.value, torch_outputs.detach().numpy())
    # backward
    reduce_mean(outputs).backward()
    torch_outputs.mean().backward()
    np.testing.assert_allclose(edugrad_batch.grad, torch_batch.grad.numpy())


def test_model_forward():
    # build model and manually set weights
    lm = model.FeedForwardLanguageModel(2, 2, 2, 2)
    lm.embedding.weight.value = np.eye(2)
    # repeat identity matrix twice
    lm.fc.weight.value = np.tile(np.eye(2), (2, 1))
    lm.fc.bias.value = np.zeros(2)
    lm.output.weight.value = np.eye(2)
    lm.output.bias.value = np.zeros(2)
    # run batch through model
    batch = [Tensor(np.array([0, 1, 0])), Tensor(np.array([1, 1, 0]))]
    # embeddings (after concat) == [[1, 0, 0, 1], [0, 1, 0, 1], [1, 0, 1, 0]]
    # hidden (pre tanh) == [[1, 1], [0, 2], [2, 0]]
    np.testing.assert_allclose(
        lm(batch).value,
        np.array([[0.76156, 0.76156], [0.0, 0.96402], [0.96402, 0.0]]),
        rtol=1e-4,
    )


def test_next_character():
    # test with "fully peaked" probability distrubitions, so no real randomness in sampling
    probabilities = np.eye(3)
    np.testing.assert_allclose(run.sample_next_char(probabilities), np.array([0, 1, 2]))