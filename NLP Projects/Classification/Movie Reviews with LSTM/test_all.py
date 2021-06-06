import torch
import numpy as np

from data import pad_batch
from model import VanillaRNNCell, LSTMCell


def test_padding():
    batch = [
        np.array([1.0, 2.0]),
        np.array([1.0, 2.0, 3.0, 4.0, 5.0]),
        np.array([3.0]),
        np.array([1.0, 2.0, 3.0]),
    ]
    # index 0
    padded = pad_batch(batch, 0)
    np.testing.assert_allclose(
        padded,
        np.array(
            [
                [1.0, 2.0, 0.0, 0.0, 0.0],
                [1.0, 2.0, 3.0, 4.0, 5.0],
                [3.0, 0.0, 0.0, 0.0, 0.0],
                [1.0, 2.0, 3.0, 0.0, 0.0],
            ]
        ),
    )
    # index 11
    padded = pad_batch(batch, 11)
    np.testing.assert_allclose(
        padded,
        np.array(
            [
                [1.0, 2.0, 11.0, 11.0, 11.0],
                [1.0, 2.0, 3.0, 4.0, 5.0],
                [3.0, 11.0, 11.0, 11.0, 11.0],
                [1.0, 2.0, 3.0, 11.0, 11.0],
            ]
        ),
    )


class TestRNN:

    hidden_dim = 3
    embedding_dim = 3
    batch = np.array([[1.0, -2.0, 3.0], [0.5, -0.3, -1.0], [-1.1, 0.2, -0.1]])

    def test_vanilla_rnn(self):
        cell = VanillaRNNCell(TestRNN.hidden_dim, TestRNN.embedding_dim)
        cell.load_state_dict(torch.load("test_vanilla.pt"))
        output = (
            cell(torch.Tensor(TestRNN.batch), torch.Tensor(TestRNN.batch))
            .detach()
            .numpy()
        )
        np.testing.assert_allclose(
            output,
            np.array(
                [
                    [-0.9246, 0.9902, -0.9916],
                    [0.7450, -0.3092, 0.9505],
                    [-0.0487, -0.3007, 0.7462],
                ]
            ),
            atol=1e-4,
        )

    def test_lstm(self):
        cell = LSTMCell(TestRNN.hidden_dim, TestRNN.embedding_dim)
        cell.load_state_dict(torch.load("test_lstm.pt"))
        output, memory = cell(
            torch.Tensor(TestRNN.batch),
            torch.Tensor(TestRNN.batch),
            torch.Tensor(TestRNN.batch),
        )
        np.testing.assert_allclose(
            output.detach().numpy(),
            np.array(
                [
                    [0.4927, -0.0141, 0.7474],
                    [0.0008, -0.1224, -0.1754],
                    [-0.2102, 0.0106, -0.1920],
                ]
            ),
            atol=1e-4
        )
        np.testing.assert_allclose(
            memory.detach().numpy(),
            np.array(
                [
                    [0.9710, -0.0180, 1.2575],
                    [0.0014, -0.2666, -0.4490],
                    [-0.6839, 0.0206, -0.2812]
                ]
            ),
            atol=1e-4
        )