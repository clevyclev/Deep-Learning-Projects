import torch
import torch.nn as nn

from torch import Tensor


class VanillaRNNCell(nn.Module):
    def __init__(self, hidden_dim: int, embedding_dim: int):
        super(VanillaRNNCell, self).__init__()
        self.hidden2hidden = nn.Linear(hidden_dim, hidden_dim)
        self.input2hidden = nn.Linear(embedding_dim, hidden_dim)

    def forward(self, prev_hidden: Tensor, cur_input: Tensor) -> Tensor:
        """Compute one time step of a Vanilla RNN:
            h_t = tanh(h_{t-1}W_h + b_h + x_tW_x + b_x)

        Arguments:
            prev_hidden: [batch_size, hidden_dim] Tensor
                this corresponds to h_{t-1} in the eqn above
            cur_input: [batch_size, embedding_dim] Tensor
                this corresponds to x_t in the eqn above

        Returns:
            [batch_size, hidden_dim] Tensor
            this corresponds to h_t in the eqn above
        """
        # TODO: implement forward pass here! ~3-4 lines
        # Note: torch.tanh() will apply tanh element-wise
        stuffs = torch.tanh(self.hidden2hidden(prev_hidden) + self.input2hidden(cur_input))
        return stuffs


class VanillaRNN(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        embedding_dim: int,
        vocab_size: int,
        padding_index: int,
        dropout: float = None,
    ):
        super(VanillaRNN, self).__init__()
        self.embedding = nn.Embedding(
            vocab_size, embedding_dim, padding_idx=padding_index
        )
        self.cell = VanillaRNNCell(hidden_dim, embedding_dim)
        self.hidden_dim = hidden_dim
        self.dropout_prob = dropout
        if dropout:
            self.dropout = nn.Dropout(p=dropout)

    def init_hidden(self, batch_size: int) -> Tensor:
        return torch.zeros((batch_size, self.hidden_dim))

    def forward(self, sequences: Tensor) -> Tensor:
        """Loop through time and apply RNN cell.

        Arguments:
            sequences: [batch_size, seq_len]
            indices of tokens at each position

        Returns:
            [seq_len, batch_size, hidden_dim]
        """
        # get embeddings and transpose
        # [batch_size, seq_len, embedding_dim]
        embeddings = self.embedding(sequences)
        if self.dropout_prob:
            embeddings = self.dropout(embeddings)
        # [seq_len, batch_size, embedding_dim]
        embeddings = embeddings.transpose(0, 1)
        # initial hidden state
        batch_size = sequences.shape[0]
        # [batch_size, hidden_dim]
        hidden = self.init_hidden(batch_size)
        hidden_states = []
        for timestep in embeddings:
            # apply the recurrence
            # [batch_size, hidden_dim]
            hidden = self.cell(hidden, timestep)
            # store the hidden state
            hidden_states.append(hidden)
        # [seq_len, batch_size, hidden_dim]
        return torch.stack(hidden_states)


class LSTMCell(nn.Module):
    def __init__(self, hidden_dim: int, embedding_dim: int):
        super(LSTMCell, self).__init__()
        self.forget_linear = nn.Linear(hidden_dim + embedding_dim, hidden_dim)
        self.input_linear = nn.Linear(hidden_dim + embedding_dim, hidden_dim)
        self.candidate_linear = nn.Linear(hidden_dim + embedding_dim, hidden_dim)
        self.output_linear = nn.Linear(hidden_dim + embedding_dim, hidden_dim)

    def forward(
        self, prev_hidden: Tensor, prev_memory: Tensor, cur_input: Tensor
    ) -> tuple[Tensor, Tensor]:
        """One step of LSTM computation.  See the slides from April 21 for the LSTM equations.

        Arguments:
            prev_hidden: [batch_size, hidden_dim] h_{t-1}
            prev_memory: [batch_size, hidden_dim] c_{t-1}
            cur_input: [batch_size, embedding_dim] x_t

        Returns:
            two [batch_size, hidden_dim] tensors
            h_t, c_t
        """
        # first, we concatenate the previous hidden state and the current input
        # this produces a batch of the vectors denoted h_{t-1}x_t in the slides
        # each row corresponds to that vector for one example in the batch
        # [batch_size, hidden_dim + embeding_dim]
        combined_input = torch.cat((prev_hidden, cur_input), dim=1)
        # TODO: implement LSTM cell computation here.  ~6-7 lines
        # Note: torch.sigmoid() will apply sigmoid element-wise
        # If x and y are two torch.Tensors of the same shape, x*y will do
        # element-wise multiplication, and x+y addition
        #c_t = prev_memory * self.forget_linear(torch.sigmoid(combined_input))
        #c_t += self.input_linear(torch.sigmoid(combined_input)) * self.candidate_linear(torch.tanh(combined_input))
        #h_t = self.output_linear(torch.sigmoid(combined_input)) * torch.tanh(c_t)
        c_t = prev_memory * torch.sigmoid(self.forget_linear(combined_input))
        c_t += torch.sigmoid(self.input_linear(combined_input)) * torch.tanh(self.candidate_linear(combined_input))
        h_t = torch.sigmoid(self.output_linear(combined_input)) * torch.tanh(c_t)
        return h_t, c_t


class LSTM(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        embedding_dim: int,
        vocab_size: int,
        padding_index: int,
        dropout: float = None,
    ):
        super(LSTM, self).__init__()
        self.embedding = nn.Embedding(
            vocab_size, embedding_dim, padding_idx=padding_index
        )
        self.cell = LSTMCell(hidden_dim, embedding_dim)
        self.hidden_dim = hidden_dim
        self.dropout_prob = dropout
        if dropout:
            self.dropout = nn.Dropout(p=dropout)

    def init_hidden_and_memory(self, batch_size: int) -> tuple[Tensor, Tensor]:
        return [torch.zeros((batch_size, self.hidden_dim))] * 2

    def forward(self, sequences: Tensor) -> Tensor:
        """Loop through time and apply RNN cell.

        Arguments:
            sequences: [batch_size, seq_len]
            indices of tokens at each position

        Returns:
            [seq_len, batch_size, hidden_dim]
        """
        # get embeddings and transpose
        # [batch_size, seq_len, embedding_dim]
        embeddings = self.embedding(sequences)
        if self.dropout_prob:
            embeddings = self.dropout(embeddings)
        # [seq_len, batch_size, embedding_dim]
        embeddings = embeddings.transpose(0, 1)
        # initial hidden state
        batch_size = sequences.shape[0]
        # [batch_size, hidden_dim]
        hidden, memory = self.init_hidden_and_memory(batch_size)
        hidden_states = []
        for timestep in embeddings:
            # apply the recurrence
            # [batch_size, hidden_dim]
            hidden, memory = self.cell(hidden, memory, timestep)
            # store the hidden state
            hidden_states.append(hidden)
        # [seq_len, batch_size, hidden_dim]
        return torch.stack(hidden_states)


class RNNClassifier(nn.Module):
    def __init__(
        self, hidden_dim: int, output_size: int, rnn: nn.Module, dropout: float = None
    ):
        super(RNNClassifier, self).__init__()
        self.rnn = rnn
        self.output = nn.Linear(hidden_dim, output_size)
        self.hidden_dim = hidden_dim
        self.dropout_prob = dropout
        if dropout:
            self.dropout = nn.Dropout(p=dropout)

    def forward(self, sequences: Tensor, lengths: Tensor) -> Tensor:
        """Get true last hidden states out, pass through hidden layer.

        Arguments:
            sequences: [batch_size, seq_len]
            lengths: [batch_size]

        Returns:
            [batch_size, output_size]
        """
        # [seq_len, batch_size, hidden_dim]
        rnn_output = self.rnn(sequences)
        # get true last hidden states out; some painful torch indexing
        # [1, batch_size, 1]
        lengths = (lengths - 1).unsqueeze(0).unsqueeze(2)
        # [1, batch_size, hidden_dim]
        lengths = lengths.expand((1, -1, self.hidden_dim))
        # [1, batch_size, hidden_dim]
        final_hiddens = torch.gather(rnn_output, 0, lengths)
        # [batch_size, hidden_dim]
        final_hiddens = final_hiddens.squeeze()
        if self.dropout_prob:
            final_hiddens = self.dropout(final_hiddens)
        # pass through linear layer to get output
        # [batch_size, output_size]
        output = self.output(final_hiddens)
        return output
