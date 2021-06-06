import torch
import torch.nn as nn
from transformers import BertModel


class PretrainedClassifier(nn.Module):
    """A PretrainedClassifier will use a pretrained BertModel to do sequence classification.

    In particular, it will extract the representation
    of the [CLS] token and then pass it through a linear layer, in order to
    predict some labels.

    Attributes:
        num_labels: how many labels in the task
        hidden_size: the size of the BertModel's representations
        encoder: the pretrained BertModel
        output: a Linear layer for making final predictions (returns logits)
    """

    def __init__(self, encoder: BertModel, num_labels: int):
        super(PretrainedClassifier, self).__init__()
        self.num_labels = num_labels
        self.hidden_size = encoder.config.hidden_size
        self.encoder = encoder
        # TODO (~1 line): create a linear output layer that will consume
        # a representation from self.encoder, and produce logits over the labels
        # Note: store this linear layer as self.output
        self.output = nn.Linear(self.hidden_size, num_labels)

    def forward(
        self,
        tokens: torch.LongTensor,
        attention_mask: torch.Tensor,
        token_type_ids: torch.LongTensor,
    ) -> torch.Tensor:
        """Get the [CLS] representation out of the encoder,
        and use the output layer to get logits.
        """
        # TODO (~6-10 lines, depending on formatting etc): implement forward!
        # This method works in a few steps:
        # 1. Get the embeddings for [CLS] from the top layer of encoder (a BertModel)
        # Note that step (1) can be done in more than one way.
        # Please do carefully read the documentation for .forward() of BertModel here:
        # https://huggingface.co/transformers/model_doc/bert.html#bertmodel
        # The final output from step (1) should be [batch_size, hidden_size] shape
        # 2. Pass the [CLS] representations through your linear layer, to produce logits
        # The shape of the returned Tensor should be [batch_size, num_labels]
        embeds = self.encoder.forward(tokens,attention_mask=attention_mask,token_type_ids=token_type_ids)
        CLS_reps = embeds.last_hidden_state[:,0,:]
        out = self.output(CLS_reps)
        return out
