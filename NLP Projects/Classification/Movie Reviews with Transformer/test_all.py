import torch
from transformers import BertModel, BertTokenizer

from model import PretrainedClassifier


class TestClassifier:
    sentences = [
        "here is the first sentence.",
        "the second sentence has a different length.",
    ]
    model_string = "google/bert_uncased_L-2_H-128_A-2"

    def test_forward(self):
        encoder = BertModel.from_pretrained(TestClassifier.model_string)
        model = PretrainedClassifier(encoder, 4)
        model.load_state_dict(torch.load("test_model.pt"))
        tokenizer = BertTokenizer.from_pretrained(TestClassifier.model_string)
        example_inputs = tokenizer(TestClassifier.sentences, padding=True)
        example_outputs = model(
            torch.LongTensor(example_inputs["input_ids"]),
            torch.Tensor(example_inputs["attention_mask"]),
            torch.LongTensor(example_inputs["token_type_ids"]),
        )
        gold_outputs = torch.load("test_outputs.pt")
        assert torch.allclose(example_outputs, gold_outputs)
