import torch
import torch.nn as nn
from transformers import BertModel

class BertTextClassfication(nn.Module):

    def __init__(self, bert_model, dropout_rate, num_labels):
        super().__init__()
        self.encoder = BertModel.from_pretrained(bert_model, add_pooling_layer=False)
        self.dropout = nn.Dropout(dropout_rate)
        self.clf = nn.Linear(self.encoder.config.hidden_size, num_labels)

    def forward(self, input_ids, token_type_ids, attention_mask, labels=None):
        embedding_res = self.encoder.embeddings(input_ids, token_type_ids, attention_mask)
        outputs = self.encoder.encoder(embedding_res)
        sequence_output = self.dropout(outputs.last_hidden_state)
        pool_output = torch.mean(sequence_output, axis=1)
        logits = self.clf(pool_output)
        if labels is not None:
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(logits, labels.reshape(-1))
        return loss if labels is not None else logits

if __name__ == "__main__":
    model = BertTextClassfication('junnyu/wobert_chinese_plus_base', dropout_rate=0.2, num_labels=4)
    print(model)
