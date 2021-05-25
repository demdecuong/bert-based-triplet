import torch
import torch.nn as nn

from transformers import AutoModel, AutoTokenizer, AdamW, get_linear_schedule_with_warmup

class JointSpace(nn.Module):
    def __init__(self, hidden_size, n_dim):
        super(JointSpace, self).__init__()
        self.hidden_size = hidden_size
        self.out1 = nn.Linear(hidden_size, hidden_size // 2)
        self.out2 = nn.Linear(hidden_size // 2, n_dim)

    def forward(self, x):
        out = self.out1(x)
        out = self.out2(out)
        return out


class SimilarityClassifier(nn.Module):
    def __init__(self, PRE_TRAINED_MODEL_NAME, embed_dim, dropout_p, freeze=False):
        super(SimilarityClassifier, self).__init__()
        # self.bert = DistilBertModel.from_pretrained(PRE_TRAINED_MODEL_NAME)
        self.bert = AutoModel.from_pretrained(PRE_TRAINED_MODEL_NAME)
        self.drop = nn.Dropout(p=0.3)
        self.sigmoid = nn.Sigmoid()
        self.space_joiner = JointSpace(self.bert.config.hidden_size, embed_dim)
        if freeze:
            for p in self.bert.parameters():
                p.requires_grad = False

    def forward(self, input_ids, attention_mask):
        bert_output = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )[0]
        bert_output = torch.mean(bert_output, dim=1)
        output = self.drop(bert_output)
        out = self.space_joiner(output)
        # return self.sigmoid(out)

        return out

def get_model(config):
    model = SimilarityClassifier(config.PRE_TRAINED_MODEL_NAME, config.embed_dim, config.dropout,config.freeze)
    return model