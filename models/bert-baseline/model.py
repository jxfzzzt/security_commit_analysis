import torch.nn as nn
from config import *
from transformers import BertModel, AutoModel


class MessageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.bert = BertModel.from_pretrained(message_model_name)
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(768, 384)

    def forward(self, input_id, mask):
        _, pooled_output = self.bert(input_ids=input_id, attention_mask=mask, return_dict=False)
        dropout_output = self.dropout(pooled_output)
        linear_output = self.linear(dropout_output)
        return linear_output


class CodeModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.codebert = AutoModel.from_pretrained("microsoft/codebert-base")
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(768, 384)

    def forward(self, input_id, mask):
        _, pooled_output = self.codebert(input_ids=input_id, attention_mask=mask, return_dict=False)
        dropout_output = self.dropout(pooled_output)
        linear_output = self.linear(dropout_output)
        return linear_output


class SecurityCommitModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.message_encoder = MessageModel()
        self.code_encoder = CodeModel()
        self.dropout = nn.Dropout(dropout)
        self.linear1 = nn.Linear(768, 384)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(384, 2)

    def forward(self, message_input, message_mask, code_input, code_mask):
        message_embedding = self.message_encoder(message_input, message_mask)
        code_embedding = self.code_encoder(code_input, code_mask)
        embedding = torch.cat((message_embedding, code_embedding), dim=-1)
        X = self.dropout(embedding)
        X = self.linear1(X)
        X = self.relu(X)
        X = self.linear2(X)
        return X
