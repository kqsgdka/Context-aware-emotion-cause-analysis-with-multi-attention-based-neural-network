import torch
import torch.nn as nn
import numpy as np
from utils.attention import Attention
import torch.nn.functional as F
from icecream import ic

class MyLSTM(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, n_label, embedding_matrix, dropout=0.5):
        super(MyLSTM, self).__init__()
        self.embedding = nn.Embedding.from_pretrained(torch.tensor(embedding_matrix,dtype=torch.float))
        for param in self.embedding.parameters():
            param.requires_grad = False
        self.c_clause_lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=1, batch_first=True, bidirectional=True)
        self.e_clause_lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=1, batch_first=True, bidirectional=True)
        self.c_attention = Attention(embedding_dim, hidden_dim, score_function='bi_linear')
        self.e_attention = Attention(embedding_dim, hidden_dim, score_function='bi_linear')

# convolution
        self.conv2 = nn.Conv2d(1, 100, (2, embedding_dim))
        self.conv3 = nn.Conv2d(1, 100, (3, embedding_dim))
        self.conv4 = nn.Conv2d(1, 100, (4, embedding_dim))

        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(100 * 6 , n_label)

    def forward(self, x1, x2):
# embedding
        c_clause = self.embedding(x1)
        e_clause = self.embedding(x2)

        c_clause = c_clause.float()
        e_clause = e_clause.float()

# lstm层
        c_clause_out, (hn1, _) = self.c_clause_lstm(c_clause)
        e_clause_out, (hn2, _) = self.e_clause_lstm(e_clause) 

        hn1 = hn1.reshape(-1, c_clause_out.shape[-1])
        hn2 = hn1.reshape(-1, c_clause_out.shape[-1])  
# 正常的注意力机制
        _, c_scores = self.c_attention(c_clause_out, hn2)
        _, e_scores = self.c_attention(e_clause_out, hn1)
        c_scores = c_scores.transpose(1,2)
        e_scores = e_scores.transpose(1,2)
        output1 = c_clause_out * c_scores
        output2 = e_clause_out * e_scores

        output1 = output1.unsqueeze(1)
        output2 = output2.unsqueeze(1)

# CNN
# Convolution
        c1 = torch.tanh(self.conv2(output1)).squeeze(-1)
        c2 = torch.tanh(self.conv3(output1)).squeeze(-1)
        c3 = torch.tanh(self.conv4(output1)).squeeze(-1)
        ce1 = torch.tanh(self.conv2(output2)).squeeze(-1)
        ce2 = torch.tanh(self.conv3(output2)).squeeze(-1)
        ce3 = torch.tanh(self.conv4(output2)).squeeze(-1)
# pooling
        c1 = F.max_pool1d(c1, c1.size(2))
        c2 = F.max_pool1d(c2, c2.size(2))
        c3 = F.max_pool1d(c3, c3.size(2)) 
        ce1 = F.max_pool1d(ce1, ce1.size(2))
        ce2 = F.max_pool1d(ce2, ce2.size(2))
        ce3 = F.max_pool1d(ce3, ce3.size(2))
# capture and concatenate the features
        x = torch.cat((c1, c2, c3,ce1, ce2, ce3), 1)
        x=x.view(x.shape[0],-1)
        x = self.dropout(x)
        out = self.fc(x)
        return out
