import torch.nn as nn
import torch
import math
from torch.nn.init import xavier_uniform_
import config
import numpy as np
import torch.nn.functional as F

# class Encoders(nn.Module):
#     def __init__(self, config):
#         super(Encoders, self).__init__()
#         self.layers = nn.ModuleList([EncoderLayer(config) for _ in range(config.num_encoder_layers)])
#         self.num_layers = config.num_encoder_layers
        
#     def forward(self, batch, neighbor, embeddings):
#         output = embeddings
#         for encoder in self.layers:
#             output = encoder(batch, neighbor, embeddings)
#         return output 
    
class EncoderLayer(nn.Module):
    def __init__(self, config):
        super(EncoderLayer, self).__init__()
        self.atten = MultiheadAttention(config)
        
        self.dropout1 = nn.Dropout(config.dropout)
        self.dropout2 = nn.Dropout(config.dropout)
        self.dropout3 = nn.Dropout(config.dropout)
        
        
        self.linear1 = nn.Linear(config.d_model, config.dim_feedforward)
        self.linear2 = nn.Linear(config.dim_feedforward, config.d_model)
        self.activation = F.relu
        
        self.norm1 = nn.LayerNorm(config.d_model)
        self.norm2 = nn.LayerNorm(config.d_model)
       
    def forward(self, batch, neighbor, embeddings):
        batch_embd = embeddings[batch,:]
        attout = self.atten(batch, neighbor, embeddings)
        #ResNet
        sum_embds = self.norm1(batch_embd + self.dropout1(attout))
        #FF, ResNet
        attout = self.activation(self.linear1(sum_embds))
        attout = self.activation(self.linear2(self.dropout2(attout)))
        sum_embds =self.norm2(sum_embds + self.dropout3(attout))
        return sum_embds
    
class MultiheadAttention(nn.Module):
    def __init__(self, config):
        super(MultiheadAttention, self).__init__()
        
        self.d_model = config.d_model
        self.head_dim = config.d_model//config.num_head
        self.num_head = config.num_head
        self.dropout = config.dropout
        
        self.q = nn.Linear(config.d_model, config.d_model)
        self.k = nn.Linear(config.d_model, config.d_model)
        self.v = nn.Linear(config.d_model, config.d_model)
        self.att_out = nn.Linear(config.d_model, config.d_model)
        
        self._reset_params()
        
    def _reset_params(self):
        for v in self.parameters():
            if v.dim()>1:
                xavier_uniform_(v)
     
    def forward(self, batch, neighbor, embeddings):
        '''
        batch: list of training genes
        neighbors: dictionary, {gene1:{}, gene2:{}, gene3:{}, ...}
        embeddings (N_nodes, d_model)
        '''

        #add the gene itself to the neighbor list
        neis_ = [[k]+neighbor[k] for k in batch]
        max_nei_len = max([len(v) for v in neis_])
        #neighbors to array
        neis_ = np.array([v+[np.nan]*(max_nei_len-len(v)) for v in neis_], dtype=float)
        
        #create mask
        nei_masked = torch.Tensor(np.where(neis_>=0, False, True))
        nei_masked = nei_masked.bool()
        #replace nan with an empty embd
        neis_ = np.where(neis_>=0, neis_, 0)
        
        
        batch_embd = embeddings[batch,:] #(batch, d_model)
        nei_embd = embeddings[neis_,:] #(batch, max_nei, d_model)

        q_cal = self.q(batch_embd) #(batch, d_model)
        k_cal = self.k(nei_embd) #(batch, max_nei, d_model)
        v_cal = self.v(nei_embd) #(batch, max_nei, d_model)

        scaling_factor = float(self.head_dim)**(-0.5)
        q_cal = q_cal*scaling_factor

        
        # #expand the dimension of batch
        # #(batch, dim) -> (batch*num_head,  head_dim) -> (batch*num_head, 1, head_dim)
        q_cal = q_cal.reshape(-1, self.head_dim).unsqueeze(1)
        #(batch, max_nei, d_model) -> (max_nei, batch, d_model) -> (max_nei, batch*nheads, d_model) -> (batch*nheads, max_nei, d_model) 
        k_cal = k_cal.transpose(0,1).reshape(-1, len(batch)*self.num_head, self.head_dim).transpose(0, 1)
        v_cal = v_cal.transpose(0,1).reshape(-1, len(batch)*self.num_head, self.head_dim).transpose(0, 1)
        # #(batch*num_head, 1, head_dim)*(batch*num_head, max_nei, head_dim) -> (batch*num_head, 1, max_nei) 
        atten_output = torch.bmm(q_cal, k_cal.transpose(1, 2))
        #(batch*num_head, 1, max_nei) -> (batch, num_head, 1, max_nei)
        atten_output = atten_output.reshape(len(batch), self.num_head, 1, max_nei_len)
        atten_output = atten_output.masked_fill_(nei_masked.unsqueeze(1).unsqueeze(2), -np.inf)
        #(batch, num_head, 1, max_nei) -> (batch*num_head, 1, max_nei)
        atten_output = atten_output.reshape(len(batch)*self.num_head, 1, max_nei_len)
        atten_output = F.softmax(atten_output, dim=-1)
        atten_output = F.dropout(atten_output, p=self.dropout, training = self.training)
        #(batch*num_head, 1, max_nei)*(batch*nheads, max_nei, d_model)  -> (batch*num_head, 1, d_model) -> (batch*num_head, d_model)
        atten_output = torch.bmm(atten_output, v_cal).squeeze(1)
        atten_output = atten_output.reshape(len(batch), self.d_model)

        return self.att_out(atten_output)


# if __name__ == '__main__':
#     config = config.Config()
#     # mha =  Encoders(config)
#     embds = torch.randn(10,512)
#     batch = [0,1,3]
#     neis = {0:[1,2],1:[2,3],3:[1,5, 6]}
#     mha(batch, neis, embds)
    
    

