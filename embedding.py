import torch.nn as nn
import torch
import math
from config import Config
import csv
from collections import defaultdict
import pandas as pd 
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

class embedding(nn.Module):
    def __init__(self, config, in_feat):
        super(embedding,self).__init__()
        
        self.ff = nn.Linear(in_feat, config.d_model)
        
        self.dropout = nn.Dropout(config.dropout)
        print(self.ff.weight.dtype)
        self._init_params()
        
    def _init_params(self):
        for p in self.parameters():
            if p.dim()>1:
                nn.init.xavier_uniform_(p)   
                
    def forward(self, omics):
        
        return  self.ff(omics)
        # return self.dropout(self.ff(omics))
