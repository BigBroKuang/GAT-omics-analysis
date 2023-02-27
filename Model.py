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
from embedding import embedding
from Attention import EncoderLayer
from loadnet import LoadNetwork

class ClassificationModel(nn.Module):
    def __init__(self, config, Data):
        super(ClassificationModel, self).__init__()
        self.config = config
       
        self.embds =  embedding(config, Data.omics.size(1))
        self.encoder = EncoderLayer(config)
        self.net = LoadNetwork(config, Data.genes)
        self.classifier = nn.Sequential(nn.Linear(config.d_model, config.dim_classification),
                                        nn.Dropout(config.dropout),
                                        nn.Linear(config.dim_classification, Data.labels.size(1)))
        self._reset_params()
        
    def _reset_params(self):
        for p in self.parameters():
            if p.dim()>1:
                nn.init.xavier_uniform_(p)    
                
    def forward(self, omics, batch, label):
        
        omics_data = self.embds(omics)
        neighbor = self.net.sub_net(batch)
        out = self.encoder(batch, neighbor, omics_data)
        out = self.classifier(out)
        return out 
        
        
        
        
        