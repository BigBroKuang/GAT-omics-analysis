import torch.nn as nn
import torch
from config import Config
import csv
from collections import defaultdict
import pandas as pd 
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader


class LoadOmicsData():
    def __init__(self, config):
        self.config = config
        self.genes = []
        
        self.load_omics_label()
        
    def load_omics_label(self):

        gene_label_dict, label2index = self.load_label()
        csv_file = self.load_omics()

        self.genes = list(set(gene_label_dict.keys())&set(csv_file.index))

        self.labels = torch.zeros(len(self.genes), len(label2index))
        for gid,ge in enumerate(self.genes):
            label_idx = [label2index[e] for e in gene_label_dict[ge]]
            self.labels[gid, label_idx] = 1
            
        self.omics = torch.from_numpy(csv_file.loc[self.genes].values) #(N, feats)
        self.omics = self.omics.to(torch.float32)
        
        return self.train_test(self.labels)
    
    def load_label(self):
        genelabel = defaultdict(list)
        label2index = {}
        with open(self.config.label_path,'r', encoding = 'utf8') as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            for row in csv_reader:
                if len(row)>20:
                    label2index[row[0]] = len(label2index)
                    for gene in row[1:]:
                        genelabel[gene].append(row[0])    
        return genelabel,label2index
    
    def load_omics(self):
        csv_file = pd.read_csv(self.config.omics_path,sep=',',index_col =0)
        csv_file = np.log2(csv_file+1)
        #comment the syntax to use the raw data
        # quantile = csv_file.quantile(q=self.config.quantile, axis=1, numeric_only=True).to_numpy()
        # quantile = pd.Series(list(np.where(quantile>0, quantile, 1)), index=csv_file.index)
        
        csv_file = csv_file.fillna(0)
        #comment the syntax to use raw data
        # csv_file = csv_file.div(quantile, axis=0)

        return csv_file   
     
    def train_test(self, labels):
        train_test_size = train_test_split(range(len(self.genes)), train_size=self.config.train_size, shuffle=True, random_state= self.config.rseed)
        
        train_set = []
        test_set = []
        for idx,v in enumerate(train_test_size[0]):
            train_set.append((v, labels[v,:]))
        for idx,v in enumerate(train_test_size[1]):
            test_set.append((v, labels[v,:]))            
        train_iter = DataLoader(train_set, batch_size=self.config.batch_size)
        test_iter = DataLoader(test_set, batch_size=self.config.batch_size)   
        # for x_batch, y_batch in train_iter:
        #     print(x_batch.size(), y_batch.size())   
            
        return train_iter, test_iter

if __name__=="__main__":
    config = Config()
    data = LoadOmicsData(config)