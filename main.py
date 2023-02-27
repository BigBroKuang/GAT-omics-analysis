import torch.nn as nn
import torch
from config import Config
import os
import time
from copy import deepcopy
from OmicsProcess import LoadOmicsData
from Model import ClassificationModel
import numpy as np
from sklearn.metrics import hamming_loss

class traing_setting(nn.Module):
    def __init__(self, config):
        super(traing_setting, self).__init__()
        self.d_model  = torch.tensor(config.d_model)
        self.warmup = config.warmup
        self.step = 1.0
    def __call__(self):
        arg1 = self.step**(-0.5)
        arg2 = self.step*(self.warmup**(-1.5))
        self.step +=1.
        return (self.d_model**-0.5)*min(arg1, arg2)
    
def train(config):
    
    Data = LoadOmicsData(config)
    train_data, test_data = Data.load_omics_label()
    
    model = ClassificationModel(config, Data)
    model = model.to(config.device)
    
    loss_fn = nn.BCEWithLogitsLoss()
    learning_rate = traing_setting(config)
    optimizer = torch.optim.Adam(model.parameters(), lr=0., betas = (config.beta1, config.beta2), eps=config.epsilon)
    model.train()
    
    print('Start training...')
    for epoch in range(config.epoch):
        losses = 0
        start_time = time.time()
        for idx, (batch, label) in enumerate(train_data):

            batch = batch.to(config.device)
            label = label.to(config.device)
            batch = batch.tolist()
            logits = model(Data.omics, batch, label)
            
            optimizer.zero_grad()
            loss = loss_fn(logits, label)
            loss.backward()
            lr = learning_rate()
            for p in optimizer.param_groups:
                p['lr'] = lr
            optimizer.step()
            
            losses += loss.item()
            acc = accuracy(logits, label, config)
            
            if idx % 10 == 0:
                acc_test = evaluate(test_data, Data, model, config)
                print(f"Loss :{loss.item():.3f}, Train acc: {acc:.3f}, Test acc: {acc_test:.3f}")  
                

def evaluate(test_data, Data, model, config):
    model.eval()
    acc = 0
    n_sample = 0
    acc_sum = []
    with torch.no_grad():
        for batch, label in test_data:
            batch = batch.to(config.device)
            label = label.to(config.device)
            batch = batch.tolist()
            logits = model(Data.omics, batch, label)
            acc_sum.append(accuracy(logits, label, config))

        model.train()
    return np.mean(acc_sum)  

def accuracy(pred, target, config):
    pred = pred.detach().numpy()
    target = target.detach().numpy()
    pred = np.where(pred > config.threshold, 1, 0)
    return hamming_loss(target, pred)
    
if __name__ == '__main__':
    config = Config()
    train(config)
    