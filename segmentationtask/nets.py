import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
from tqdm import tqdm
import segmentation_models_pytorch as smp

class Net:
    def __init__(self, net, params, device):
        self.net = net
        self.params = params
        self.device = device
        self.clf = self.net().to(self.device)
        
    def train(self, data):
        n_epoch = self.params['n_epoch']
        #self.clf = self.net().to(self.device)
        self.clf.train()
        optimizer = optim.SGD(self.clf.parameters(), **self.params['optimizer_args'])

        loader = DataLoader(data, shuffle=True, **self.params['train_args'])
        for epoch in tqdm(range(1, n_epoch+1), ncols=100):
            for batch_idx, (x, y, idxs) in enumerate(loader):
                x, y = x.to(self.device), y.to(self.device)
                optimizer.zero_grad()
                out = self.clf(x)
                loss = F.cross_entropy(out, y, ignore_index=0)
                loss.backward()
                optimizer.step()

    def predict(self, data):
        self.clf.eval()
        preds = torch.zeros((len(data), 128, 128), dtype=torch.long)  # For pixel-wise predictions
        loader = DataLoader(data, shuffle=False, **self.params['test_args'])
        with torch.no_grad():
            for x, y, idxs in loader:
                x, y = x.to(self.device), y.to(self.device)
                out = self.clf(x)
                pred = out.max(1)[1]  # (N, 128, 128), pixel-wise prediction
                preds[idxs] = pred.cpu()
        return preds

    def predict_prob_dropout_split(self, data, n_drop=10):
        self.clf.train()  # Enable dropout by setting the model to train mode
        probs = torch.zeros([n_drop, len(data), 11, 128, 128])  # Assuming 11 classes and (N, 128, 128) image shape
        loader = DataLoader(data, shuffle=False, **self.params['test_args'])
        
        for i in range(n_drop):
            with torch.no_grad():
                for x, y, idxs in loader:
                    x, y = x.to(self.device), y.to(self.device)
                    out = self.clf(x)
                    prob = F.softmax(out, dim=1)  # Apply softmax for pixel-wise class probabilities
                    probs[i][idxs] += prob.cpu()  # Accumulate probabilities across multiple dropout iterations
    
        probs = probs[:, :, 1:, :, :]  # Remove the first class if needed (like in your original code)
        return probs


    def predict_prob(self, data):
        self.clf.eval()
        # Adjusted to return pixel-wise probabilities (N, 11, 128, 128)
        probs = torch.zeros([len(data), 11, 128, 128])
        loader = DataLoader(data, shuffle=False, **self.params['test_args'])
        with torch.no_grad():
            for x, y, idxs in loader:
                x, y = x.to(self.device), y.to(self.device)
                out = self.clf(x)
                prob = F.softmax(out, dim=1)  # (N, 11, 128, 128), pixel-wise softmax
                probs[idxs] = prob.cpu()
        probs = probs[:, 1:, :, :]
        return probs

    def predict_prob_dropout(self, data, n_drop=10):
        self.clf.train()  # Enable dropout by setting the model to train mode
        probs = torch.zeros([len(data), 11, 128, 128])  # Assuming 11 classes and (N, 128, 128) image shape
        loader = DataLoader(data, shuffle=False, **self.params['test_args'])
        
        for i in range(n_drop):
            with torch.no_grad():
                for x, y, idxs in loader:
                    x, y = x.to(self.device), y.to(self.device)
                    out = self.clf(x)
                    prob = F.softmax(out, dim=1)  # Apply softmax for pixel-wise class probabilities
                    probs[idxs] += prob.cpu()  # Accumulate probabilities across multiple dropout iterations
    
        probs /= n_drop  # Average probabilities over all dropout runs
        probs = probs[:, 1:, :, :]  # Remove the first class if needed (like in your original code)
        return probs
        
class DW_Net(nn.Module):
    def __init__(self):
        super(DW_Net, self).__init__()
        self.model = smp.Unet(
            encoder_name="resnet50",        
            encoder_weights="imagenet",
            in_channels=13,                
            classes=11                   
        )
        
    def forward(self, x):
        encoder_features = self.model.encoder(x)

        encoder_features = list(encoder_features)
        encoder_features[-1] = F.dropout(encoder_features[-1], p=0.5, training=self.training)
        
        y_pred = self.model.decoder(*encoder_features)
        y_pred = self.model.segmentation_head(y_pred)

        return y_pred