import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
from tqdm import tqdm

class Net:
    def __init__(self, net, params, device):
        self.net = net
        self.params = params
        self.device = device
        self.clf = self.net().to(self.device)
        
    def train(self, data):
        n_epoch = self.params['n_epoch']
        self.clf.train()
        optimizer = optim.SGD(self.clf.parameters(), **self.params['optimizer_args'])

        loader = DataLoader(data, shuffle=True, **self.params['train_args'])
        for epoch in tqdm(range(1, n_epoch+1), ncols=100):
            for batch_idx, (x, y, idxs) in enumerate(loader):
                x, y = x.to(self.device), y.to(self.device)
                optimizer.zero_grad()
                out = self.clf(x)
                loss = F.cross_entropy(out, y)
                loss.backward()
                optimizer.step()

    def predict(self, data):
        self.clf.eval()
        preds = torch.zeros(len(data), dtype=data.Y.dtype)
        loader = DataLoader(data, shuffle=False, **self.params['test_args'])
        with torch.no_grad():
            for x, y, idxs in loader:
                x, y = x.to(self.device), y.to(self.device)
                out = self.clf(x)
                if len(out.shape) == 1:
                    out = out.unsqueeze(0)
                pred = out.max(1)[1]
                preds[idxs] = pred.cpu()
        return preds
    
    def predict_prob(self, data):
        self.clf.eval()
        probs = torch.zeros([len(data), len(np.unique(data.Y))])
        loader = DataLoader(data, shuffle=False, **self.params['test_args'])
        with torch.no_grad():
            for x, y, idxs in loader:
                x, y = x.to(self.device), y.to(self.device)
                out = self.clf(x)
                if len(out.shape) == 1:
                    out = out.unsqueeze(0)
                prob = F.softmax(out, dim=1)
                probs[idxs] = prob.cpu()
        return probs
    
    def predict_prob_dropout(self, data, n_drop=10):
        self.clf.train()
        probs = torch.zeros([len(data), len(np.unique(data.Y))])
        loader = DataLoader(data, shuffle=False, **self.params['test_args'])
        for i in range(n_drop):
            with torch.no_grad():
                for x, y, idxs in loader:
                    x, y = x.to(self.device), y.to(self.device)
                    out = self.clf(x)
                    if len(out.shape) == 1:
                        out = out.unsqueeze(0)
                    prob = F.softmax(out, dim=1)
                    probs[idxs] += prob.cpu()
        probs /= n_drop
        return probs
    
    def predict_prob_dropout_split(self, data, n_drop=10):
        self.clf.train()
        probs = torch.zeros([n_drop, len(data), len(np.unique(data.Y))])
        loader = DataLoader(data, shuffle=False, **self.params['test_args'])
        for i in range(n_drop):
            with torch.no_grad():
                for x, y, idxs in loader:
                    x, y = x.to(self.device), y.to(self.device)
                    out = self.clf(x)
                    if len(out.shape) == 1:
                        out = out.unsqueeze(0)
                    prob = F.softmax(out, dim=1)
                    probs[i][idxs] += F.softmax(out, dim=1).cpu()
        return probs

class EuroSAT_Net(nn.Module):
    def __init__(self):
        super(EuroSAT_Net, self).__init__()
        self.net = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)
        self.net.conv1 = torch.nn.Conv2d(13, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.net.fc = torch.nn.Linear(512,10)

    def forward(self, x):
        x = self.net.conv1(x)
        x = self.net.bn1(x)
        x = self.net.relu(x)
        x = self.net.maxpool(x)
        x = self.net.layer1(x)
        x = self.net.layer2(x)
        x = self.net.layer3(x)
        x = self.net.layer4(x)
        x = self.net.avgpool(x).squeeze()
        x = F.dropout(x, p=0.3, training=self.training)
        y_pred = self.net.fc(x)
        return y_pred
