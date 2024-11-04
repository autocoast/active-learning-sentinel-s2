import numpy as np
import torch
from .strategy import Strategy

class LeastConfidenceEntropy(Strategy):
    def __init__(self, dataset, net):
        super(LeastConfidenceEntropy, self).__init__(dataset, net)

    def compute_entropy(self, probs):
        # Add a small epsilon to avoid log(0) issues
        print('B1')
        epsilon = 1e-10
        print('B2')
        entropy = -probs * torch.log(probs + epsilon)
        print('B3')
        entropy_maps = entropy.sum(dim=1)
        print('B4')
        mean_entropy = entropy_maps.mean(dim=(1, 2))
        print('B5')
        return mean_entropy  # Nx512x512, pixel-wise entropy
        
    
    def query(self, n, cfg):
        unlabeled_idxs, unlabeled_data = self.dataset.get_unlabeled_data()
        print("A")
        probs = self.predict_prob(unlabeled_data) # Nx11x512x512
        print("B")
        mean_entropy = self.compute_entropy(probs)
        print("C")
        sorted_indices = torch.argsort(mean_entropy, descending=True)
        print("D")
        return unlabeled_idxs[sorted_indices[:n]]