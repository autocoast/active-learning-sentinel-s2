import numpy as np
import torch
from .strategy import Strategy

class EntropySamplingDropout(Strategy):
    def __init__(self, dataset, net, cfg):
        super(EntropySamplingDropout, self).__init__(dataset, net, cfg)

    def get_confidences(self, probs):
        # Reshape to Nx10x(512*512) for easier percentile computation
        uncertainties = probs.view(probs.shape[0], 10, -1)  # Shape: Nx10x(512*512)
        
        # Calculate entropy for each pixel and reshape back
        # Entropy H = -sum(p * log(p))
        entropy = -torch.sum(probs * torch.log(probs + 1e-10), dim=1)  # Shape: Nx512x512
        entropy = entropy.view(entropy.shape[0], -1)  # Flatten to Nx(512*512)
        
        # Compute 5th and 95th percentiles across flattened pixels
        p5 = torch.tensor(np.percentile(entropy.cpu().numpy(), 5, axis=1), device=entropy.device).unsqueeze(1)
        p95 = torch.tensor(np.percentile(entropy.cpu().numpy(), 95, axis=1), device=entropy.device).unsqueeze(1)

        # Filter values based on 5th and 95th percentiles
        mask = (entropy >= p5) & (entropy <= p95)
        filtered_entropy = torch.where(mask, entropy, torch.tensor(float('nan'), device=entropy.device))
        
        # Calculate mean and standard deviation of filtered values, ignoring NaNs
        filtered_mean = torch.nanmean(filtered_entropy, dim=1)
        diffs_squared = (filtered_entropy - filtered_mean.unsqueeze(1)) ** 2
        filtered_var = torch.nanmean(torch.where(mask, diffs_squared, torch.tensor(float('nan'), device=entropy.device)), dim=1)
        filtered_std = torch.sqrt(filtered_var)
        
        # Combine filtered mean and std to get the final uncertainty measure
        final_uncertainties = filtered_mean - filtered_std
        return final_uncertainties

    def query(self, n, cfg):
        unlabeled_idxs, unlabeled_data = self.dataset.get_unlabeled_data()
        probs = self.predict_prob_dropout(unlabeled_data)
        uncertainties = self.get_confidences(probs)
        return unlabeled_idxs[uncertainties.sort()[1][:n]]
