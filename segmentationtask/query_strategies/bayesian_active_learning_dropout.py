import numpy as np
import torch
from .strategy import Strategy

class BALDDropout(Strategy):
    def __init__(self, dataset, net, cfg, n_drop=10):
        super(BALDDropout, self).__init__(dataset, net, cfg)
        self.n_drop = n_drop

    def get_confidences(self, probs):
        # Average over the dropout dimension to get the mean probability distribution per pixel
        pb = probs.mean(0)  # Shape: [N, c, w, h]
        
        # Flatten spatial dimensions for percentile and entropy calculations
        pb_flat = pb.view(pb.shape[0], pb.shape[1], -1)  # Shape: [N, c, (w*h)]
        
        # Calculate entropy for averaged probabilities (Entropy 1)
        entropy1 = -torch.sum(pb_flat * torch.log(pb_flat + 1e-10), dim=1)  # Shape: [N, (w*h)]

        # Calculate entropy for each individual dropout prediction (Entropy 2) and average across n_drop
        entropy2 = -torch.sum(probs * torch.log(probs + 1e-10), dim=2).mean(0)  # Shape: [N, w, h]
        entropy2 = entropy2.view(entropy2.shape[0], -1)  # Flatten to [N, (w*h)]

        # BALD uncertainty measure: difference between entropy2 and entropy1
        bald_uncertainty = entropy2 - entropy1  # Shape: [N, (w*h)]
        
        # Compute 5th and 95th percentiles for filtering
        p5 = torch.tensor(np.percentile(bald_uncertainty.cpu().numpy(), 5, axis=1), device=bald_uncertainty.device).unsqueeze(1)
        p95 = torch.tensor(np.percentile(bald_uncertainty.cpu().numpy(), 95, axis=1), device=bald_uncertainty.device).unsqueeze(1)

        # Apply percentile mask
        mask = (bald_uncertainty >= p5) & (bald_uncertainty <= p95)
        filtered_uncertainty = torch.where(mask, bald_uncertainty, torch.tensor(float('nan'), device=bald_uncertainty.device))
        
        # Calculate mean and standard deviation of filtered values, ignoring NaNs
        filtered_mean = torch.nanmean(filtered_uncertainty, dim=1)
        diffs_squared = (filtered_uncertainty - filtered_mean.unsqueeze(1)) ** 2
        filtered_var = torch.nanmean(torch.where(mask, diffs_squared, torch.tensor(float('nan'), device=bald_uncertainty.device)), dim=1)
        filtered_std = torch.sqrt(filtered_var)
        
        # Final uncertainty measure
        final_uncertainties = filtered_mean - filtered_std
        return final_uncertainties

    def query(self, n, cfg):
        unlabeled_idxs, unlabeled_data = self.dataset.get_unlabeled_data()
        # Perform multiple stochastic forward passes with dropout
        probs = self.predict_prob_dropout_split(unlabeled_data, n_drop=self.n_drop)  # Shape: [n_drop, N, c, w, h]
        uncertainties = self.get_confidences(probs)
        return unlabeled_idxs[uncertainties.sort()[1][:n]]
