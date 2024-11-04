import torch
import numpy as np
from .strategy import Strategy

class LeastConfidenceMeanPercentile(Strategy):
    def __init__(self, dataset, net):
        super(LeastConfidenceMeanPercentile, self).__init__(dataset, net)

    def get_confidences(self, probs):
        # Calculate max probability per pixel to get uncertainty
        uncertainties = probs.max(1)[0]  # Shape: Nx512x512
        uncertainties = uncertainties.view(probs.shape[0], -1)  # Shape: Nx(512*512)
        
        # Compute percentiles along each row and convert them to PyTorch tensors
        p5 = torch.tensor(np.percentile(uncertainties.cpu().numpy(), 5, axis=1), device=uncertainties.device).unsqueeze(1)
        p95 = torch.tensor(np.percentile(uncertainties.cpu().numpy(), 95, axis=1), device=uncertainties.device).unsqueeze(1)
        
        # Filter values in each row based on its specific 5th and 95th percentiles
        mask = (uncertainties >= p5) & (uncertainties <= p95)
        filtered_x = torch.where(mask, uncertainties, torch.tensor(float('nan'), device=uncertainties.device))
        
        # Calculate mean of filtered values, ignoring NaNs
        filtered_mean = torch.nanmean(filtered_x, dim=1)

        diffs = filtered_x - filtered_mean.unsqueeze(1)
        diffs_squared = diffs ** 2
        diffs_squared = torch.where(mask, diffs_squared, torch.tensor(float('nan'), device=uncertainties.device))
        filtered_var = torch.nanmean(diffs_squared, dim=1)
        filtered_std = torch.sqrt(filtered_var)
        
        # Combine the filtered mean and std for final uncertainty measure
        final_uncertainties = filtered_mean - filtered_std

        return final_uncertainties


    def query(self, n, cfg):
        unlabeled_idxs, unlabeled_data = self.dataset.get_unlabeled_data()
        probs = self.predict_prob(unlabeled_data)

        confidences = self.get_confidences(probs)
                
        # Return indices of selected samples based on sorted uncertainty values
        return unlabeled_idxs[confidences.argsort()[:n]]
