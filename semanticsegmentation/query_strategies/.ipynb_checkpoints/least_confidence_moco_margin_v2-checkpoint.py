import numpy as np
from scipy.spatial.distance import cdist
from .strategy import Strategy
from sklearn.neighbors import NearestNeighbors
import torch


class LeastConfidenceMocoMarginV2(Strategy):
    def __init__(self, dataset, net):
        super(LeastConfidenceMocoMarginV2, self).__init__(dataset, net)
        self.embeddings = np.load(f'/work/gg0877/g260217/al_paper/eurosat_s2_al/moco_dw/2048/128_2048_embedding.npy')

    def get_uncertainties(self, probs):
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
        
        # Return indices of selected samples based on sorted uncertainty values
        return final_uncertainties
    
    def query(self, n, cfg):
        unlabeled_idxs, unlabeled_data = self.dataset.get_unlabeled_data()
        
        probs = self.predict_prob(unlabeled_data) # Shape Nx11x128x128

        predominant = torch.argmax(probs, dim=1)  # Nx128x128
        predominant_classes = []
        
        predicted_classes = torch.argmax(probs, dim=1).view(probs.shape[0], -1) # N x 128*128
        
        for i in range(predicted_classes.shape[0]):  # Iterate over each image
            # comput the predominant class per image
            unique, counts = torch.unique(predicted_classes[i], return_counts=True)
            predominant_class = unique[torch.argmax(counts)]
            predominant_classes.append(predominant_class.item())  # Store predominant class for each image

           
        predominant_classes = np.array(predominant_classes)

        print(np.unique(predicted_classes, return_counts=True))

        knn = NearestNeighbors(n_neighbors=10).fit(self.embeddings[unlabeled_idxs])
        neighbors = knn.kneighbors(self.embeddings[unlabeled_idxs], return_distance=False)

        border_samples = []
        for ord_idx in range(predominant_classes.shape[0]):
            neighbor_idx = neighbors[ord_idx] # 10 nearest neighbors
            neighbor_labels = predominant_classes[neighbor_idx]

            current_label = predominant_classes[ord_idx]
            different_class_count = np.sum(neighbor_labels != current_label)
            if different_class_count / 10 >= 0.5:
                border_samples.append(ord_idx)
        
        print(f'{len(border_samples)} / {len(unlabeled_idxs)} at borders')
        border_selection = []
        if len(border_samples) > 0:
            border_selection = border_samples[:n]

        print(f'selected {len(border_selection)} from border decisions')
        
        uncertainties = self.get_uncertainties(probs)
        candidate_indices = unlabeled_idxs[uncertainties.sort()[1]]
        
        if len(border_selection) != n:
            remaining = n - len(border_selection)
            c = 0
            while len(border_selection) != n:
                if candidate_indices[c] not in border_selection:
                    border_selection.append(candidate_indices[c])
                c += 1

        

        return border_selection