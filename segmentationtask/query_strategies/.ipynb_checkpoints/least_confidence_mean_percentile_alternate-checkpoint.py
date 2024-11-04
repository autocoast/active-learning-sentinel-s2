import torch
import numpy as np
from .strategy import Strategy
from sklearn.cluster import KMeans

class LeastConfidenceMeanPercentileAlternate(Strategy):
    def __init__(self, dataset, net):
        super(LeastConfidenceMeanPercentileAlternate, self).__init__(dataset, net)
        self.embedding = np.load(f'/work/gg0877/g260217/al_paper/eurosat_s2_al/moco_dw/2048/128_2048_embedding.npy')

    def query(self, n, cfg):
        labeled_idxs, labeled_data = self.dataset.get_labeled_data()
        #labeled_idx ~> shape 64

        N = 64

        if len(labeled_idxs) % (N * 2) == 0:
            # k-means sampling
            unlabeled_idxs, unlabeled_data = self.dataset.get_unlabeled_data()
            M = self.embedding[unlabeled_idxs]
            cluster_learner = KMeans(n_clusters=64)
            cluster_learner.fit(M)
            
            cluster_idxs = cluster_learner.predict(M)
            centers = cluster_learner.cluster_centers_[cluster_idxs]
            dis = (M - centers)**2
            dis = dis.sum(axis=1)
            q_idxs = np.array([np.arange(M.shape[0])[cluster_idxs==i][dis[cluster_idxs==i].argmin()] for i in range(n)])
            
            return unlabeled_idxs[q_idxs]
            
        else:
            unlabeled_idxs, unlabeled_data = self.dataset.get_unlabeled_data()
            probs = self.predict_prob(unlabeled_data)
            
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
            return unlabeled_idxs[final_uncertainties.argsort()[:n]]
        
    def farthest_point_sampling(data, num_samples, initial_indices=None, ignore_indices=None, seed=None, half_farthest=False):
        """
        Perform Farthest Point Sampling on a dataset, with an option to select the farthest or half-farthest points.
    
        :param data: A NumPy array of shape (n_points, n_features)
        :param num_samples: The total number of samples to select, including initial samples
        :param initial_indices: Indices of the initial pre-selected samples
        :param ignore_indices: Indices of the points to ignore during sampling
        :param seed: Optional random seed for reproducibility
        :param half_farthest: If True, select points that are halfway between the closest and farthest
        :return: Indices of the sampled points
        """
    
        if seed is not None:
            np.random.seed(seed)
    
        n_points = data.shape[0]
        if initial_indices is None:
            initial_indices = []
        if ignore_indices is None:
            ignore_indices = []
    
        # If no initial indices are provided, choose a random initial point
        if len(initial_indices) == 0:
            candidate = np.random.choice(np.setdiff1d(np.arange(n_points), ignore_indices))
            initial_indices = [candidate]
    
        # Ensure the initial indices do not include ignored points
        initial_indices = [idx for idx in initial_indices if idx not in ignore_indices]
    
        # Initialize distances with infinity
        distances = np.full(n_points, np.inf)
    
        # Set distances for ignored points to negative infinity
        distances[ignore_indices] = -np.inf
    
        # Update distances based on initial samples
        for idx in initial_indices:
            dist_to_point = np.sum((data - data[idx]) ** 2, axis=1)
            distances = np.minimum(distances, dist_to_point)
    
        sampled_indices = initial_indices.copy()
    
        while len(sampled_indices) < num_samples:
            if half_farthest:
                # Sort indices by distance and choose a point near the middle
                valid_indices = np.where(distances != -np.inf)[0]
                sorted_indices = valid_indices[np.argsort(distances[valid_indices])]
                mid_idx = len(sorted_indices) // 2
                selected_idx = sorted_indices[mid_idx]
            else:
                # Select the farthest point
                selected_idx = np.argmax(distances)
    
            sampled_indices.append(selected_idx)
    
            # Update distances based on the newly added point
            new_point = data[selected_idx]
            dist_to_new_point = np.sum((data - new_point) ** 2, axis=1)
            distances = np.minimum(distances, dist_to_new_point)
    
            # Ensure ignored points remain ignored
            distances[ignore_indices] = -np.inf
    
        return sampled_indices
