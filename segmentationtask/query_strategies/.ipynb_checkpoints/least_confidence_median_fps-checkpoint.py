import numpy as np
from .strategy import Strategy

class LeastConfidenceMedianFPS(Strategy):
    def __init__(self, dataset, net):
        super(LeastConfidenceMedianFPS, self).__init__(dataset, net)
        self.embeddings = self.get_embedding()

    def get_embedding(self):
        #embedding = np.load(f'/work/gg0877/g260217/al_paper/eurosat_s2_al/moco_dw/512_64_NTXentLossWithIndices_42_100/42_embeddings.npy')
        embedding = np.load(f'/work/gg0877/g260217/al_paper/eurosat_s2_al/moco_dw/2048/128_2048_embedding.npy')
        return embedding  # Shape Nx128x128

    def query(self, n, cfg):
        unlabeled_idxs, unlabeled_data = self.dataset.get_unlabeled_data()
        probs = self.predict_prob(unlabeled_data)
        uncertainties = probs.max(1)[0].view(probs.shape[0], -1).median(dim=1)[1] # <-- from shape Nx11x512x512 to Nx1
        sorted_indices = unlabeled_idxs[uncertainties.sort()[1][:n * 2]]

        selected_indices = farthest_point_sampling(self.embeddings[sorted_indices], 64)

        return sorted_indices[selected_indices]
    


# Helper function: Farthest Point Sampling (FPS)
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
