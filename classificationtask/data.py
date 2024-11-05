import numpy as np
import torch
from torchvision import datasets
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture

class Data:
    def __init__(self, X_train, Y_train, X_test, Y_test, handler, cfg):
        self.X_train = X_train
        self.Y_train = Y_train
        self.X_test = X_test
        self.Y_test = Y_test
        self.handler = handler
        
        self.n_pool = len(X_train)
        self.n_test = len(X_test)

        print('pool', self.n_pool)
        print('test', self.n_test)
        
        self.labeled_idxs = np.zeros(self.n_pool, dtype=bool)
        
        if cfg.balance == True:
            self.embeddings = np.load(cfg.balance_embedding_path)
        else:
            self.embeddings = np.load(cfg.unbalanced_embedding_path)

    def kmeans_initialization(self, num, seed):
        # Perform k-means clustering on the embeddings
        kmeans = KMeans(n_clusters=num, random_state=seed).fit(self.embeddings)
        # Find the closest points to each cluster centroid
        idxs = []
        for center in kmeans.cluster_centers_:
            distances = np.sum((self.embeddings - center) ** 2, axis=1)
            idx = np.argmin(distances)
            idxs.append(idx)
        return np.array(idxs)    

    def gmm_initialization(self, num, seed):
        # Fit a Gaussian Mixture Model with `num` components
        gmm = GaussianMixture(n_components=num, random_state=seed).fit(self.embeddings)
        # Compute responsibilities for each component and each sample
        responsibilities = gmm.predict_proba(self.embeddings)
        # For each component, select the point with the highest responsibility
        idxs = []
        for i in range(num):
            component_responsibility = responsibilities[:, i]
            idx = np.argmax(component_responsibility)
            idxs.append(idx)
        return np.array(idxs)
        
    def initialize_labels(self, num, init_strategy, seed):
        if init_strategy == 'fps':
            idxs = farthest_point_sampling(self.embeddings, num)
            self.labeled_idxs[idxs] = True        
        elif init_strategy == 'kmeans':
            idxs = self.kmeans_initialization(num, seed)
            self.labeled_idxs[idxs] = True
        elif init_strategy == 'gmm':
            idxs = self.gmm_initialization(num, seed)
            self.labeled_idxs[idxs] = True
        else:
            # generate initial labeled pool
            tmp_idxs = np.arange(self.n_pool)
            np.random.shuffle(tmp_idxs)
            self.labeled_idxs[tmp_idxs[:num]] = True
    
    def get_labeled_data(self):
        labeled_idxs = np.arange(self.n_pool)[self.labeled_idxs]
        return labeled_idxs, self.handler(self.X_train[labeled_idxs], self.Y_train[labeled_idxs])
    
    def get_unlabeled_data(self):
        unlabeled_idxs = np.arange(self.n_pool)[~self.labeled_idxs]
        return unlabeled_idxs, self.handler(self.X_train[unlabeled_idxs], self.Y_train[unlabeled_idxs])
    
    def get_train_data(self):
        return self.labeled_idxs.copy(), self.handler(self.X_train, self.Y_train)
        
    def get_test_data(self):
        return self.handler(self.X_test, self.Y_test)
    
    def cal_test_acc(self, preds):
        return 1.0 * (self.Y_test==preds).sum().item() / self.n_test

def get_EuroSAT(handler, cfg):
    if cfg.balance:
        train_x = torch.load(cfg.balance_x_train_path)
        train_y = torch.load(cfg.balance_y_train_path)
    else:
        train_x = torch.load(cfg.unbalanced_x_train_path)
        train_y = torch.load(cfg.unbalanced_y_train_path)
        
    test_x = torch.load(cfg.x_test_path)
    test_y = torch.load(cfg.y_test_path)
    return Data(train_x, torch.LongTensor(train_y), test_x, torch.LongTensor(test_y), handler, cfg)

def farthest_point_sampling(data, num_samples, initial_indices=None, ignore_indices=None, seed=None):
    """
    Perform Farthest Point Sampling on a dataset.

    :param data: A NumPy array of shape (n_points, n_features)
    :param num_samples: The total number of samples to select, including initial samples
    :param initial_indices: Indices of the initial pre-selected samples
    :param ignore_indices: Indices of the points to ignore during sampling
    :param seed: Optional random seed for reproducibility
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
