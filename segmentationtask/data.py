import numpy as np
import torch
from torchvision import datasets
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from math import isnan


class Data:
    def __init__(self, X_train, Y_train, X_test, Y_test, handler, cfg):
        self.X_train = X_train
        self.Y_train = Y_train
        self.X_test = X_test
        self.Y_test = Y_test
        self.handler = handler
        
        self.n_pool = len(X_train)
        self.n_test = len(X_test)

        self.labeled_idxs = np.zeros(self.n_pool, dtype=bool)
        self.embedding = np.load(cfg.embedding_path)
        
    def initialize_labels(self, num, strategy, seed=1):
        if strategy == "fps":
            print(num)
            idxs = farthest_point_sampling(self.embedding, num)
            print(len(idxs))
            self.labeled_idxs[idxs] = True
        elif strategy == "gmm":
            idxs = gmm_sampling(self.embedding, num)
            self.labeled_idxs[idxs] = True
        elif strategy == "kmeans":
            #idxs = kmeans_sampling(self.embedding, num)
            idxs = self.kmeans_initialization(num, seed)
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
    
    '''
    def cal_test_acc(self, preds):
        return (preds.view(-1) == self.Y_test.view(-1)).sum().item() / preds.numel()
    
    def cal_test_acc(self, preds):
        valid_mask = self.Y_test.view(-1) != 0 # Y_test will have shape 128x128 and preds also
        return (preds.view(-1)[valid_mask] == self.Y_test.view(-1)[valid_mask]).sum().item() / valid_mask.sum().item()
    '''

    def cal_test_acc(self, preds):
        # Flatten predictions and ground truth
        preds_flat = preds.view(-1)
        y_test_flat = self.Y_test.view(-1)
        
        # Compute IoU for each class
        iou_per_class = []
        for cls in range(1, preds.max().item() + 1):  # Assuming class labels start from 1
            # Create masks for current class in predictions and ground truth
            pred_mask = preds_flat == cls
            true_mask = y_test_flat == cls
            
            # Calculate intersection and union
            intersection = (pred_mask & true_mask).sum().item()
            union = (pred_mask | true_mask).sum().item()
            
            if union == 0:  # Avoid division by zero
                iou_per_class.append(float('nan'))
            else:
                iou_per_class.append(intersection / union)
        
        # Calculate mean IoU, ignoring NaN values
        mean_iou = sum(iou for iou in iou_per_class if not isnan(iou)) / len(iou_per_class)
        
        return mean_iou


    def kmeans_initialization(self, num, seed):
        # Perform k-means clustering on the embeddings
        kmeans = KMeans(n_clusters=num, random_state=seed).fit(self.embedding)
        # Find the closest points to each cluster centroid
        idxs = []
        for center in kmeans.cluster_centers_:
            distances = np.sum((self.embedding - center) ** 2, axis=1)
            idx = np.argmin(distances)
            idxs.append(idx)
        return np.array(idxs)   



def get_DW(handler, cfg):
    train_x = torch.load(cfg.x_train_path)
    train_y = torch.load(cfg.y_train_path)
    test_x = torch.load(cfg.x_test_path)
    test_y = torch.load(cfg.y_test_path)
    return Data(train_x, train_y, test_x, test_y, handler, cfg)


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



def kmeans_sampling(data, num):
    kmeans = KMeans(n_clusters=num).fit(data)
        
    # Get predicted labels from KMeans
    kmeans_labels = kmeans.labels_
    diverse_samples = []

    for cluster_label in np.unique(kmeans_labels):
        # Get indices of samples in this cluster
        cluster_indices = np.where(kmeans_labels == cluster_label)[0]
        
        # Get the embeddings of the samples in this cluster
        cluster_samples = data[cluster_indices]
        
        # Get the cluster center for this cluster from KMeans
        cluster_center = kmeans.cluster_centers_[cluster_label]
        
        # Compute the Euclidean distance between each sample and the cluster center
        distances = np.linalg.norm(cluster_samples - cluster_center, axis=1)
        
        # Find the index of the sample closest to the cluster center
        closest_idx = cluster_indices[np.argmin(distances)]
        
        # Add this sample to the diverse samples list
        diverse_samples.append(closest_idx)

    return diverse_samples

def gmm_sampling(data, num):
    # Fit a Gaussian Mixture Model to the data
    gmm = GaussianMixture(n_components=num).fit(data)
    
    # Get the predicted labels from GMM (most probable component for each sample)
    gmm_labels = gmm.predict(data)
    diverse_samples = []

    for component_label in np.unique(gmm_labels):
        # Get indices of samples in this component
        component_indices = np.where(gmm_labels == component_label)[0]
        
        # Get the embeddings of the samples in this component
        component_samples = data[component_indices]
        
        # Get the mean of the Gaussian for this component
        component_mean = gmm.means_[component_label]
        
        # Compute the Euclidean distance between each sample and the component mean
        distances = np.linalg.norm(component_samples - component_mean, axis=1)
        
        # Find the index of the sample closest to the component mean
        closest_idx = component_indices[np.argmin(distances)]
        
        # Add this sample to the diverse samples list
        diverse_samples.append(closest_idx)

    return diverse_samples

