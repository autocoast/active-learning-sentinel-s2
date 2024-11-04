import numpy as np
from .strategy import Strategy
from sklearn.cluster import KMeans

class LeastConfidenceMoCoKMeans(Strategy):
    def __init__(self, dataset, net):
        super(LeastConfidenceMoCoKMeans, self).__init__(dataset, net)
        self.embeddings = self.get_embedding()

    def get_embedding(self):
        embedding = np.load(f'/work/gg0877/g260217/al_paper/eurosat_s2_al/moco_dw/512_64_NTXentLossWithIndices_42_100/42_embeddings.npy')
        return embedding  # <- this has shape Nx128x128

    def query(self, n, cfg):
        unlabeled_idxs, unlabeled_data = self.dataset.get_unlabeled_data()
        probs = self.predict_prob(unlabeled_data)  # Shape Nx11x128x128
        uncertainties = probs.max(1)[0].view(probs.shape[0], -1).median(dim=1)[1]  # From shape Nx11x128x128 to Nx1

        sorted_indices = uncertainties.sort()[1]

        N_UNCERTAINTY = int(cfg.n_query * cfg.uncertainty_ratio)

        top = unlabeled_idxs[sorted_indices[:N_UNCERTAINTY]]

        if N_UNCERTAINTY == n:
            return top

        # Cluster the remaining samples based on MoCo embeddings using KMeans
        embeddings_top_uncertain = self.embeddings[sorted_indices[N_UNCERTAINTY:]]
        kmeans = KMeans(n_clusters=(n - N_UNCERTAINTY)).fit(embeddings_top_uncertain)
        
        # Get predicted labels from KMeans
        kmeans_labels = kmeans.labels_
        diverse_samples = []

        for cluster_label in np.unique(kmeans_labels):
            # Get indices of samples in this cluster
            cluster_indices = np.where(kmeans_labels == cluster_label)[0]
            
            # Get the embeddings of the samples in this cluster
            cluster_samples = embeddings_top_uncertain[cluster_indices]
            
            # Get the cluster center for this cluster from KMeans
            cluster_center = kmeans.cluster_centers_[cluster_label]
            
            # Compute the Euclidean distance between each sample and the cluster center
            distances = np.linalg.norm(cluster_samples - cluster_center, axis=1)
            
            # Find the index of the sample closest to the cluster center
            closest_idx = cluster_indices[np.argmin(distances)]
            
            # Add this sample to the diverse samples list
            diverse_samples.append(sorted_indices[N_UNCERTAINTY + closest_idx])

        final_selection = np.concatenate([top, np.array(diverse_samples)])

        return final_selection[:n]  # Return the final n selections
