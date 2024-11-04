import numpy as np
from .strategy import Strategy
from sklearn.mixture import GaussianMixture

class LeastConfidenceMoCoGMM(Strategy):
    def __init__(self, dataset, net):
        super(LeastConfidenceMoCoGMM, self).__init__(dataset, net)
        self.embeddings = self.get_embedding()

    def get_embedding(self):
        embedding = np.load(f'/work/gg0877/g260217/al_paper/eurosat_s2_al/moco_dw/512_64_NTXentLossWithIndices_42_100/42_embeddings.npy')
        return embedding  # Shape Nx128x128

    def query(self, n, cfg):
        unlabeled_idxs, unlabeled_data = self.dataset.get_unlabeled_data()
        probs = self.predict_prob(unlabeled_data)  # Shape Nx11x128x128
        uncertainties = probs.max(1)[0].view(probs.shape[0], -1).median(dim=1)[1]  # From shape Nx11x128x128 to Nx1

        sorted_indices = uncertainties.sort()[1]
        
        N_UNCERTAINTY = int(cfg.n_query * cfg.uncertainty_ratio)

        top = unlabeled_idxs[sorted_indices[:N_UNCERTAINTY]]

        if N_UNCERTAINTY == n:
            return top

        # Cluster the remaining samples based on MoCo embeddings using GMM
        embeddings_top_uncertain = self.embeddings[sorted_indices[N_UNCERTAINTY:]]
        gmm = GaussianMixture(n_components=n - N_UNCERTAINTY).fit(embeddings_top_uncertain)
        
        # Get predicted labels from GMM
        gmm_labels = gmm.predict(embeddings_top_uncertain)
        
        diverse_samples = []

        for cluster_label in np.unique(gmm_labels):
            # Get indices of samples in this cluster
            cluster_indices = np.where(gmm_labels == cluster_label)[0]
            
            # Get the embeddings of the samples in this cluster
            cluster_samples = embeddings_top_uncertain[cluster_indices]
            
            # Get the cluster center (mean) for this cluster from GMM
            cluster_center = gmm.means_[cluster_label]
            
            # Compute the Euclidean distance between each sample and the cluster center
            distances = np.linalg.norm(cluster_samples - cluster_center, axis=1)
            
            # Find the index of the sample closest to the cluster center
            closest_idx = cluster_indices[np.argmin(distances)]
            
            # Add this sample to the diverse samples list
            diverse_samples.append(sorted_indices[N_UNCERTAINTY + closest_idx])

        final_selection = np.concatenate([top, np.array(diverse_samples)])

        return final_selection[:n]  # Return the final n selections
