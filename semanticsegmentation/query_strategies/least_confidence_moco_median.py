import numpy as np
from scipy.spatial.distance import cdist
from .strategy import Strategy


class LeastConfidenceMocoMedian(Strategy):
    def __init__(self, dataset, net):
        super(LeastConfidenceMocoMedian, self).__init__(dataset, net)
        self.embeddings = self.get_embedding()

    def get_embedding(self):
        embedding = np.load(f'/work/gg0877/g260217/al_paper/eurosat_s2_al/moco_dw/512_64_NTXentLossWithIndices_42_100/42_embeddings.npy')
        return embedding # <- this has shape Nx128x128
    
    def query(self, n, cfg):
        unlabeled_idxs, unlabeled_data = self.dataset.get_unlabeled_data()
        probs = self.predict_prob(unlabeled_data)
        uncertainties = probs.max(1)[0].view(probs.shape[0], -1).median(dim=1)[1]
        candidate_indices = unlabeled_idxs[uncertainties.sort()[1][:n * 5]]

        top_embeddings = self.embeddings[candidate_indices]
        pairwise_distances = cdist(top_embeddings, top_embeddings, metric='euclidean')

        selected_idxs = [0]
        for _ in range(n - 1):
            # Find the farthest point from the already selected points
            farthest_idx = np.argmax(np.min(pairwise_distances[selected_idxs], axis=0))
            selected_idxs.append(farthest_idx)

        return candidate_indices[selected_idxs]