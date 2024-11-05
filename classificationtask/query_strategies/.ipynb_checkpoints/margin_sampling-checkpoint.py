import numpy as np
from .strategy import Strategy

class MarginSampling(Strategy):
    def __init__(self, dataset, net, cfg):
        super(MarginSampling, self).__init__(dataset, net, cfg)

    def query(self, n):
        unlabeled_idxs, unlabeled_data = self.dataset.get_unlabeled_data()
        probs = self.predict_prob(unlabeled_data)
        print(probs.shape)
        probs_sorted, idxs = probs.sort(descending=True)
        uncertainties = probs_sorted[:, 0] - probs_sorted[:,1]
        return unlabeled_idxs[uncertainties.sort()[1][:n]]
