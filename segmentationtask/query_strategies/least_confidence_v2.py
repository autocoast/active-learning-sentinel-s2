import numpy as np
from .strategy import Strategy

class LeastConfidenceV2(Strategy):
    def __init__(self, dataset, net):
        super(LeastConfidenceV2, self).__init__(dataset, net)

    def query(self, n, cfg):
        unlabeled_idxs, unlabeled_data = self.dataset.get_unlabeled_data()
        probs = self.predict_prob(unlabeled_data)
        probs_mean = probs.mean(dim=(2, 3))
        probs_std = probs.std(dim=(2, 3))
        uncertainties = (probs_mean.mean(dim=1) + probs_std.mean(dim=1)) / 2 
        return unlabeled_idxs[uncertainties.sort()[1][:n]]
