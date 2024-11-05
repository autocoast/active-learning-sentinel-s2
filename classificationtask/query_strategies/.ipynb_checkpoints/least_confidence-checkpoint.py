import numpy as np
from .strategy import Strategy

class LeastConfidence(Strategy):
    def __init__(self, dataset, net, cfg):
        super(LeastConfidence, self).__init__(dataset, net, cfg)

    def query(self, n):
        unlabeled_idxs, unlabeled_data = self.dataset.get_unlabeled_data()
        probs = self.predict_prob(unlabeled_data) # N x 10
        uncertainties = probs.max(1)[0]
        return unlabeled_idxs[uncertainties.sort()[1][:n]]
