import numpy as np
from .strategy import Strategy

class LeastConfidenceMedian(Strategy):
    def __init__(self, dataset, net):
        super(LeastConfidenceMedian, self).__init__(dataset, net)

    def query(self, n, cfg):
        unlabeled_idxs, unlabeled_data = self.dataset.get_unlabeled_data()
        probs = self.predict_prob(unlabeled_data)
        uncertainties = probs.max(1)[0].view(probs.shape[0], -1).median(dim=1)[1] # <-- from shape Nx11x512x512 to Nx1
        return unlabeled_idxs[uncertainties.sort()[1][:n]]
