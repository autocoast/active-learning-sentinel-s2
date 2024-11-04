import numpy as np
from scipy.spatial.distance import cdist
from .strategy import Strategy
from sklearn.neighbors import NearestNeighbors
import torch


class LeastConfidenceMocoMargin(Strategy):
    def __init__(self, dataset, net):
        super(LeastConfidenceMocoMargin, self).__init__(dataset, net)
        self.embeddings = self.get_embedding()
        self.ran_once = False

    def get_embedding(self):
        embedding = np.load(f'/work/gg0877/g260217/al_paper/eurosat_s2_al/moco_dw/512_64_NTXentLossWithIndices_42_100/42_embeddings.npy')
        return embedding # <- this has shape Nx128x128
    
    def query(self, n, cfg):
        unlabeled_idxs, unlabeled_data = self.dataset.get_unlabeled_data()
        
        probs = self.predict_prob(unlabeled_data) # Shape Nx11x128x128

        predominant = torch.argmax(probs, dim=1) # Nx128x128
        predominant_classes = []
        
        second_classes = []
        second_dominant_classes = []
        
        predicted_classes = torch.argmax(probs, dim=1).view(probs.shape[0], -1)
        
        for i in range(predicted_classes.shape[0]):  # Iterate over each image
            unique, counts = torch.unique(predicted_classes[i], return_counts=True)

            #top2 = torch.topk(counts, 2).indices
            
            # Step 3: Find the class with the maximum count
            predominant_class = unique[torch.argmax(counts)]
            predominant_classes.append(predominant_class.item())  # Store predominant class for each image

            #if len(counts) > 1:
            #    second_class = unique[top2[1]]
            #    second_classes.append(second_class.item())  # Store predominant class for each image
            #else:
            #    second_classes.append(predominant_class.item())
        
        predominant_classes = np.array(predominant_classes)
        #second_dominant_classes = np.array(second_dominant_classes)

        print(np.unique(predominant_classes, return_counts=True))
        print(np.unique(second_dominant_classes, return_counts=True))
        
        uncertainties = probs.max(1)[0].view(probs.shape[0], -1).median(dim=1)[1]
        
        candidate_indices = unlabeled_idxs[uncertainties.sort()[1]] # problem hier 2470 moeglich
        top_32 = candidate_indices[:32]
        
        knn = NearestNeighbors(n_neighbors=10).fit(self.embeddings[candidate_indices])
        neighbors = knn.kneighbors(self.embeddings[candidate_indices], return_distance=False)
        
        border_samples = []

        if True: #self.ran_once == False:
            for idx in range(32, len(candidate_indices), 1): #candidate_indices[32:]:
                
                # Get 10 nearest neighbors and their labels
                neighbor_idx = neighbors[idx]
                #neighbor_idx = unlabeled_idxs[neighbor_idx]
                neighbor_labels = predominant_classes[neighbor_idx] #self.labels[neighbors[idx]]
                #neighbor_2nd_labels = second_dominant_classes[neighbor_idx]
    
                # Check if at least 30% of the neighbors are from a different class
                current_label = predominant_classes[idx]
                #current_2nd_label = second_dominant_classes[idx]
                
                different_class_count = np.sum(neighbor_labels != current_label)
                #different_2nd_class_count = np.sum(neighbor_2nd_labels != current_2nd_label)
                if different_class_count / 10 >= 0.5: # and different_2nd_class_count / 10 >= 0.3:
                    border_samples.append(idx)

        border_selection = []
        if len(border_samples) > 0:
            border_selection = candidate_indices[border_samples[:32]]
        
        if len(border_selection) < 32:
            remaining = 32 - len(border_selection)
            for i in range(32, len(candidate_indices), 1):
                if candidate_indices[i] not in border_selection:
                    border_selection.append(candidate_indices[i])
                remaining -= 1
                if remaining == 0:
                    break

        final_selection = np.concatenate([top_32, border_selection])

        self.ran_once = True

        return final_selection