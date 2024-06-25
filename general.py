#!/usr/bin/env python
# coding: utf-8

# In[5]:


import argparse
import builtins
import collections
import gc
import math
import os
import pdb
import pickle
import random
import time
import uuid
from sklearn.cluster import KMeans
import cv2
import numpy as np
import rasterio
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms.functional as TF
from PIL import Image
from pathlib import Path
from scipy.stats import entropy
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from torch.optim.lr_scheduler import ReduceLROnPlateau, _LRScheduler
from torch.utils.data import Dataset, DataLoader, Subset
from torch.utils.tensorboard import SummaryWriter
from sklearn.manifold import TSNE
import hashlib

# Local imports
from ca_transforms import RandomRotation, RandomHorizontalFlip, RandomVerticalFlip, RandomBrightnessContrast, RandomGammaCorrection, GaussianNoise, RandomMirrorSplit
from cvtorchvision import cvtransforms

collections.Iterable = collections.abc.Iterable

import argparse

parser = argparse.ArgumentParser()

# Step 2: Add arguments
parser.add_argument('--random', action='store_true', help='Run random model')
parser.add_argument('--mcfps', action='store_true', help='Run MCFPS model')
parser.add_argument('--mcfpsv2', action='store_true', help='Run MCFPS v2 model')
parser.add_argument('--mcfpsv3', action='store_true', help='Run MCFPS v23model')
parser.add_argument('--fps', action='store_true', help='Run FPS model')
parser.add_argument('--cluster-osal', action='store_true', help='Run one shot active learning')
parser.add_argument('--cluster-fps', action='store_true', help='Run cluster + FPS model')
parser.add_argument('--cluster-random', action='store_true', help='Run cluster + random model')
parser.add_argument('--cluster-mcfps', action='store_true', help='Run cluster + MCFPS model')
parser.add_argument('--balanced', action='store_true', help='Balanced mode')
parser.add_argument('--candidates-per-round', type=int, default=64, help='Candidates per round')
parser.add_argument('--criterion', type=int, default=90, help='Stopping criterion')
parser.add_argument('--neighbour-size', type=int, default=15, help='Size of the neighbourhood')
parser.add_argument('--prefix', type=str, default='NO_PREFIX', help='Prefix for the log files')
parser.add_argument('--random-state', type=int, default=42)

args, unknown = parser.parse_known_args()

args.cluster_osal = True
args.balanced = True
args.prefix = "2048_cluster_osal"
args.random_state = 58
args.neighbour_size = 5
args.candidates_per_round = 64


# In[6]:


BANDS = list(['B01', 'B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08', 'B09', 'B10', 'B11', 'B12', 'B8A'])


# In[7]:


def is_valid_file(filename):
    return filename.lower().endswith('.tif')

class EurosatDataset(Dataset):

    def __init__(self, root):
        self.root = Path(root)
        
        self.classes = sorted([d.name for d in self.root.iterdir() if d.is_dir()])
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}

        self.samples = []
        self.targets = []

        printed = False
        for froot, _, fnames in sorted(os.walk(root, followlinks=True)):
            for fname in sorted(fnames):
                if is_valid_file(fname):
                    path = os.path.join(froot, fname)
                    if '.ipynb_checkpoints' in path:
                        continue
                    self.samples.append(path)
                    target = self.class_to_idx[Path(path).parts[-2]]
                    if target == 0 and printed == False:
                        print(fname)
                        printed=True
                    self.targets.append(target)

    def __getitem__(self, index):
        path = self.samples[index]
        target = self.targets[index]
        
        with rasterio.open(path) as f:
            array = f.read().astype(np.int16)
                            
            img = array.transpose(1, 2, 0)

        channels = []
        
        for i,b in enumerate(BANDS):
            ch = img[:,:,i]
            ch = (ch / 10000.0 * 255.0).astype('uint8')

            if b=='B8A': 
                channels.insert(8,ch)
            else:
                channels.append(ch)
        img = np.dstack(channels)

        return img, target

    def __len__(self):
        return len(self.samples)


class Subset(Dataset):
    def __init__(self, dataset, indices, transform=None):
        self.dataset = dataset
        self.indices = indices
        self.transform = transform

    def __getitem__(self, idx):
        im, target = self.dataset[self.indices[idx]]
        
        if self.transform:
            im = self.transform(im)
        return im, target

    def __len__(self):
        return len(self.indices)


# In[8]:


EurosatDataset('/home/g/xxxxxxx/mc-dropout/paper_eurosat/EuroSAT_MS')


# In[9]:


def farthest_point_sampling(data, num_samples, initial_indices=None, ignore_indices=None):
    """
    Perform Farthest Point Sampling on a dataset, starting with pre-selected samples and ignoring specified points.
    
    :param data: A NumPy array of shape (n_points, n_features)
    :param num_samples: The total number of samples to select, including initial samples
    :param initial_indices: Indices of the initial pre-selected samples
    :param ignore_indices: Indices of the points to ignore during sampling
    :return: Indices of the sampled points
    """
    n_points = data.shape[0]
    if initial_indices is None:
        initial_indices = []
    if ignore_indices is None:
        ignore_indices = []

    

    # Find the center point if initial_indices is empty
    if len(initial_indices) == 0:
        counter = 0
        while True:
            random.seed(args.random_state + counter)
            candidate = random.choice(range(len(data)))
            if candidate not in ignore_indices:
                initial_indices = [candidate]
                break
            counter += 1
    #    center_point = np.mean(data, axis=0)
    #    dist_to_center = np.sum((data - center_point) ** 2, axis=1)
    #    initial_indices = [np.argmin(dist_to_center)]
    
    # Ensure the initial indices do not include ignored points
    initial_indices = [idx for idx in initial_indices if idx not in ignore_indices]

    # Initialize distances with infinity
    distances = np.full(n_points, np.inf)

    # Set distances for ignored points to negative infinity
    distances[ignore_indices] = -np.inf

    # Update distances based on initial samples
    for idx in initial_indices:
        if idx in ignore_indices:
            continue
        dist_to_point = np.sum((data - data[idx]) ** 2, axis=1)
        distances = np.minimum(distances, dist_to_point)

    sampled_indices = initial_indices.copy()

    while len(sampled_indices) < num_samples:
        # Select the farthest point
        farthest_point_idx = np.argmax(distances)
        sampled_indices.append(farthest_point_idx)

        # Update distances based on the newly added point
        new_point = data[farthest_point_idx]
        dist_to_new_point = np.sum((data - new_point) ** 2, axis=1)
        distances = np.minimum(distances, dist_to_new_point)

        # Ensure ignored points remain ignored
        distances[ignore_indices] = -np.inf

    return sampled_indices


# In[10]:


class MCDropout(nn.Module):
    def __init__(self, p=0, uncertainty_p=0.5):
        super(MCDropout, self).__init__()
        self.active = True
        self.p = p
        self.uncertainty_p = uncertainty_p
        self.uncertainty_mode = False      
        
    def forward(self, x):
        if self.active:
            if self.uncertainty_mode == True:                 
                return nn.functional.dropout(x, self.uncertainty_p, True, False)
            else:                                     
                return nn.functional.dropout(x, self.p, True, False)                              
        return x
    
    def enable_uncertainty_mode(self):
        self.uncertainty_mode = True
    
    def disable_uncertainty_mode(self):
        self.uncertainty_mode = False
    
    def enable(self):
        self.active = True
        
    def disable(self):
        self.active = False        
    
def adjust_learning_rate(optimizer, lr, epoch, epochs):
    lr *= 0.5 * (1. + math.cos(math.pi * epoch / epochs))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr    
        
        
def fix_random_seeds(seed=42):
    """
    Fix random seeds.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

class EurosatExperiment:
    
    def __init__(self, main_batch_size=args.candidates_per_round, mcfps_batch_size=args.candidates_per_round, neighborhood_size=args.neighbour_size):
        
        self.main_batch_size = 64 #main_batch_size
        self.mcfps_batch_size = 64 #mcfps_batch_size        
        self.neighborhood_size = neighborhood_size
        self.mcfps_dataloader_workers = 64 #args.candidates_per_round
        self.main_dataloader_workers = 64 #args.candidates_per_round,
        self.mcfps_epochs = 100
        self.main_model_epochs = 100
        self.n_candidates_per_round = 32
        self.experiment_id = [] #"unbalanced_32_mcfps_2048_" + str(uuid.uuid4())
        
        self.experiment_id.append(args.prefix)
        
        if args.balanced:
            self.experiment_id.append("balanced")
        else:
            self.experiment_id.append("unbalanced")
            
        self.experiment_id.append(str(args.candidates_per_round))
        
        if args.mcfps:
            self.experiment_id.append("mcfps")
            
        if args.fps:
            self.experiment_id.append("fps")            
        
        if args.cluster_fps:
            self.experiment_id.append("clusterfps")
            
        if args.cluster_random:
            self.experiment_id.append("clusterrandom")            
        
        if args.cluster_mcfps:
            self.experiment_id.append("clustermcfps")
            
        if args.random:
            self.experiment_id.append("random")
            
        if args.mcfpsv2:
            self.experiment_id.append("mcfpsv2")            

        if args.mcfpsv3:
            self.experiment_id.append("mcfpsv3")                    

        if args.cluster_osal:
            self.experiment_id.append("clusterosal")        
            
        self.experiment_id.append(str(args.random_state))
            
        self.experiment_id.append(str(uuid.uuid4()))
        
        self.experiment_id = "_".join(self.experiment_id)
        
        
        self.transforms = cvtransforms.Compose([
            cvtransforms.RandomResizedCrop(224),
            #cvtransforms.Resize(args.in_size),
            cvtransforms.RandomHorizontalFlip(),
            cvtransforms.ToTensor()
            ])

        self.val_transforms = cvtransforms.Compose([
                cvtransforms.Resize(256),
                cvtransforms.CenterCrop(224),
                cvtransforms.ToTensor()
            ])
        
        self.eurosat_dataset = EurosatDataset('/home/g/xxxxxxx/mc-dropout/paper_eurosat/EuroSAT_MS')
        indices = np.arange(len(self.eurosat_dataset))
        self.original_indices = indices
        train_indices, test_indices = train_test_split(indices, train_size=0.8, stratify=self.eurosat_dataset.targets, random_state=42)
        
        
        #embeddings = np.load('/work/gg0877/xxxxxxx/al_paper/eurosat_ssl/embeddings_pca_50.npy')
        embeddings = np.load('/work/gg0877/xxxxxxx/al_paper/eurosat_ssl/embeddings_2048.npy')
        #embeddings = np.load('/work/gg0877/xxxxxxx/al_paper/eurosat_ssl/tsne_2.npy')
        # embeddings = np.load('/work/gg0877/xxxxxxx/al_paper/eurosat_ssl/embeddings_pca_512.npy')
        labels = np.load('/work/gg0877/xxxxxxx/al_paper/eurosat_ssl/labels_2048.npy')
        
        self.whole_embeddings = embeddings
        self.whole_labels = labels
        
        self.eurosat_embeddings = embeddings[train_indices]
        self.embedding_labels = labels[train_indices]        
        
        self.ignore_indices = []
        
        self.skippables = []
        
        if args.balanced:
            print('going for balanced')
            pass
        #    self.trainix_to_fullix = {k:v for k, v in enumerate(train_indices)}
        #    self.fullix_to_trainix = {v:k for k, v in enumerate(train_indices)}
        
        
            
        else:
            print('unbalanced mode')
            unbalanced_indices = []
            #self.trainix_to_fullix = {}
            
            for i in range(10):
                random.seed(args.random_state + i)
                len_i = len(np.where(self.embedding_labels == i)[0])
                # these will be the indices of embedding_labels (0-21000)
                # number of indices to ignore
                N = random.randint(0, int(len_i * 0.8))
                ixs = np.where(self.embedding_labels == i)[0][:N]
                self.ignore_indices += ixs.tolist()
#                unbalanced_indices += ixs.tolist()
                
#            for i, ix in enumerate(unbalanced_indices):
#                self.trainix_to_fullix[i] = train_indices[ix]

            #self.eurosat_embeddings = self.eurosat_embeddings[unbalanced_indices]
            #self.embedding_labels = self.embedding_labels[unbalanced_indices]       
            
#        print(unbalanced_indices)
        
        #print(np.unique(self.embedding_labels, return_counts=True))

        
        #self.trainix_to_fullix = {k:v for k, v in enumerate(train_indices)}
        #self.fullix_to_trainix = {v:k for k, v in enumerate(train_indices)}

        self.train_indices = train_indices
        
        self.train_dataset = Subset(self.eurosat_dataset, train_indices, transform=self.transforms)
        self.test_dataset = Subset(self.eurosat_dataset, test_indices, transform=self.val_transforms)

        self.train_loader = DataLoader(self.train_dataset, batch_size=64, num_workers=64, drop_last=False, shuffle=False)
        self.test_loader = DataLoader(self.test_dataset, batch_size=64, num_workers=64, drop_last=False)
        
        clusterer = KMeans(n_clusters=10, init='k-means++', random_state=42)
        self.cluster_labels = clusterer.fit_predict(self.eurosat_embeddings)
        
        
        self.cluster_samplesize = self.get_n_candidate_per_cluster()
        
        print('len test set', len(self.test_dataset))
        #fix_random_seeds()
        
    def get_n_candidate_per_cluster(self):
        import heapq

        heap = []
        N = len(self.eurosat_embeddings)
        M = args.candidates_per_round
        s_ = 0
        cluster_N = {}

        for  i in range(10):
            c = np.where(self.cluster_labels == i)[0].shape[0]
            n_candidates_cluster_i = math.ceil(M / 100 * (c / N * 100))
            cluster_N[i] = n_candidates_cluster_i
            s_ += n_candidates_cluster_i
            heapq.heappush(heap, (-n_candidates_cluster_i, i))

        while s_ != M:
            minus_n_candidate, cluster = heapq.heappop(heap)
            cluster_N[cluster] = -minus_n_candidate - 1
            s_ = 0
            for val in cluster_N.values():
                s_ += val

        return cluster_N        
        
    def write_log(self, string: str):
        with open(str(self.experiment_id), "a") as myfile:
            myfile.write(f"{string}\n")          
        
        
    def setup_loaders(self, indices):
        #converted_ix = []
        #for ix in indices:
        #    converted_ix.append(self.trainix_to_fullix[ix])
            
        subset_train_dataset = Subset(self.eurosat_dataset, indices, transform=self.transforms)
        subset_train_loader = DataLoader(
            subset_train_dataset, 
            batch_size=self.mcfps_batch_size, 
            num_workers=self.mcfps_dataloader_workers, 
            drop_last=True
        )
        return subset_train_loader           
    
    def predict_with_uncertainty(self, model, input_tensor, n_iter=10):
        input_tensor = input_tensor.to('cuda')
        model.eval()
        model.mc_dropout.enable()
        model.mc_dropout.enable_uncertainty_mode()

        outputs = []

        for _ in range(n_iter):
            with torch.no_grad():
                output = model(input_tensor)
            outputs.append(output)

        outputs = torch.stack(outputs) # n_iter x NBRSIZE x classes
        
        mean_outputs = outputs.mean(dim=0)
        mean_outputs = F.softmax(mean_outputs, dim=1) # NBRSIZE x classes

        max_outputs, _ = torch.max(mean_outputs, dim=1)
        


        return max_outputs.squeeze().detach().cpu().numpy()
        #return mean_outputs.squeeze().detach().cpu().numpy()
    
    def free_model(self, model):
        del model
        torch.cuda.empty_cache()
        gc.collect()    
        
    def random_cluster(self, already_chosen_indices):
        possible_indices = []
        new_indices = []
        for cluster_i in range(10):
            cluster_indices = np.where(self.cluster_labels == cluster_i)[0]
            possible_indices = []
            for cluster_ix in cluster_indices:
                if cluster_ix in already_chosen_indices:
                    continue
                possible_indices.append(cluster_ix)
            
            sample_size = self.cluster_samplesize[cluster_i]
            
            ixx = []
            while len(ixx) < sample_size:
                next_int = random.choice(possible_indices)
                if next_int in self.ignore_indices:
                    continue
                ixx.append(next_int)
            new_indices += ixx

        return new_indices
    
    
    def fps_cluster(self, already_chosen_indices):
        possible_indices = []
        new_indices = []
        for cluster_i in range(10):
            past_indices = []
            cluster_indices = np.where(self.cluster_labels == cluster_i)[0]
            possible_indices = []
            for cluster_ix in cluster_indices:
                if cluster_ix in already_chosen_indices:
                    past_indices.append(cluster_ix)
                    continue
                possible_indices.append(cluster_ix)
                
            sample_size = self.cluster_samplesize[cluster_i]
            
            ignore_indices_in_fps = np.array(np.where(self.cluster_labels != cluster_i)[0].tolist() + self.ignore_indices)
            
            ixx = farthest_point_sampling(self.eurosat_embeddings, sample_size + len(past_indices), initial_indices=past_indices, ignore_indices=ignore_indices_in_fps)
            
            for ix in ixx:
                if ix not in past_indices:
                    new_indices.append(ix)

        return new_indices
    
    def mcfps_cluster(self, mc_model, nbrs, already_chosen_indices, blacklist=[]):
        possible_indices = []
        sampled_emb_indices = []
        for cluster_i in range(10):
            past_indices = []
            cluster_indices = np.where(self.cluster_labels == cluster_i)[0]
            possible_indices = []
            for cluster_ix in cluster_indices:
                if cluster_ix in already_chosen_indices:
                    past_indices.append(cluster_ix)
                    continue
                possible_indices.append(cluster_ix)
                
            sample_size = self.cluster_samplesize[cluster_i]
            
            ignore_indices_in_fps = np.array(np.where(self.cluster_labels != cluster_i)[0].tolist() + self.ignore_indices)
            
            ixx = farthest_point_sampling(self.eurosat_embeddings, sample_size + len(past_indices), initial_indices=past_indices, ignore_indices=ignore_indices_in_fps)
            
            for ix in ixx:
                if ix not in past_indices:
                    sampled_emb_indices.append(ix)
                    
        print(len(sampled_emb_indices))

        return_indices = []
        
        print('new selection round', len(sampled_emb_indices))
        
        
        log_labels = []
        log_entropies = []
        log_sample_ix = []
        
        for ii, sample_i in enumerate(sampled_emb_indices):  
            #ok = False
                                
            _, neighbor_indices = nbrs.kneighbors([self.eurosat_embeddings[sample_i]])
            neighbor_indices = neighbor_indices.flatten()

            #neighborhood_real_data = torch.zeros((len(neighbor_indices), 13, 64, 64))
            neighborhood_real_data = torch.zeros((len(neighbor_indices), 13, 224, 224))            
            for i in range(len(neighborhood_real_data)):
                neighborhood_real_data[i] = self.transforms(self.eurosat_dataset[neighbor_indices[i]][0])
            neighborhood_real_data = neighborhood_real_data.float()

            # mean_predictions.shape = 30x10
            # mean_predictions = self.predict_with_uncertainty(mc_model, neighborhood_real_data)

            # entropies = entropy(mean_predictions, base=2, axis=1)

            #sorted_indices = np.argsort(entropies).tolist() # from low ent to high ent         
            
            # will return a vector of size N
            mean_accuracies = self.predict_with_uncertainty(mc_model, neighborhood_real_data)
            sorted_indices = np.argsort(mean_accuracies).tolist() # sort is asc, higher accuracies at the end of the list
            

            # choose those with lowest accuracy
            for candidate_i in range(len(sorted_indices)):
                #percentile_10 = math.ceil(args.neighbour_size/100*10)
                percentile_10 = 0 #math.ceil(args.candidates_per_round/100*30)
                candidate = neighbor_indices[sorted_indices][candidate_i + percentile_10]
                if candidate in already_chosen_indices:
                    continue
                    
                log_labels.append(self.whole_labels[sample_i])
                log_entropies.append(mean_accuracies[sorted_indices][candidate_i + percentile_10])    
                log_sample_ix.append(sample_i)
                return_indices.append(candidate)                        
                break
                
        self.write_log("SAMPLEIXSTART")                
        self.write_log(log_sample_ix)
        self.write_log("SAMPLEINXEND")        
        self.write_log("ENTROPYSTART")
        self.write_log(log_entropies)
        self.write_log("ENTROPYEND")    
        self.write_log("LABELSTART")            
        self.write_log(log_labels)  
        self.write_log("LABELEND")    
            
        if len(return_indices) != args.candidates_per_round:
            raise Exception('return indices not', args.candidates_per_round, 'size is:', len(return_indices))
        
        return return_indices

    def fps_get_next_indices(self, already_chosen_indices, sample_size):
 
        sampled_emb_indices = farthest_point_sampling(self.whole_embeddings, sample_size + len(already_chosen_indices), initial_indices=already_chosen_indices, ignore_indices=self.ignore_indices)
        filtered_sampled_indices = []
        for ix in sampled_emb_indices:
            if ix in already_chosen_indices:
                continue
            filtered_sampled_indices.append(ix)
        sampled_emb_indices = filtered_sampled_indices
    
        
        return sampled_emb_indices


    def cluster_osal_get_next_indices(self, iteration):

        N_clusters = 2
        
        candidates = []
        # cluster k++ 
        # we found out by applying the silhouette score from 2 to 15 including both
        clusterer = KMeans(n_clusters=N_clusters, init='k-means++', random_state=42)
        cluster_labels = clusterer.fit_predict(self.eurosat_embeddings)    

        cluster_labels[self.ignore_indices] = N_clusters # cluster doesnt exist yet, but we create one and put the indices to ignore there

        for i in range(N_clusters):
            print('Cluster', i)
            print(np.unique(self.embedding_labels[np.where(cluster_labels==i)[0]], return_counts=True))
            frayed_cluster_ix = np.where(cluster_labels==i)[0]
            sequential_cluster_to_frayed = {k: frayed_cluster_ix[k] for k in range(len(frayed_cluster_ix)) }
            next_candidates = farthest_point_sampling(self.eurosat_embeddings[np.where(cluster_labels==i)[0]], round(64 * (iteration + 1) / N_clusters), initial_indices=None, ignore_indices=None)
    
            for candidate in next_candidates:
                if len(candidates) >= (64 * (iteration + 1)):
                    break
                candidates.append(sequential_cluster_to_frayed[candidate])    
            '''
            frayed_cluster1_ix = np.where(cluster_labels==1)[0]
            sequential_cluster1_to_frayed = {k: frayed_cluster1_ix[k] for k in range(len(frayed_cluster1_ix)) }
            next_candidates = farthest_point_sampling(self.eurosat_embeddings[np.where(cluster_labels==1)[0]], 64 * (iteration + 1) / 2, initial_indices=None, ignore_indices=None)
    
            for candidate in next_candidates:
                candidates.append(sequential_cluster1_to_frayed[candidate])    
            '''


        return candidates
    
    
    def random_get_next_indices(self, mc_model, nbrs, already_chosen_indices, sample_size, blacklist=[]):
        new_indices = []
        while len(new_indices) < sample_size:
            next_int = random.randint(0, len(self.original_indices) - 1)
            if next_int in already_chosen_indices or next_int in self.ignore_indices:
                continue
            new_indices.append(next_int)
            
        return new_indices
            
    def setup_model(self, dropout_prob=0.3):
        net = torchvision.models.resnet50(pretrained=False)
        net.fc = torch.nn.Linear(2048,10)

        net.conv1 = torch.nn.Conv2d(13, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)

        for name, param in net.named_parameters():
            if name not in ['fc.weight','fc.bias']:
                param.requires_grad = False
        
        if os.path.isfile('/home/g/xxxxxxx/mc-dropout/paper_eurosat/B13_rn50_moco_0099.pth'):
            print("=> loading checkpoint '{}'".format('B13_rn50_moco_0099.pth'))
            checkpoint = torch.load('/home/g/xxxxxxx/mc-dropout/paper_eurosat/B13_rn50_moco_0099.pth', map_location="cpu")

            # rename moco pre-trained keys
            state_dict = checkpoint['state_dict']

            for k in list(state_dict.keys()):
                # retain only encoder up to before the embedding layer
                if k.startswith('module.encoder_q') and not k.startswith('module.encoder_q.fc'):
                    #pdb.set_trace()
                    # remove prefix
                    state_dict[k[len("module.encoder_q."):]] = state_dict[k]
                # delete renamed or unused k
                del state_dict[k]
            '''
            # remove prefix
            state_dict = {k.replace("module.", ""): v for k,v in state_dict.items()}
            '''
            #args.start_epoch = 0
            msg = net.load_state_dict(state_dict, strict=False)
            #pdb.set_trace()
            assert set(msg.missing_keys) == {"fc.weight", "fc.bias"}

            print("=> loaded pre-trained model '{}'".format('B13_rn50_moco_0099.pth'))
        else:
            print("=> no checkpoint found at '{}'".format('B13_rn50_moco_0099.pth'))
        
        mc_dropout = MCDropout()
        net.fc = nn.Sequential(
            mc_dropout,
            torch.nn.Linear(2048, 10)
        )
        
        net.mc_dropout = mc_dropout
        
        net.fc[1].weight.data.normal_(mean=0.0,std=0.01)
        net.fc[1].bias.data.zero_()
        
        '''
        layer1 = net.layer1
        layer2 = net.layer2
        layer3 = net.layer3
        layer4 = net.layer4

        # Adding dropout after each block
        for layer in [layer4]:
            for i in range(len(layer)):
                layer[i] = nn.Sequential(layer[i], MCDropout(0.1))    
        '''

        #if torch.cuda.device_count() > 1:
        #    net = torch.nn.DataParallel(net)      

        net.cuda()
        return net

    def mcfpsv3_get_next_indices(self, mc_model, nbrs, already_chosen_indices, sample_size, blacklist=[]):
            
        #return [17906, 2457, 20133, 16487, 19340, 16821, 3838, 10873, 1387, 5202, 18206, 18569, 284, 12074, 5821, 8567, 10895, 7706, 5630, 14289, 8608, 11198, 7970, 2434, 3796, 342, 14220, 17949, 4698, 14447, 7203, 17237, 4376, 2438, 19950, 6297, 16736, 21302, 11333, 5883, 9171, 10110, 17691, 18369, 6872, 19286, 12518, 13328, 8153, 316, 7531, 3521, 5336, 12905, 20609, 9409, 15131, 64, 6490, 6755, 4659, 19221, 7541, 20706, 16941, 5463, 9160, 10473, 18711, 3029, 16807, 565, 21273, 19784, 14301, 18874, 8485, 16015, 5422, 11571, 16909, 16880, 3526, 7979, 699, 1814, 19380, 2739, 16034, 10192, 14984, 13045, 11284, 15496, 2986, 1416, 7873, 6514, 19160, 10161, 24, 6174, 5448, 13894, 5108, 7582, 209, 8903, 19361, 6243, 2280, 20790, 21029, 6333, 4130, 9762, 1722, 11537, 14561, 13313, 7495, 6421, 9539, 4463, 4111, 7503, 14138, 6250, 6357, 4425, 19295, 12727, 5291, 18001, 12565, 18153, 20914, 6771, 7835, 1166, 21195, 17817, 11801, 4140, 3114, 18731, 4898, 5378, 6240, 10108, 10261, 12295, 6099, 10223, 9077, 15913, 12267, 15706, 3087, 10045, 11337, 14718, 3381, 10842, 18809, 15148, 15477, 7109, 21218, 18849, 12826, 20811, 9800, 5421, 16503, 17027, 10423, 751, 12323, 78, 2311, 5901, 6397, 9072, 16180, 6318, 13132, 3525, 13582, 18700, 10238, 17645, 10167, 21542, 11063, 2983, 11837, 8480, 18919, 11944, 12743, 5095, 13250, 19945, 11771, 10083, 3785, 20643, 3324, 16616, 5518, 8728, 6330, 17886, 14166, 8642, 8984, 11713, 9571, 20603, 5169, 19565, 17603, 14707, 16275, 4814, 12189, 21201, 8541, 17994, 3551, 13803, 6146, 5149, 6474, 21598, 10690, 7930, 1933, 13963, 18893, 12038, 19485, 8296, 18669, 4988, 19350, 8224, 15006, 8840, 8262, 20730, 13821, 17341, 5146, 6399]
    
        """
        This is the most complicated part.
        It performs the following steps:
        - Perform farthest point sampling
        - For each new sampled point (already chosen indices from the past won't be taken into account),
          we search the neighborhood
        - We measure the certainty for each neighbor
        """
        
        # farthest_point_sampling(data, num_samples, initial_indices=None, ignore_indices=None):        
        
        indices_to_ignore = self.ignore_indices + blacklist
        
        sampled_emb_indices = farthest_point_sampling(self.whole_embeddings, sample_size + len(already_chosen_indices), initial_indices=already_chosen_indices, ignore_indices=indices_to_ignore)
        print(sampled_emb_indices)
        filtered_sampled_indices = []
        for ix in sampled_emb_indices:
            if ix in already_chosen_indices or ix in indices_to_ignore:
                continue
            filtered_sampled_indices.append(ix)
        sampled_emb_indices = filtered_sampled_indices
    
        return_indices = []
        
        print('new selection round', len(sampled_emb_indices))
        
        log_labels = []
        log_entropies = []
        log_sample_ix = []
        
        for ii, sample_i in enumerate(sampled_emb_indices):
            

            
            #ok = False
                                
            _, neighbor_indices = nbrs.kneighbors([self.whole_embeddings[sample_i]])
            neighbor_indices = neighbor_indices.flatten()

            #neighborhood_real_data = torch.zeros((len(neighbor_indices), 13, 64, 64))
            neighborhood_real_data = torch.zeros((len(neighbor_indices), 13, 224, 224))            
            for i in range(len(neighborhood_real_data)):
                neighborhood_real_data[i] = self.transforms(self.eurosat_dataset[neighbor_indices[i]][0])
            neighborhood_real_data = neighborhood_real_data.float()

            # mean_predictions.shape = 30x10
            # mean_predictions = self.predict_with_uncertainty(mc_model, neighborhood_real_data)

            # entropies = entropy(mean_predictions, base=2, axis=1)

            #sorted_indices = np.argsort(entropies).tolist() # from low ent to high ent         
            
            # will return a vector of size N
            mean_accuracies = self.predict_with_uncertainty(mc_model, neighborhood_real_data)
            sorted_indices = np.argsort(mean_accuracies).tolist() # sort is asc, higher accuracies at the end of the list
            

            # choose those with lowest accuracy
            for candidate_i in range(len(sorted_indices)):
                percentile_10 = math.ceil(args.neighbour_size/100*10)
                percentile_10 = 0
                candidate = neighbor_indices[sorted_indices][candidate_i + percentile_10]
                if candidate in already_chosen_indices:
                    continue
                    
                log_labels.append(self.whole_labels[sample_i])
                log_entropies.append(mean_accuracies[sorted_indices][candidate_i + percentile_10])    
                log_sample_ix.append(sample_i)
                return_indices.append(candidate)                        
                break
                
        self.write_log("SAMPLEIXSTART")                
        self.write_log(log_sample_ix)
        self.write_log("SAMPLEINXEND")        
        self.write_log("ENTROPYSTART")
        self.write_log(log_entropies)
        self.write_log("ENTROPYEND")    
        self.write_log("LABELSTART")            
        self.write_log(log_labels)  
        self.write_log("LABELEND")        
            
        if len(return_indices) != args.candidates_per_round:
            raise Exception('return indices not', args.candidates_per_round, 'size is:', len(return_indices))
        
        return return_indices   
     
    def mcfpsv2_get_next_indices(self, mc_model, nbrs, already_chosen_indices, sample_size, blacklist=[]):
        
        print("xx: ", len(already_chosen_indices))
        #return [17906, 2457, 20133, 16487, 19340, 16821, 3838, 10873, 1387, 5202, 18206, 18569, 284, 12074, 5821, 8567, 10895, 7706, 5630, 14289, 8608, 11198, 7970, 2434, 3796, 342, 14220, 17949, 4698, 14447, 7203, 17237, 4376, 2438, 19950, 6297, 16736, 21302, 11333, 5883, 9171, 10110, 17691, 18369, 6872, 19286, 12518, 13328, 8153, 316, 7531, 3521, 5336, 12905, 20609, 9409, 15131, 64, 6490, 6755, 4659, 19221, 7541, 20706, 16941, 5463, 9160, 10473, 18711, 3029, 16807, 565, 21273, 19784, 14301, 18874, 8485, 16015, 5422, 11571, 16909, 16880, 3526, 7979, 699, 1814, 19380, 2739, 16034, 10192, 14984, 13045, 11284, 15496, 2986, 1416, 7873, 6514, 19160, 10161, 24, 6174, 5448, 13894, 5108, 7582, 209, 8903, 19361, 6243, 2280, 20790, 21029, 6333, 4130, 9762, 1722, 11537, 14561, 13313, 7495, 6421, 9539, 4463, 4111, 7503, 14138, 6250, 6357, 4425, 19295, 12727, 5291, 18001, 12565, 18153, 20914, 6771, 7835, 1166, 21195, 17817, 11801, 4140, 3114, 18731, 4898, 5378, 6240, 10108, 10261, 12295, 6099, 10223, 9077, 15913, 12267, 15706, 3087, 10045, 11337, 14718, 3381, 10842, 18809, 15148, 15477, 7109, 21218, 18849, 12826, 20811, 9800, 5421, 16503, 17027, 10423, 751, 12323, 78, 2311, 5901, 6397, 9072, 16180, 6318, 13132, 3525, 13582, 18700, 10238, 17645, 10167, 21542, 11063, 2983, 11837, 8480, 18919, 11944, 12743, 5095, 13250, 19945, 11771, 10083, 3785, 20643, 3324, 16616, 5518, 8728, 6330, 17886, 14166, 8642, 8984, 11713, 9571, 20603, 5169, 19565, 17603, 14707, 16275, 4814, 12189, 21201, 8541, 17994, 3551, 13803, 6146, 5149, 6474, 21598, 10690, 7930, 1933, 13963, 18893, 12038, 19485, 8296, 18669, 4988, 19350, 8224, 15006, 8840, 8262, 20730, 13821, 17341, 5146, 6399]
    
        """
        This is the most complicated part.
        It performs the following steps:
        - Perform farthest point sampling
        - For each new sampled point (already chosen indices from the past won't be taken into account),
          we search the neighborhood
        - We measure the certainty for each neighbor
        """
        
        # farthest_point_sampling(data, num_samples, initial_indices=None, ignore_indices=None):        
        
        if len(already_chosen_indices) == 64:
            self.ignore_indices += np.where(self.embedding_labels == 9)[0].tolist()
        
        indices_to_ignore = self.ignore_indices + blacklist
        
        sampled_emb_indices = farthest_point_sampling(self.whole_embeddings, sample_size + len(already_chosen_indices) + len(self.skippables), initial_indices=(already_chosen_indices + self.skippables), ignore_indices=indices_to_ignore)
    
        filtered_sampled_indices = []
        for ix in sampled_emb_indices:
            if ix in already_chosen_indices and ix not in self.skippables:
                continue
            filtered_sampled_indices.append(ix)
        sampled_emb_indices = filtered_sampled_indices
    
        return_indices = []
        
        print('new selection round', len(sampled_emb_indices))
        
        log_labels = []
        log_entropies = []
        log_sample_ix = []
        
        for ii, sample_i in enumerate(sampled_emb_indices):
            
            ok = False
            
            while ok == False:
            
                print('selecting', (ii+1), '/', args.candidates_per_round)

                _, neighbor_indices = nbrs.kneighbors([self.whole_embeddings[sample_i]])
                neighbor_indices = neighbor_indices.flatten()

                #neighborhood_real_data = torch.zeros((len(neighbor_indices), 13, 64, 64))
                neighborhood_real_data = torch.zeros((len(neighbor_indices), 13, 224, 224))            
                for i in range(len(neighborhood_real_data)):
                    neighborhood_real_data[i] = self.transforms(self.eurosat_dataset[neighbor_indices[i]][0])
                neighborhood_real_data = neighborhood_real_data.float()

                # mean_predictions.shape = 30x10
                # mean_predictions = self.predict_with_uncertainty(mc_model, neighborhood_real_data)

                # entropies = entropy(mean_predictions, base=2, axis=1)

                #sorted_indices = np.argsort(entropies).tolist() # from low ent to high ent         

                # will return a vector of size N
                mean_accuracies = self.predict_with_uncertainty(mc_model, neighborhood_real_data, n_iter=3)
                sorted_indices = np.argsort(mean_accuracies).tolist() # sort is asc, higher accuracies at the end of the list

                #print('sorted indices', len(sorted_indices), 'nbrs', self.neighborhood_size)


                # choose those with lowest accuracy
                for candidate_i in range(len(sorted_indices)):
                    
                    if self.embedding_labels[neighbor_indices[sorted_indices][candidate_i + percentile_10]] == 9:
                        continue
                    
                    percentile_10 = math.ceil(args.neighbour_size/100*10)
                    percentile_10 = 0
                    candidate = neighbor_indices[sorted_indices][candidate_i + percentile_10]

                    if candidate in already_chosen_indices:
                        continue

                    if mean_accuracies[candidate_i + percentile_10] >= 0.8:
                        self.skippables.append(neighbor_indices[sorted_indices][candidate_i + percentile_10])
                        print('Certainty too high with accuracy', mean_accuracies[candidate_i + percentile_10], 'for label', self.whole_labels[neighbor_indices[sorted_indices][candidate_i + percentile_10]])
                        continue
                    else:
                        log_labels.append(self.whole_labels[sample_i])
                        log_entropies.append(mean_accuracies[sorted_indices][candidate_i + percentile_10])    
                        log_sample_ix.append(sample_i)
                        return_indices.append(candidate)         
                        ok = True
                        break
                if ok:
                    break
                else:
                    max_jumps = 5
                    while True:
                        max_jumps -= 1
                        
                        print('reducing jump', 'current budget:', max_jumps)

                        to_skip = already_chosen_indices + sampled_emb_indices + return_indices + self.skippables
                        backup_samples = farthest_point_sampling(self.whole_embeddings, 1 + len(to_skip), initial_indices=to_skip, ignore_indices=indices_to_ignore)

                        for backup_sample in backup_samples:
                            if backup_sample not in to_skip:
                                sample_i = backup_sample
                                #indices_to_ignore += [sample_i]
                                self.skippables.append(sample_i)
                                break
                        
                        torch_sample = torch.zeros((1, 13, 224, 224))
                        torch_sample[0] = self.transforms(self.eurosat_dataset[sample_i][0])
                        mean_accuracies = self.predict_with_uncertainty(mc_model, torch_sample, n_iter=3)
                        print(mean_accuracies)
                        if mean_accuracies < 0.8:
                            break
                        elif mean_accuracies >= 0.8 and max_jumps == 0:
                            print('High accuracy, but max jumps exceeded')
                            break
                
        self.write_log("SAMPLEIXSTART")                
        self.write_log(log_sample_ix)
        self.write_log("SAMPLEINXEND")        
        self.write_log("ENTROPYSTART")
        self.write_log(log_entropies)
        self.write_log("ENTROPYEND")    
        self.write_log("LABELSTART")            
        self.write_log(log_labels)  
        self.write_log("LABELEND")        
            
        if len(return_indices) > args.candidates_per_round:
            return_indices = return_indices[:args.candidates_per_round]            
        elif len(return_indices) < args.candidates_per_round:
            raise Exception('return indices not', args.candidates_per_round, 'size is:', len(return_indices))
        
        return return_indices    

    # mc_model, nearest_neighbors, chosen_indices, n_candidates_per_round, blacklist=blacklist
    def mcfps_get_next_indices(self, mc_model, nbrs, already_chosen_indices, sample_size, blacklist=[]):
        
        print("xx: ", len(already_chosen_indices))
        #return [17906, 2457, 20133, 16487, 19340, 16821, 3838, 10873, 1387, 5202, 18206, 18569, 284, 12074, 5821, 8567, 10895, 7706, 5630, 14289, 8608, 11198, 7970, 2434, 3796, 342, 14220, 17949, 4698, 14447, 7203, 17237, 4376, 2438, 19950, 6297, 16736, 21302, 11333, 5883, 9171, 10110, 17691, 18369, 6872, 19286, 12518, 13328, 8153, 316, 7531, 3521, 5336, 12905, 20609, 9409, 15131, 64, 6490, 6755, 4659, 19221, 7541, 20706, 16941, 5463, 9160, 10473, 18711, 3029, 16807, 565, 21273, 19784, 14301, 18874, 8485, 16015, 5422, 11571, 16909, 16880, 3526, 7979, 699, 1814, 19380, 2739, 16034, 10192, 14984, 13045, 11284, 15496, 2986, 1416, 7873, 6514, 19160, 10161, 24, 6174, 5448, 13894, 5108, 7582, 209, 8903, 19361, 6243, 2280, 20790, 21029, 6333, 4130, 9762, 1722, 11537, 14561, 13313, 7495, 6421, 9539, 4463, 4111, 7503, 14138, 6250, 6357, 4425, 19295, 12727, 5291, 18001, 12565, 18153, 20914, 6771, 7835, 1166, 21195, 17817, 11801, 4140, 3114, 18731, 4898, 5378, 6240, 10108, 10261, 12295, 6099, 10223, 9077, 15913, 12267, 15706, 3087, 10045, 11337, 14718, 3381, 10842, 18809, 15148, 15477, 7109, 21218, 18849, 12826, 20811, 9800, 5421, 16503, 17027, 10423, 751, 12323, 78, 2311, 5901, 6397, 9072, 16180, 6318, 13132, 3525, 13582, 18700, 10238, 17645, 10167, 21542, 11063, 2983, 11837, 8480, 18919, 11944, 12743, 5095, 13250, 19945, 11771, 10083, 3785, 20643, 3324, 16616, 5518, 8728, 6330, 17886, 14166, 8642, 8984, 11713, 9571, 20603, 5169, 19565, 17603, 14707, 16275, 4814, 12189, 21201, 8541, 17994, 3551, 13803, 6146, 5149, 6474, 21598, 10690, 7930, 1933, 13963, 18893, 12038, 19485, 8296, 18669, 4988, 19350, 8224, 15006, 8840, 8262, 20730, 13821, 17341, 5146, 6399]
    
        """
        This is the most complicated part.
        It performs the following steps:
        - Perform farthest point sampling
        - For each new sampled point (already chosen indices from the past won't be taken into account),
          we search the neighborhood
        - We measure the certainty for each neighbor
        """
        
        # farthest_point_sampling(data, num_samples, initial_indices=None, ignore_indices=None):        
        
        indices_to_ignore = self.ignore_indices + blacklist
        
        sampled_emb_indices = farthest_point_sampling(self.whole_embeddings, sample_size + len(already_chosen_indices), initial_indices=already_chosen_indices, ignore_indices=indices_to_ignore)
        print(sampled_emb_indices)
        filtered_sampled_indices = []
        for ix in sampled_emb_indices:
            if ix in already_chosen_indices:
                continue
            filtered_sampled_indices.append(ix)
        sampled_emb_indices = filtered_sampled_indices
    
        return_indices = []
        
        print('new selection round', len(sampled_emb_indices))
        
        log_labels = []
        log_entropies = []
        log_sample_ix = []
        
        for ii, sample_i in enumerate(sampled_emb_indices):
            

            
            #ok = False
                                
            _, neighbor_indices = nbrs.kneighbors([self.whole_embeddings[sample_i]])
            neighbor_indices = neighbor_indices.flatten()

            #neighborhood_real_data = torch.zeros((len(neighbor_indices), 13, 64, 64))
            neighborhood_real_data = torch.zeros((len(neighbor_indices), 13, 224, 224))            
            for i in range(len(neighborhood_real_data)):
                neighborhood_real_data[i] = self.transforms(self.eurosat_dataset[neighbor_indices[i]][0])
            neighborhood_real_data = neighborhood_real_data.float()

            # mean_predictions.shape = 30x10
            # mean_predictions = self.predict_with_uncertainty(mc_model, neighborhood_real_data)

            # entropies = entropy(mean_predictions, base=2, axis=1)

            #sorted_indices = np.argsort(entropies).tolist() # from low ent to high ent         
            
            # will return a vector of size N
            mean_accuracies = self.predict_with_uncertainty(mc_model, neighborhood_real_data)
            sorted_indices = np.argsort(mean_accuracies).tolist() # sort is asc, higher accuracies at the end of the list
            

            # choose those with lowest accuracy
            for candidate_i in range(len(sorted_indices)):
                percentile_10 = math.ceil(args.neighbour_size/100*10)
                percentile_10 = 0
                candidate = neighbor_indices[sorted_indices][candidate_i + percentile_10]
                if candidate in already_chosen_indices:
                    continue
                    
                log_labels.append(self.whole_labels[sample_i])
                log_entropies.append(mean_accuracies[sorted_indices][candidate_i + percentile_10])    
                log_sample_ix.append(sample_i)
                return_indices.append(candidate)                        
                break
                
        self.write_log("SAMPLEIXSTART")                
        self.write_log(log_sample_ix)
        self.write_log("SAMPLEINXEND")        
        self.write_log("ENTROPYSTART")
        self.write_log(log_entropies)
        self.write_log("ENTROPYEND")    
        self.write_log("LABELSTART")            
        self.write_log(log_labels)  
        self.write_log("LABELEND")        
            
        if len(return_indices) != args.candidates_per_round:
            raise Exception('return indices not', args.candidates_per_round, 'size is:', len(return_indices))
        
        return return_indices
    
    def check_old_state(self):
        model = self.setup_model()
        train_loader = self.setup_loaders([619,16385,26137,24227,18755,8760,7805,17387,11735,6947,20310,3093,21589,16255,10524,18808,14281,14770,22968,2781,19029,23595,5170,16548,5059,6333,14030,23505,1640,6135,21983,15279,5839,3684,1740,5289,23934,10881,252,349,394,26150,7527,16715,23071,15667,5162,9676,20288,26419,26006,7252,17122,25233,22794,12109,18083,19063,13912,20724,19524,26171,13121,2742,16826,5081,7054,10147,16845,10069,4620,483,21773,18342,25051,7293,22847,2372,9695,11628,22213,1870,4651,13896,26521,21830,11232,20903,722,26312,10454,6436,20242,12455,26101,24314,12120,5417,15323,6856,16305,18094,1006,18695,19257,16438,10424,9372,12162,7725,2270,10923,9701,11226,9833,25101,8757,25491,468,7941,6347,7864,13033,246,8259,25203,654,4993,23757,6154,4761,12117,22601,3097,4924,25127,23683,22414,1337,4897,749,18047,22069,20019,1334,19550,15364,19924,26437,16721,8720,24962,17453,13725,16483,6230,10049,9325,24997,8640,22900,8195,5264,3918,17647,8099,6453,10900,23577,13683,16661,4215,11262,22223,12873,3871,9651,10482,18088,20308,6095,1913,20882,3445,10406,5475,5103,7889,21759,25031,23800,24149,1420,6376,18697,462,22287,4026,8825,23911,7176,23709,7209,575,14399,18860,8766,25851,2812,19850,21608,2789,16444,6619,5664,24630,18232,14824,10300,25137,6025,26068,11997,3292,6134,13509,19558,5493,2847,4362,15651,20363,24278,15255,9293,20171,16810,24711,8902,22992,2257,20291,22690,8900,7696,11949,6806,6099,16652,12011,7986,22513,6172,19698,4459,16240,21566,23573,25038,5364,9118,17186,16710,10054,22313,17311,24360,10601,23421,18420,15792,8425,24545,19079,15991,24217,2505,11517,13146,13136,25635,1073,26418,23440,4676,19836,13046,1281,3705,7642,11820,8192,5973,14859,1305,19561,7251,22934,1332,1594,20846,17881,26821,9110,12989,19420,2028,22932,25866,5731,5933,19470,2170,12655,8270,26690,24565,18628,2223,24304])
        val_accuracy = self.train_model(model, train_loader)     
        self.free_model(model)
        
            
            
    def mc_fps_experiment(self):
        self.write_log("N = random.randint(0, int(len_i * 0.8))")
        print("Running " + self.experiment_id)
        n_candidates_per_round=args.candidates_per_round
        
        #filtered_embeddings = np.full(self.eurosat_embeddings.shape, np.finfo(np.float64).max)
        #filtered_embeddings[~np.isin(np.arange(self.eurosat_embeddings.shape[0]), self.ignore_indices)] = self.eurosat_embeddings[~np.isin(np.arange(self.eurosat_embeddings.shape[0]), self.ignore_indices)]
        

        chosen_indices = []
        blacklist = []
        val_accuracy = 0
        
        all_accuracies = []

        mc_model = None #self.setup_model()
        main_model = None
        
#        experiment_id = uuid.uuid4()
        
        iteration = 0
        while val_accuracy < args.criterion and len(chosen_indices) < 576: # iteration < 16 + 1: #

            '''
            if len(chosen_indices) == 64:
                water_ix = np.where(self.whole_labels == 9)[0].tolist()
                for wix in water_ix:
                    if wix not in chosen_indices:
                        self.ignore_indices.append(wix)

            filtered_embeddings = np.full(self.whole_embeddings.shape, np.finfo(np.float64).max)
            filtered_embeddings[~np.isin(np.arange(self.whole_embeddings.shape[0]), self.ignore_indices)] = self.whole_embeddings[~np.isin(np.arange(self.whole_embeddings.shape[0]), self.ignore_indices)]
            nearest_neighbors = NearestNeighbors(n_neighbors=args.neighbour_size).fit(filtered_embeddings)
            '''

            if iteration == 0:
                filtered_embeddings = np.full(self.whole_embeddings.shape, np.finfo(np.float64).max)
                filtered_embeddings[~np.isin(np.arange(self.whole_embeddings.shape[0]), self.ignore_indices)] = self.whole_embeddings[~np.isin(np.arange(self.whole_embeddings.shape[0]), self.ignore_indices)]
                nearest_neighbors = NearestNeighbors(n_neighbors=args.neighbour_size).fit(filtered_embeddings)
            
            mc_model = self.setup_model()
            if iteration > 0:
                train_loader = self.setup_loaders(chosen_indices)
                val_accuracy = self.train_model(mc_model, train_loader)      
                
                self.write_log(f"# samples: {len(chosen_indices)} accuracy: {val_accuracy}")
                self.write_log(f"+---------------------------------------------------------------+")                  

                if args.mcfps or args.mcfpsv2 or args.mcfpsv3:
                    
                    predictions = []                
                    for input_tensor in self.train_loader:
                        prediction = self.predict_with_uncertainty(mc_model, input_tensor[0])
                        predictions.append(prediction)
                        
                    finallist = []
                    for pred in predictions:
                        finallist += pred.tolist()
                    
                    new_ignorables = self.train_indices[np.where(np.array(finallist) > 0.8)[0]].tolist()  # 0.8 is too certain

                    added = []
                    for wix in new_ignorables:
                        if wix not in chosen_indices:
                            self.ignore_indices.append(wix)
                            added.append(wix)

                    self.write_log(f"added to ignore: {added}")
                    
                    filtered_embeddings = np.full(self.whole_embeddings.shape, np.finfo(np.float64).max)
                    filtered_embeddings[~np.isin(np.arange(self.whole_embeddings.shape[0]), self.ignore_indices)] = self.whole_embeddings[~np.isin(np.arange(self.whole_embeddings.shape[0]), self.ignore_indices)]
                    nearest_neighbors = NearestNeighbors(n_neighbors=args.neighbour_size).fit(filtered_embeddings)


            next_ix = []
            
            if args.mcfps:
                print("choosing for mcfps")
                self.write_log('choosing for mcfps')
                next_ix = self.mcfps_get_next_indices(mc_model, nearest_neighbors, chosen_indices, n_candidates_per_round)
            
            if args.random:
                print('choosing for random')                      
                self.write_log('choosing for random')                
                next_ix = self.random_get_next_indices(mc_model, nearest_neighbors, chosen_indices, n_candidates_per_round)
                
            if args.fps:
                print('choosing for fps')                              
                self.write_log('choosing for fps')                
                next_ix = self.fps_get_next_indices(chosen_indices, n_candidates_per_round)
                
            if args.cluster_fps:
                print('choosing for cluster fps')                              
                self.write_log('choosing for cluster fps')
                next_ix = self.fps_cluster(chosen_indices)
                
            if args.cluster_mcfps:
                print('choosing for cluster mcfps')                           
                self.write_log('choosing for cluster mcfps')                
                next_ix = self.mcfps_cluster(mc_model, nearest_neighbors, chosen_indices)
                
            if args.cluster_random:
                print('choosing for cluster random')                          
                self.write_log('choosing for cluster random')                
                next_ix = self.random_cluster(chosen_indices)
                
            if args.mcfpsv2:
                print("choosing for mcfps v2")
                self.write_log('choosing for mcfps v2')
                next_ix = self.mcfpsv2_get_next_indices(mc_model, nearest_neighbors, chosen_indices, n_candidates_per_round)      

            if args.mcfpsv3:
                print("choosing for mcfps v3")
                self.write_log('choosing for mcfps v3')
                next_ix = self.mcfpsv3_get_next_indices(mc_model, nearest_neighbors, chosen_indices, n_candidates_per_round)   


            if args.cluster_osal:
                print("choosing for cluster osal")
                self.write_log('choosing for cluster osal')
                chosen_indices = self.cluster_osal_get_next_indices(iteration)   
                
            self.free_model(mc_model)

            if args.cluster_osal == False:
                chosen_indices += next_ix

            print(chosen_indices)
            
            self.write_log(chosen_indices)
            
            all_accuracies.append(val_accuracy)          
            

            iteration += 1
        
        with open(self.experiment_id + "_accuracies.npy", "wb") as f:
            np.save(f, np.array(all_accuracies))
            
    def train_model(self, mc_model, train_loader):
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(mc_model.parameters(), lr=0.1, momentum=0.9)
        
        mc_model.mc_dropout.enable()
        mc_model.mc_dropout.enable_uncertainty_mode()                
        
        lr = 0.1
        last_acc = 0
        best_acc = 0
        epochs_no_improve = 0
        early_stop = False

        for epoch in range(self.mcfps_epochs):
            if early_stop == True:
                print('early stop')
                break
            print("mc model epoch", epoch)
            mc_model.train()    
            mc_model.mc_dropout.enable()
            mc_model.mc_dropout.enable_uncertainty_mode()            
            
            adjust_learning_rate(optimizer, lr, epoch, self.mcfps_epochs)
            for i, (inputs, labels) in enumerate(train_loader, 0):
                has_nan = torch.isnan(inputs).any()
                if has_nan:
                    raise('A: has nans')
                inputs, labels = inputs.cuda(), labels.cuda()
                optimizer.zero_grad()
                outputs = mc_model(inputs)
                has_nan = torch.isnan(outputs).any()
                if has_nan:
                    raise('B: has nans') 
                loss = criterion(outputs, labels.long())
                loss.backward()
                optimizer.step()

            if epoch%50==49:    
                running_loss_val = 0.0
                running_acc_val = 0.0
                count_val = 0
                mc_model.eval()            
                with torch.no_grad():
                    mc_model.mc_dropout.disable()
                    mc_model.mc_dropout.disable_uncertainty_mode()
                    counter = 0
                    loss_val_i = 0
                    avg_acc = 0
                    for j, data_val in enumerate(self.test_loader, 0):
                        inputs_val, labels_val = data_val[0].cuda(), data_val[1].cuda()
                        outputs_val = mc_model(inputs_val)
                        has_nan = torch.isnan(outputs_val).any()
                        if has_nan:
                            print("b: there are nans")
                        loss_val = criterion(outputs_val, labels_val.long())   
                        score_val = torch.sigmoid(outputs_val).detach().cpu()
                        average_precision_val = accuracy_score(labels_val.cpu(), torch.argmax(score_val,axis=1)) * 100.0

                        count_val += 1
                        running_loss_val += loss_val.item()
                        running_acc_val += average_precision_val   

                    #last_acc = avg_acc / counter
                    #print("epoch:", epoch, loss_val_i / counter, "average acc:", avg_acc / counter)
                    new_acc = running_acc_val/count_val
                    if new_acc > best_acc:
                        best_acc = new_acc
                    print('Epoch %d val_loss: %.3f val_acc: %.3f.' % (epoch+1, running_loss_val/count_val, running_acc_val/count_val))
                    '''
                    if last_acc > best_acc:
                        epochs_no_improve = 0
                    else:
                        epochs_no_improve += 1

                    if epochs_no_improve == 8:
                        early_stop = True

                    best_acc = last_acc
                    '''
        return best_acc

   
eex = EurosatExperiment()


# In[11]:


mask = np.ones(eex.embedding_labels.shape[0])
mask = mask.astype(np.bool)
mask[eex.ignore_indices] = False
import matplotlib.pyplot as plt
plt.bar(range(10), np.unique(eex.embedding_labels[mask], return_counts=True)[1])


# In[12]:


args


# In[13]:


eex.cluster_samplesize


# In[14]:


eex.mc_fps_experiment()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




