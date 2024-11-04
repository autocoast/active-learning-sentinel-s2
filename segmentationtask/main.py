import numpy as np
import torch
from utils import get_dataset, get_net, get_strategy
from pprint import pprint
import uuid
from time import time

import sys
sys.path.append("/home/g/g260217/mc-dropout/paper_eurosat/algorithm/hydra-dw-2")

import hydra
from omegaconf import DictConfig, OmegaConf
    

@hydra.main(config_path="config", config_name="config.yaml", version_base="1.1")
def run_demo(cfg):

    experiment_id = str(uuid.uuid4())
    
    # fix random seed
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    #torch.backends.cudnn.enabled = False
    
    # device
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    
    dataset = get_dataset(cfg.dataset_name)                   # load dataset
    net = get_net(cfg.dataset_name, device)                   # load network
    strategy = get_strategy(cfg.strategy_name)(dataset, net)  # load strategy
    
    # start experiment
    
    dataset.initialize_labels(cfg.n_init_labeled, cfg.init_strategy, cfg.seed)
    
    print(f"number of labeled pool: {cfg.n_init_labeled}")
    print(f"number of unlabeled pool: {dataset.n_pool-cfg.n_init_labeled}")
    print(f"number of testing pool: {dataset.n_test}")
    print()
    
    # round 0 accuracy
    print("Round 0")
    strategy.train()
    preds = strategy.predict(dataset.get_test_data())
    print(f"Round 0 testing accuracy: {dataset.cal_test_acc(preds)}")
    
    with open(f'/home/g/g260217/mc-dropout/paper_eurosat/algorithm/hydra-dw-2/n32_n15_experiments', 'a') as f:
        f.write(f'{cfg.strategy_name}_{cfg.uncertainty_ratio}_{cfg.init_strategy},{experiment_id},{dataset.cal_test_acc(preds)}\n')
        
    for rd in range(1, cfg.n_round+1):
        print(f"Round {rd}")
    
        # query

        query_idxs = strategy.query(cfg.n_query, cfg)    
        # update labels
        strategy.update(query_idxs)
        strategy.train()
    
        # calculate accuracy
        preds = strategy.predict(dataset.get_test_data())
        print(f"Round {rd} testing accuracy: {dataset.cal_test_acc(preds)}")

        with open(f'/home/g/g260217/mc-dropout/paper_eurosat/algorithm/hydra-dw-2/n32_n15_experiments', 'a') as f:
            f.write(f'{cfg.strategy_name}_{cfg.uncertainty_ratio}_{cfg.init_strategy},{experiment_id},{dataset.cal_test_acc(preds)}\n')

if __name__ == "__main__":
    start = time()
    run_demo()
    print(f'experiment took {(time() - start) / 60}mins')