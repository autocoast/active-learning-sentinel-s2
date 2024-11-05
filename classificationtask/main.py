import argparse
import numpy as np
import torch
from utils import get_dataset, get_net, get_strategy
from pprint import pprint
import uuid

import sys
sys.path.append("/home/g/g260217/mc-dropout/paper_eurosat/deep-active-learning")

import hydra
from omegaconf import DictConfig, OmegaConf
    

@hydra.main(config_path="config", config_name="config.yaml", version_base="1.1")
def run_demo(cfg):
    #try:
    print(cfg.n_round)

    experiment_id = str(uuid.uuid4())
    
    # fix random seed
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    #torch.backends.cudnn.enabled = False
    
    # device
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    
    dataset = get_dataset(cfg.dataset_name, cfg)                   # load dataset
    net = get_net(cfg.dataset_name, device)                        # load network
    strategy = get_strategy(cfg.strategy_name)(dataset, net, cfg)  # load strategy
    
    # start experiment
    dataset.initialize_labels(cfg.n_init_labeled, cfg.init_strategy, cfg.seed)
    print(f"number of labeled pool: {cfg.n_init_labeled}")
    print(f"number of unlabeled pool: {dataset.n_pool-cfg.n_init_labeled}")
    print(f"number of testing pool: {dataset.n_test}")
    print(cfg.strategy_name, cfg.init_strategy)
    
    # round 0 accuracy
    print("Round 0")
    strategy.train()
    preds = strategy.predict(dataset.get_test_data())
    print(f"Round 0 testing accuracy: {dataset.cal_test_acc(preds)}")

    print(cfg.strategy_name, cfg.init_strategy)

    if cfg.balance:
        save_file = f'/home/g/g260217/mc-dropout/paper_eurosat/deep-active-learning/v2_balance_experiments_n16_e15'
    else:
        save_file = f'/home/g/g260217/mc-dropout/paper_eurosat/deep-active-learning/v2_unbalance_experiments_n16_e15'
        
    with open(save_file, 'a') as f:
        f.write(f'{cfg.strategy_name}_{cfg.init_strategy},{cfg.seed},{experiment_id},{dataset.cal_test_acc(preds)}\n')
    
    for rd in range(1, cfg.n_round+1):
        print(f"Round {rd}")
    
        # query
        query_idxs = strategy.query(cfg.n_query)
    
        # update labels
        strategy.update(query_idxs)
        strategy.train()
    
        # calculate accuracy
        preds = strategy.predict(dataset.get_test_data())
        print(f"Round {rd} testing accuracy: {dataset.cal_test_acc(preds)}")

        with open(save_file, 'a') as f:
            f.write(f'{cfg.strategy_name}_{cfg.init_strategy},{cfg.seed},{experiment_id},{dataset.cal_test_acc(preds)}\n')
    '''
    except Exception as e:
        with open(f'/home/g/g260217/mc-dropout/paper_eurosat/deep-active-learning/errors', 'a') as f:
            f.write(str(e))
            f.write('\n')
            f.write(OmegaConf.to_yaml(cfg))
            f.write('=============')
    '''

if __name__ == "__main__":
    run_demo()