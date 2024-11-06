from torchvision import transforms
from handlers import DW_Handler
from data import get_DW
from nets import Net, DW_Net
from query_strategies import RandomSampling, LeastConfidenceMeanPercentile, MarginSampling, LeastConfidenceMeanPercentileDropout, KCenterGreedy, KMeansSampling, MarginSamplingDropout, EntropySampling, EntropySamplingDropout, BALDDropout

params = {
          'DW':
              {'n_epoch': 15, 
               'train_args':{'batch_size': 32, 'num_workers': 1},
               'test_args':{'batch_size': 32, 'num_workers': 1},
               'optimizer_args':{'lr': 0.05, 'momentum': 0.3}
              }          
          }

def get_handler(name):
    if name == 'DW':
        return DW_Handler

def get_dataset(name, cfg):
    if name == 'DW':
        return get_DW(get_handler(name), cfg)
        
def get_net(name, device):
    if name == 'DW':
        return Net(DW_Net, params[name], device)
    else:
        raise NotImplementedError
    
def get_params(name):
    return params[name]

def get_strategy(name):
    if name == "RandomSampling":
        return RandomSampling
    elif name == "LeastConfidenceMeanPercentile":
        return LeastConfidenceMeanPercentile
    elif name == "MarginSampling":
        return MarginSampling
    elif name == "LeastConfidenceMeanPercentileDropout":
        return LeastConfidenceMeanPercentileDropout
    elif name == "KCenterGreedy":
        return KCenterGreedy
    elif name == "KMeansSampling":
        return KMeansSampling
    elif name == "MarginSamplingDropout":
        return MarginSamplingDropout
    elif name == "EntropySampling":
        return EntropySampling
    elif name == "EntropySamplingDropout":
        return EntropySamplingDropout
    elif name == "BALDDropout":
        return BALDDropout
    else:
        raise NotImplementedError
