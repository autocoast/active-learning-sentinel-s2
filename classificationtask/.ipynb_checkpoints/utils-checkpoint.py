from torchvision import transforms
from handlers import EuroSAT_Handler
from data import get_EuroSAT
from nets import Net, EuroSAT_Net
from query_strategies import RandomSampling, LeastConfidence, MarginSampling, EntropySampling, \
                             LeastConfidenceDropout, MarginSamplingDropout, EntropySamplingDropout, \
                             KMeansSampling, KCenterGreedy, BALDDropout, \
                             AdversarialBIM, AdversarialDeepFool

params = {
            'EuroSAT':
            {
                'n_epoch':15, 
                'train_args':{'batch_size': 16, 'num_workers': 1},
                'test_args':{'batch_size': 16, 'num_workers': 1},
                'optimizer_args':{'lr': 0.05, 'momentum': 0.3}
            }
         }

def get_handler(name):
    if name == 'EuroSAT':
        return EuroSAT_Handler

def get_dataset(name, cfg):
    if name == 'EuroSAT':
        return get_EuroSAT(get_handler(name), cfg)
    else:
        raise NotImplementedError
        
def get_net(name, device):
    if name == 'EuroSAT':
        return Net(EuroSAT_Net, params[name], device)
    else:
        raise NotImplementedError
    
def get_params(name):
    return params[name]

def get_strategy(name):
    if name == "RandomSampling":
        return RandomSampling
    elif name == "LeastConfidence":
        return LeastConfidence
    elif name == "MarginSampling":
        return MarginSampling
    elif name == "EntropySampling":
        return EntropySampling
    elif name == "LeastConfidenceDropout":
        return LeastConfidenceDropout
    elif name == "MarginSamplingDropout":
        return MarginSamplingDropout
    elif name == "EntropySamplingDropout":
        return EntropySamplingDropout
    elif name == "KMeansSampling":
        return KMeansSampling
    elif name == "KCenterGreedy":
        return KCenterGreedy
    elif name == "BALDDropout":
        return BALDDropout
    elif name == "AdversarialBIM":
        return AdversarialBIM
    elif name == "AdversarialDeepFool":
        return AdversarialDeepFool
    else:
        raise NotImplementedError