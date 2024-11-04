from torchvision import transforms
from handlers import DW_Handler
from data import get_DW
from nets import Net, DW_Net
from query_strategies import RandomSampling, LeastConfidence, LeastConfidenceMoco, LeastConfidenceV2, LeastConfidenceEntropy, LeastConfidenceMedian, LeastConfidenceMocoMedian, LeastConfidenceMocoMargin, LeastConfidenceMocoMarginV2, LeastConfidenceMoCoGMM,  LeastConfidenceMoCoKMeans, LeastConfidenceMoCoFPS, LeastConfidenceMedianFPS, LeastConfidenceMeanPercentile, MarginSampling, LeastConfidenceMeanPercentileAlternate, LeastConfidenceMeanPercentileDropout, KCenterGreedy, KMeansSampling, MarginSamplingDropout, EntropySampling, EntropySamplingDropout, BALDDropout

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

def get_dataset(name):
    if name == 'DW':
        return get_DW(get_handler(name))
        raise NotImplementedError
        
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
    elif name == "LeastConfidence":
        return LeastConfidence
    elif name == "LeastConfidenceMoco":
        return LeastConfidenceMoco
    elif name == "LeastConfidenceV2":
        return LeastConfidenceV2
    elif name == "LeastConfidenceEntropy":
        return LeastConfidenceEntropy
    elif name == "LeastConfidenceMedian":
        return LeastConfidenceMedian
    elif name == "LeastConfidenceMocoMedian":
        return LeastConfidenceMocoMedian
    elif name == "LeastConfidenceMocoMargin":
        return LeastConfidenceMocoMargin
    elif name == "LeastConfidenceMoCoGMM":
        return LeastConfidenceMoCoGMM
    elif name == "LeastConfidenceMoCoKMeans":
        return LeastConfidenceMoCoKMeans
    elif name == "LeastConfidenceMoCoFPS":
        return LeastConfidenceMoCoFPS
    elif name == "LeastConfidenceMedianFPS":
        return LeastConfidenceMedianFPS
    elif name == "LeastConfidenceMeanPercentile":
        return LeastConfidenceMeanPercentile
    elif name == "MarginSampling":
        return MarginSampling
    elif name == "LeastConfidenceMocoMarginV2":
        return LeastConfidenceMocoMarginV2
    elif name == "LeastConfidenceMeanPercentileAlternate":
        return LeastConfidenceMeanPercentileAlternate
    elif name == "LeastConfidenceMocoMarginV2":
        return LeastConfidenceMocoMarginV2
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
