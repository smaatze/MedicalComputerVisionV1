
from .us_needle_dataset import USNeedleDataset
from .pennfudan_dataset import PennFudanPedDataset
from .other_dataset import Otherdataset

def get_dataset(name):

    return{
        'us_needle': USNeedleDataset,
        'od_maskrcnn': PennFudanPedDataset,
        'other': Otherdataset
    }[name]

def get_dataset_path(dataset_name, opts):
    return getattr(opts,dataset_name)