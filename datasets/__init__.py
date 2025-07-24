from .electronic import ELECDataset
from .beauty import BeautyDataset
from .ratings_Movies_and_TV import MVTVDataset
from .ml_1m import ML1MDataset
from .kindle import KinDataset
from .phone import PhoneDataset
from .books import BOOKDataset
from .toy import ToyDataset
from .sport import SportDataset
DATASETS = {
    ELECDataset.code(): ELECDataset, 
    MVTVDataset.code():MVTVDataset,
    ML1MDataset.code():ML1MDataset, 
    BeautyDataset.code():BeautyDataset,
    KinDataset.code():KinDataset,
    PhoneDataset.code():PhoneDataset,
    BOOKDataset.code():BOOKDataset,
    ToyDataset.code():ToyDataset,
    SportDataset.code():SportDataset
}


def dataset_factory(args):
    dataset = DATASETS[args.dataset_code]
    return dataset(args)
