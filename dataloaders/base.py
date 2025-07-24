from abc import *
import random


class AbstractDataloader(metaclass=ABCMeta):
    def __init__(self, args, dataset):
        self.args = args
        seed = args.dataloader_random_seed
        self.rng = random.Random(seed)
        self.save_folder = dataset._get_preprocessed_folder_path()
        dataset = dataset.load_dataset()
        self.train_source = dataset['train_source']
        self.train_target = dataset['train_target']
        self.df_source = dataset['df_source']
        self.df_target = dataset['df_target']
        self.train_combine = dataset['train_combine']
        self.val = dataset['val']
        self.test = dataset['test']
        self.umap = dataset['umap']
        self.smap = dataset['smap']
        self.user_count_target = len(self.test)
        self.item_count = len(self.smap)
        print('number of items:',self.item_count)
        self.mask_prob = args.mask_prob
        

    @classmethod
    @abstractmethod
    def code(cls):
        pass

    @abstractmethod
    def get_pytorch_dataloaders(self):
        pass
