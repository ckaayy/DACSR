from .utils import *
from config import RAW_DATASET_ROOT_FOLDER

import numpy as np
import pandas as pd
from tqdm import tqdm
tqdm.pandas()
from abc import *
from pathlib import Path
import os
import tempfile
import shutil
import pickle
import random

class AbstractDataset(metaclass=ABCMeta):
    def __init__(self, args):
        self.args = args
        
        
        self.min_uc = args.min_uc
        self.min_sc = args.min_sc
        self.split = args.split

        assert self.min_uc >= 3, 'Need at least 2 ratings per user for validation and test'

    @classmethod
    @abstractmethod
    def code(cls):
        pass

    @classmethod
    def raw_code(cls):
        return cls.code()

    def load_dataset(self):
        self.preprocess()
        dataset_path = self._get_preprocessed_dataset_path()
        dataset = pickle.load(dataset_path.open('rb'))
        return dataset

    def preprocess(self):
        dataset_path = self._get_preprocessed_dataset_path()
        print(dataset_path)
        
        if dataset_path.is_file():
            print('Already preprocessed. Skip preprocessing')
            return
        if not dataset_path.parent.is_dir():
            dataset_path.parent.mkdir(parents=True)
            
        df_source, df_target = self.load_ratings_df()
 
        df_target = self.filter_triplets(df_target)

        df = pd.concat([df_source, df_target], ignore_index=True)
        print(df.head())
        print('source len:',len(df_source))
        print('target len:',len(df_target))

        df_source, df_target, umap, smap = self.densify_index(df_source, df_target, df)
        folder_path = self._get_rawdata_folder_path()
        
        df_target= df_target[['uid','sid','timestamp']]
        #df_target.to_csv(file_path, index=False, header=False)
        #file_path = folder_path.joinpath(self.args.dataset_code +'_train.csv')
        df_source= df_source[['uid','sid','timestamp']]
        #df_source.to_csv(file_path, index=False, header=False)
        df_target.sort_values(by=['uid', 'timestamp'],inplace=True)
        df_source.sort_values(by=['uid', 'timestamp'],inplace=True)
        train_source, train_target, train_combine, val, test = self.split_df(df_source, df_target)
        
        # umap = {u: i+1 for i, u in enumerate(set(df_source['uid']))}
        # smap = {s: i+1 for i, s in enumerate(set(df_source['sid']))}
        # df_source['uid'] = df_source['uid'].map(umap)
        # df_source['sid'] = df_source['sid'].map(smap)
        # file_path = folder_path.joinpath(self.args.dataset_code +'_train.csv')
        # df_source.to_csv(file_path, index=False, header=False)
        # df_target['sid'] = df_target['sid'].map(smap)
        # umap_t = {u: i+len(umap)+1 for i, u in enumerate(set(df_target['uid']))}
        # df_target['uid'] = df_target['uid'].map(umap_t)
        # file_path = folder_path.joinpath(self.args.dataset_code + '_test_new_user.csv')
        # df_target.to_csv(file_path, index=False, header=False)
        dataset = {'train_source': train_source,
		           'train_target': train_target,
                   'train_combine': train_combine,
                   'val': val,
                   'test': test,
                   'umap': umap,
                   'smap': smap,
                   'df_target': df_target,
                   'df_source': df_source,
                   }
        with dataset_path.open('wb') as f:
            pickle.dump(dataset, f)


    def filter_triplets(self, df):
        print('Filtering triplets')
        folder_path = self._get_rawdata_folder_path()
        if self.min_sc > 0:
            item_sizes = df.groupby('sid').size()
            good_items = item_sizes.index[item_sizes >= self.min_sc]
            df = df[df['sid'].isin(good_items)]

        if self.min_uc > 0:
            user_sizes = df.groupby('uid').size()
            good_users = user_sizes.index[user_sizes >= self.min_uc]
            df = df[df['uid'].isin(good_users)]
        return df

    def densify_index(self, df_source, df_target, df):
        print('Densifying index')
        folder_path = self._get_rawdata_folder_path()
        umap = {u: i+1 for i, u in enumerate(set(df['uid']))}
        smap = {s: i+1 for i, s in enumerate(set(df['sid']))}
        df_source['uid'] = df_source['uid'].map(umap)
        df_source['sid'] = df_source['sid'].map(smap)	
        user_sizes = df_source.groupby('uid').size()
        print(user_sizes.describe())
        # print('source n-user:',len(user_sizes))
        # print('source mean:',user_sizes.mean())
        # print('source max:',user_sizes.max())
        # print('source min:',user_sizes.min())
        df_target['uid'] = df_target['uid'].map(umap)
        df_target['sid'] = df_target['sid'].map(smap)
        user_sizes = df_target.groupby('uid').size()
        print(user_sizes.describe())
        # print('target n-user:',len(user_sizes))
        # print('target mean:',user_sizes.mean())
        # print('target max:',user_sizes.max())
        # print('target min:',user_sizes.min())
        return df_source, df_target, umap, smap
    
    def split_df(self, df_source, df_target):
        if self.args.split == 'leave_one_out':
            print('leave_one_out Splitting')
            train_source,train_target, train_combine, val, test = {}, {}, {},{}, {}
            user_group = df_source.groupby('uid')
            user2items = user_group.progress_apply(lambda d: list(d['sid']))
            for user in set(df_source['uid']):    
                items = user2items[user]
                train_source[user] = items
                train_combine[user] = items

            user_group = df_target.groupby('uid')
            user2items = user_group.progress_apply(lambda d: list(d['sid'])) 
            user_cold_len = []
            for user in set(df_target['uid']):
                items = user2items[user]
                if len(items) <= self.args.max_target_len:
                    items_cold = items
                else:
                    items_cold = items[:self.args.max_target_len]
                user_cold_len.append(len(items_cold))
                train_target[user], val[user], test[user] = items_cold[:-2], items_cold[-2:-1], items_cold[-1:]
                train_combine[user] = items_cold[:-2]
            user_cold_df = pd.DataFrame(user_cold_len)
            print('user cold description:',user_cold_df.describe())
            print('Num of users in Train Combine', len(train_combine))
            
        elif self.args.split == 'random_in_target':
            random.seed(10)
            print('random_in_target Splitting')
            train_source,train_target, train_combine, val, test = {}, {}, {},{}, {}
            user_group = df_source.groupby('uid')
            user2items = user_group.progress_apply(lambda d: list(d['sid']))
            for user in set(df_source['uid']):    
                items = user2items[user]
                train_source[user] = items
                train_combine[user] = items

            user_group = df_target.groupby('uid')
            user2items = user_group.progress_apply(lambda d: list(d['sid'])) 
            user_cold_len = []
            for user in set(df_target['uid']):
                items = user2items[user]
                if len(items) <= self.args.max_target_len:
                    items_cold = items
                else:
                    items_cold = items[:self.args.max_target_len]
                if random.random()<0.3:
                    train_target[user] = items_cold[:-1]
                    val[user] =  items_cold[-1:]
                else:
                    train_target[user] = items_cold[:-1]
                    test[user] =  items_cold[-1:]
                user_cold_len.append(len(items_cold))
                train_combine[user] = train_target[user]
            user_cold_df = pd.DataFrame(user_cold_len)
            print('user cold description:',user_cold_df.describe())
            print('Num of users in Train Combine', len(train_combine))
        else:
            raise NotImplementedError

        return train_source, train_target, train_combine, val, test

    def _get_rawdata_root_path(self):
        return Path(RAW_DATASET_ROOT_FOLDER)

    def _get_rawdata_folder_path(self):
        root = self._get_rawdata_root_path()
        return root.joinpath(self.raw_code())

    def _get_preprocessed_root_path(self):
        root = self._get_rawdata_root_path()
        return root.joinpath('preprocessed')

    def _get_preprocessed_folder_path(self):
        preprocessed_root = self._get_preprocessed_root_path()
        folder_name = '{}-min_uc{}-min_sc{}-split{}-maxlen{}-itemshift{}-usersplit{}' \
            .format(self.code(),self.min_uc, self.min_sc, self.split,self.args.max_target_len,self.args.itemshift,self.args.user_split)
        return preprocessed_root.joinpath(folder_name)

    def _get_preprocessed_dataset_path(self):
        folder = self._get_preprocessed_folder_path()
        return folder.joinpath('dataset_cold.pkl')

