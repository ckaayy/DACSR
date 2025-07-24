from .base import AbstractDataset

import pandas as pd
from datetime import date
import datetime
import time



class ML1MDataset(AbstractDataset):
    @classmethod
    def code(cls):
        return 'ml-1m'

    @classmethod
    def url(cls):
        return 'http://files.grouplens.org/datasets/movielens/ml-1m.zip'

    @classmethod
    def zip_file_content_is_folder(cls):
        return True

    @classmethod
    # def all_raw_file_names(cls):
    #     return ['README',
    #             'movies.dat',
    #             'ratings.dat',
    #             'users.dat']

    # def load_ratings_df(self):
    #     folder_path = self._get_rawdata_folder_path()
    #     file_path = folder_path.joinpath('ratings.dat')
    #     df = pd.read_csv(file_path, sep='::', header=None)
    #     df.columns = ['uid', 'sid', 'rating', 'timestamp']
    #     return df
    def all_raw_file_names(cls):
         return ['ratings.dat']

    def load_ratings_df(self):
        folder_path = self._get_rawdata_folder_path()
        file_path = folder_path.joinpath('ratings.dat')
        df = pd.read_csv(file_path, sep='::', header=None)
        df.columns = ['uid', 'sid', 'rating', 'timestamp']
        df_time = df['timestamp']
        print(df_time.sort_values())
        df = df[df['rating'] >= 4]
        split='12/10/2000'
        
    # split in train/test
        month, day, year = split.split('/')
        date_time = datetime.datetime(int(year),int(month),int(day))
        unix_timestamp = int(time.mktime(date_time.timetuple()))
        print("unix_timestamp: ",(unix_timestamp))
        df_group = df.groupby('uid').agg({'timestamp':['min','max']})
        which_users = df_group[df_group.columns[0]] >= unix_timestamp
        test_users = df_group.index[which_users]
        df_target = df[df['uid'].isin(test_users)]
#    which_users = df_group[df_group.columns[1]] < unix_timestamp
        which_users = df_group[df_group.columns[0]] < unix_timestamp
        train_users = df_group.index[which_users]
        df_source = df[df['uid'].isin(train_users)]

        # filter out
        group_size = df_target.groupby('uid').size()
        test_users = group_size.index[group_size >= 3]
        df_target = df_target[df_target['uid'].isin(test_users)]
        group_size = df_source.groupby('uid').size()
        train_users = group_size.index[group_size >= 10]
        df_source = df_source[df_source['uid'].isin(train_users)]

        group_size = df_target.groupby('sid').size()
        test_users = group_size.index[group_size >= 5]
        df_target = df_target[df_target['sid'].isin(test_users)]
        group_size = df_source.groupby('sid').size()
        train_users = group_size.index[group_size >= 5]
        df_source = df_source[df_source['sid'].isin(train_users)]

        # sort the datasets
        df_target.sort_values(by=['uid', 'timestamp'],inplace=True)
        df_source.sort_values(by=['uid', 'timestamp'],inplace=True)

        print(df_target.head())
        print(df_source.head())
        return df_source,df_target


