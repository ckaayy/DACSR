# -*- coding: utf-8 -*-
"""
Created on Fri Sep 17 14:00:11 2021

@author: wangl4
"""
from .base import AbstractDataset

import pandas as pd

from datetime import date
from datetime import date
import datetime
import time
from sklearn.model_selection import train_test_split

class BeautyDataset(AbstractDataset):
    @classmethod
    def code(cls):
        return 'beauty'

   
    @classmethod
    def all_raw_file_names(cls):
        return ['ratings_Beauty.csv',
		]

    def load_ratings_df(self):
        folder_path = self._get_rawdata_folder_path()
        file_path = folder_path.joinpath('ratings_Beauty.csv')
        df = pd.read_csv(file_path, sep=',', header=None)
        df.columns = ['uid', 'sid','rating','timestamp']

        df_time = df['timestamp']
        print(df_time.sort_values())
        df = df[df['rating'] >= 4]
        print(df.head())
            
        if self.args.user_split == 'by_timestamp':
        # split in train/test
            split='1/1/2014'
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
            
            # group_size = df_target.groupby('uid').size()
            # test_users = group_size.index[group_size >= 10]
            # df_target = df_target[df_target['uid'].isin(test_users)]
            # group_size = df_source.groupby('uid').size()
            # train_users = group_size.index[group_size >= 10]
            # df_source = df_source[df_source['uid'].isin(train_users)]
            
            group_size = df_target.groupby('sid').size()
            test_users = group_size.index[group_size >= 10]
            df_target = df_target[df_target['sid'].isin(test_users)]
            group_size = df_source.groupby('sid').size()
            train_users = group_size.index[group_size >= 10]
            df_source = df_source[df_source['sid'].isin(train_users)]
            
            group_size = df_target.groupby('uid').size()
            test_users = group_size.index[group_size >= 3]
            df_target = df_target[df_target['uid'].isin(test_users)]
            group_size = df_source.groupby('uid').size()
            train_users = group_size.index[group_size >= 10]
            df_source = df_source[df_source['uid'].isin(train_users)]
               
        elif self.args.user_split == 'by_random':
            
            group_size = df.groupby('uid').size()
            group_users = group_size.index[group_size >= 10]
            df = df[df['uid'].isin(group_users)]
            users = df['uid'].unique().tolist()
            print(len(df['uid'].unique().tolist()))
            train_users,test_users = train_test_split(users,test_size=0.2,random_state=0)
            print(len(train_users))  
            print(len(test_users)) 
            df_target = df[df['uid'].isin(test_users)]
            df_source = df[df['uid'].isin(train_users)]
               

            # sort the datasets
        df_target.sort_values(by=['uid', 'timestamp'],inplace=True)
        df_source.sort_values(by=['uid', 'timestamp'],inplace=True)

        set_uid_target=set(df_target['uid'].tolist())
        set_uid_source=set(df_source['uid'].tolist())
        set_in = (set_uid_target & set_uid_source)
        print('Check common users in target and source:',set_in)
        set_sid_target=set(df_target['sid'].tolist())
        set_sid_source=set(df_source['sid'].tolist())
        set_in_sid = (set_sid_target & set_sid_source)
        if self.args.itemshift == 0:
            df_target = df_target[df_target['sid'].isin(set_in_sid)]
        print("item number in target:", len(set_sid_target))
        print("item number in source:", len(set_sid_source))
        print("Common items in target and source:", len(set_in_sid))
        print("Items in target not in source:", len(set_sid_target-set_sid_source))
        return df_source, df_target#, df







