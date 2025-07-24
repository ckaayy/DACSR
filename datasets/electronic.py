from .base import AbstractDataset

import pandas as pd

from datetime import date
from sklearn.model_selection import train_test_split

class ELECDataset(AbstractDataset):
    @classmethod
    def code(cls):
        return 'electronics'

   
    @classmethod
    def all_raw_file_names(cls):
        return ['electronics_test_new_user.csv',
                'electronics_train.csv',
                'ratings_Electronics'
		]

    def load_ratings_df(self):
        if self.args.user_split == 'by_timestamp':
            folder_path = self._get_rawdata_folder_path()
            file_path = folder_path.joinpath('electronics_train.csv')
            df_source = pd.read_csv(file_path, sep='\t', header=None)
            #print(df_source.head())
            
            folder_path = self._get_rawdata_folder_path()
            file_path = folder_path.joinpath('electronics_test_new_user.csv')
            df_target = pd.read_csv(file_path, sep='\t', header=None)
            df_source.columns = ['uid', 'sid', 'timestamp']
            df_target.columns = ['uid', 'sid', 'timestamp']
            group_size = df_target.groupby('uid').size()
            test_users = group_size.index[group_size >= 3]
            df_target = df_target[df_target['uid'].isin(test_users)]
            group_size = df_source.groupby('uid').size()
            train_users = group_size.index[group_size >= 10]
            df_source = df_source[df_source['uid'].isin(train_users)]
            
            
        elif self.args.user_split == 'by_random':
            folder_path = self._get_rawdata_folder_path()
            file_path = folder_path.joinpath('ratings_Electronics.csv')
            df = pd.read_csv(file_path, sep=',', header=None)
            df.columns = ['uid', 'sid','rating','timestamp']

            df_time = df['timestamp']
            print(df_time.sort_values())
            df = df[df['rating'] >= 4]
            print('random split the users')
                
            items = df['sid'].unique().tolist()
            train_items,test_items = train_test_split(items,test_size=0.5,random_state=0)
            df = df[df['sid'].isin(train_items)]
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
                    
        
        df_target.sort_values(by=['uid', 'timestamp'],inplace=True)
        df_source.sort_values(by=['uid', 'timestamp'],inplace=True)
            
           
        set_uid_target=set(df_target['uid'].tolist())
        set_uid_source=set(df_source['uid'].tolist())
        set_in = (set_uid_target & set_uid_source)
        print('Check common users in the target and source:',set_in)
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




