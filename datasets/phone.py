from .base import AbstractDataset

import pandas as pd

from datetime import date


class PhoneDataset(AbstractDataset):
    @classmethod
    def code(cls):
        return 'phone'

   
    @classmethod
    def all_raw_file_names(cls):
        return ['ratings_Cell_Phones_and_Accessories_test.csv',
                'ratings_Cell_Phones_and_Accessories_train.csv'
		]

    def load_ratings_df(self):
            folder_path = self._get_rawdata_folder_path()
            file_path = folder_path.joinpath('ratings_Cell_Phones_and_Accessories_train.csv')
            df_source = pd.read_csv(file_path, sep=',', header=None)
            #print(df_source.head())
            
            folder_path = self._get_rawdata_folder_path()
            file_path = folder_path.joinpath('ratings_Cell_Phones_and_Accessories_test.csv')
            df_target = pd.read_csv(file_path, sep=',', header=None)
            df_source.columns = ['uid', 'sid', 'timestamp']
            df_target.columns = ['uid', 'sid', 'timestamp']
        #df.columns = ['uid', 'sid', 'timestamp']
            return df_source, df_target#, df




