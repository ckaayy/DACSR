# -*- coding: utf-8 -*-
"""
Created on Thu Apr 22 14:30:22 2021

@author: wangl4
"""
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 19 04:22:23 2021

@author: wangl4
"""
from .base import AbstractDataloader
from .negative_samplers import negative_sampler_factory
import numpy as np
import torch
import torch.utils.data as data_utils
from torch.nn import functional as F
import sys
import pandas as pd
from utils import *
from .rbo import rbo
# import ipdb

def random_neq(l, r, s):
    t = np.random.randint(l, r)
    while t in s:
        t = np.random.randint(l, r)
    return t

class SASRecDataloader(AbstractDataloader):
    def __init__(self, args, dataset):
        super().__init__(args, dataset)
        # args.num_items = len(self.smap)
        # self.max_len = args.max_len
        # self.max_len_s = args.max_len_s
        # self.sample_type = args.sample_type
        # self.wt_sampling = args.wt_sampling
        
        # code = args.test_negative_sampler_code
        # test_negative_sampler = negative_sampler_factory(code,  self.train_target, self.val, self.test,
        #                                                  self.user_count_target, self.item_count,
        #                                                  args.test_negative_sample_size,
        #                                                  args.test_negative_sampling_seed,
        #                                                  self.save_folder)
      
        # self.test_negative_samples = test_negative_sampler.get_negative_samples()
        
        # self.pop_target = self.df_target['sid'].value_counts() # line 38-55: popularity of items in target and source for weighted sampling
        # item_set = self.df_target['sid'].unique().tolist()
        # self.df_value_counts = pd.DataFrame(self.pop_target).reset_index()
        
        # self.df_value_counts.columns = ['sid', 'counts']
        # if self.wt_sampling is None or 'pop_ratio' in self.wt_sampling:
        #     self.df_value_counts['pop']= self.df_value_counts['counts'].div(self.df_value_counts['counts'].max())
        # elif 'density_ratio' in self.wt_sampling:
        #     self.df_value_counts['pop']= self.df_value_counts['counts'].div(self.df_value_counts['counts'].sum())
        # elif 'rank_ratio' in self.wt_sampling:
        #     unique_counts = self.df_value_counts['counts'].unique()
        #     dict_count_to_rarityrank = {v:len(unique_counts)-i for i,v in enumerate(unique_counts)}
        #     rarityrank = [dict_count_to_rarityrank[v] for v in self.df_value_counts['counts']]
        #     self.df_value_counts['pop'] = rarityrank

        # print('popularity of items in target dataset')
        # #print(self.df_value_counts.head(30))
        # item_tar_pop = set(self.df_value_counts.head(30).sid.tolist())
        # print(self.df_value_counts['pop'].max())
        # print(self.df_value_counts['pop'].min())

        # #df_source_target = pd.merge(self.df_source,self.df_target, on='uid')
        # df_source_items = self.df_source[self.df_source['sid'].isin(item_set)]
        # self.pop_source = df_source_items['sid'].value_counts()
        # self.df_value_counts_source = pd.DataFrame(self.pop_source).reset_index()
        # print('popularity of the same items in source dataset')
        # self.df_value_counts_source.columns = ['sid', 'counts']
        # if self.wt_sampling is None or 'pop_ratio' in self.wt_sampling:
        #     self.df_value_counts_source['pop']= self.df_value_counts_source['counts'].div(self.df_value_counts_source['counts'].max())
        # elif 'density_ratio' in self.wt_sampling:
        #     self.df_value_counts_source['pop']= self.df_value_counts_source['counts'].div(self.df_value_counts_source['counts'].sum())
        # elif 'rank_ratio' in self.wt_sampling:
        #     unique_counts = self.df_value_counts_source['counts'].unique()
        #     dict_count_to_rarityrank = {v:len(unique_counts)-i for i,v in enumerate(unique_counts)}
        #     rarityrank = [dict_count_to_rarityrank[v] for v in self.df_value_counts_source['counts']]
        #     self.df_value_counts_source['pop'] = rarityrank
        # #print(self.df_value_counts_source.head(30))
        # item_source_pop = set(self.df_value_counts_source.head(30).sid.tolist())
        # print(self.df_value_counts_source['pop'].max())
        # print(self.df_value_counts_source['pop'].min())
        # self.test_negative_samples = test_negative_sampler.get_negative_samples()
        # print('same pop 30 item in both target  and source :', item_tar_pop & item_source_pop)
        
        
        # print(self.df_value_counts_source)
        # print(self.df_value_counts)
        
        # if args.domain_similarity is not None:
        #     export_root = setup_train(args)
        #     export_path = os.path.join(export_root, 'logs', f'{args.domain_similarity}_similarity.txt')
        #     if args.domain_similarity == 'inv_kl':  # kl = sum prop_target * log(prop_target/prop_source)
        #         # density of source and target domains, format into tensors
        #         assert ('density_ratio' in args.wt_sampling)
        #         df_value_counts_sort = self.df_value_counts.sort_values('sid')
        #         df_value_counts_source_sort = self.df_value_counts_source.sort_values('sid')
        #         density_target = torch.tensor(df_value_counts_sort['pop'])  # target
        #         density_source = torch.tensor(df_value_counts_source_sort['pop'])  # input
        #         log_density_source = torch.log(density_source)
        #         similarity = 1 / F.kl_div(log_density_source, density_target, reduction='sum').cpu().numpy()
        #     elif args.domain_similarity == 'rbo':  # rank biased overlap, implementation may be incorrect, DO NOT USE
        #         similarity = rbo(self.df_value_counts['sid'].values, self.df_value_counts_source['sid'].values, p=0.9)
        #     elif args.domain_similarity == 'jaccard':  # jaccard similarity or intersection over union of top 30 items
        #         intersection = len(item_tar_pop & item_source_pop)
        #         union = len(item_tar_pop) + len(item_source_pop) - intersection
        #         similarity = intersection / union
        #     elif args.domain_similarity == 'jaccard100':  # jaccard similarity or intersection over union
        #         item_tar_popk = set(self.df_value_counts.head(100).sid.tolist())
        #         item_source_popk = set(self.df_value_counts_source.head(100).sid.tolist())
        #         intersection = len(item_tar_popk & item_source_popk)
        #         union = len(item_tar_popk) + len(item_source_popk) - intersection
        #         similarity = intersection / union
        #         print('jaccard100',similarity)
        
        args.num_items = len(self.smap)
        self.max_len = args.max_len
        self.max_len_s = args.max_len_s
        self.sample_type = args.sample_type
        self.wt_sampling = args.wt_sampling
        
        code = args.test_negative_sampler_code
        test_negative_sampler = negative_sampler_factory(code,  self.train_target, self.val, self.test,
                                                         self.user_count_target, self.item_count,
                                                         args.test_negative_sample_size,
                                                         args.test_negative_sampling_seed,
                                                         self.save_folder)
      
        self.test_negative_samples = test_negative_sampler.get_negative_samples()
        
        self.pop_target = self.df_target['sid'].value_counts() # line 38-55: popularity of items in target and source for weighted sampling
        item_set = self.df_target['sid'].unique().tolist()
        self.df_value_counts = pd.DataFrame(self.pop_target).reset_index()
        
        self.df_value_counts.columns = ['sid', 'counts']
        if self.wt_sampling is None or 'pop_ratio' in self.wt_sampling:
            self.df_value_counts['pop']= self.df_value_counts['counts'].div(self.df_value_counts['counts'].max())
        elif 'density_ratio' in self.wt_sampling:
            self.df_value_counts['pop']= self.df_value_counts['counts'].div(self.df_value_counts['counts'].sum())
        elif 'rank_ratio' in self.wt_sampling:
            unique_counts = self.df_value_counts['counts'].unique()
            dict_count_to_rarityrank = {v:len(unique_counts)-i for i,v in enumerate(unique_counts)}
            rarityrank = [dict_count_to_rarityrank[v] for v in self.df_value_counts['counts']]
            self.df_value_counts['pop'] = rarityrank

        print('popularity of items in target dataset')
        # #print(self.df_value_counts.head(30))
        item_tar_pop = set(self.df_value_counts.head(30).sid.tolist())
        print(self.df_value_counts['pop'].max())
        print(self.df_value_counts['pop'].min())
        
        #df_source_target = pd.merge(self.df_source,self.df_target, on='uid')
        df_source_items = self.df_source[self.df_source['sid'].isin(item_set)]
        #self.pop_source = self.df_source['sid'].value_counts()
       
        self.pop_source = df_source_items['sid'].value_counts()
        self.df_value_counts_source = pd.DataFrame(self.pop_source).reset_index()
  
        print('popularity of the same items in source dataset')
        self.df_value_counts_source.columns = ['sid', 'counts']
        
        if self.wt_sampling is None or 'pop_ratio' in self.wt_sampling:
            self.df_value_counts_source['pop']= self.df_value_counts_source['counts'].div(self.df_value_counts_source['counts'].max())
          
            self.df_value_counts_source['pop']= self.df_value_counts_source['counts'].div(self.df_value_counts_source['counts'].sum())
    
        elif 'rank_ratio' in self.wt_sampling:
            unique_counts = self.df_value_counts_source['counts'].unique()
            dict_count_to_rarityrank = {v:len(unique_counts)-i for i,v in enumerate(unique_counts)}
            rarityrank = [dict_count_to_rarityrank[v] for v in self.df_value_counts_source['counts']]
            self.df_value_counts_source['pop'] = rarityrank
        #print(self.df_value_counts_source.head(30))
        item_source_pop = set(self.df_value_counts_source.head(30).sid.tolist())
        print(self.df_value_counts_source['pop'].max())
        print(self.df_value_counts_source['pop'].min())
        print(self.df_value_counts_source)
        #print(self.df_value_counts_source[self.df_value_counts_source['pop']<=0.1]['pop'])
        #print(self.df_value_counts_source[self.df_value_counts_source['sid']==12618]['pop'].item())

        
        self.test_negative_samples = test_negative_sampler.get_negative_samples()
        print('same pop 30 item in both target  and source :', item_tar_pop & item_source_pop)
     
        if args.domain_similarity is not None:
            export_root = setup_train(args)
            export_path = os.path.join(export_root, 'logs', f'{args.domain_similarity}_similarity.txt')
            if args.domain_similarity == 'inv_kl':  # kl = sum prop_target * log(prop_target/prop_source)
                # density of source and target domains, format into tensors
                assert ('density_ratio' in args.wt_sampling)
                df_value_counts_sort = self.df_value_counts.sort_values('sid')
                df_value_counts_source_sort = self.df_value_counts_source.sort_values('sid')
                density_target = torch.tensor(df_value_counts_sort['pop'])  # target
                density_source = torch.tensor(df_value_counts_source_sort['pop'])  # input
                log_density_source = torch.log(density_source)
                similarity = 1 / F.kl_div(log_density_source, density_target, reduction='sum').cpu().numpy()
            elif args.domain_similarity == 'rbo':  # rank biased overlap, implementation may be incorrect, DO NOT USE
                similarity = rbo(self.df_value_counts['sid'].values, self.df_value_counts_source['sid'].values, p=0.9)
            elif args.domain_similarity == 'jaccard':  # jaccard similarity or intersection over union of top 30 items
                intersection = len(item_tar_pop & item_source_pop)
                union = len(item_tar_pop) + len(item_source_pop) - intersection
                similarity = intersection / union
            elif args.domain_similarity == 'jaccard100':  # jaccard similarity or intersection over union
                item_tar_popk = set(self.df_value_counts.head(100).sid.tolist())
                item_source_popk = set(self.df_value_counts_source.head(100).sid.tolist())
                intersection = len(item_tar_popk & item_source_popk)
                union = len(item_tar_popk) + len(item_source_popk) - intersection
                similarity = intersection / union
                print('jaccard100',similarity)
        #     # with open(export_path, 'w') as f:
        #     #     f.write(str(similarity))
        #     #sys.exit()
        self.df_value_counts_source['pop'] = self.df_value_counts_source['pop'].apply(lambda x: 0.1 if x < 0.1 else x)   
        self.df_value_counts_source['pop'] = self.df_value_counts_source['pop'].apply(lambda x: 0.1/x)
        
        self.df_value_counts['pop'] = self.df_value_counts['pop'].apply(lambda x: 0.1 if x < 0.1 else x)  
        self.df_value_counts['pop'] = self.df_value_counts['pop'].apply(lambda x: 0.1/x)
        
        
    @classmethod
    def code(cls):
        return 'SASRec'

    def get_pytorch_dataloaders(self):
        train_source_loader = self._get_train_source_loader()
        train_combine_loader = self._get_train_combine_loader()
        train_target_loader = self._get_train_target_loader()
        val_loader = self._get_val_loader()
        test_loader = self._get_test_loader()
        return train_source_loader, train_target_loader, train_combine_loader, val_loader, test_loader
        #return train_source_loader,train_source_sample_loader, train_target_loader, val_loader, test_loader

    def _get_train_source_loader(self):
        dataset = self._get_train_source_dataset()
        dataloader = data_utils.DataLoader(dataset, batch_size=self.args.train_source_batch_size,
                                           shuffle=True, pin_memory=True)
        return dataloader
    def _get_train_combine_loader(self):
        dataset = self._get_train_combine_dataset()
        dataloader = data_utils.DataLoader(dataset, batch_size=self.args.train_combine_batch_size,
                                           shuffle=True, pin_memory=True)
        return dataloader
    def _get_train_target_loader(self):
        dataset = self._get_train_target_dataset()
        dataloader = data_utils.DataLoader(dataset, batch_size=self.args.train_target_batch_size,
                                           shuffle=True, pin_memory=True)
        return dataloader

    def _get_train_source_dataset(self):
        dataset = SASRecTrainSourceDataset(self.train_source, self.df_value_counts_source, self.df_value_counts, self.max_len_s, self.max_len,self.item_count,self.rng, self.mask_prob,self.sample_type)
        return dataset

    def _get_train_combine_dataset(self):
        dataset = SASRecTrainTargetDataset(self.train_combine, self.max_len, self.item_count,self.sample_type)
        return dataset

    def _get_train_target_dataset(self):
        dataset = SASRecTrainTargetDataset(self.train_target, self.max_len, self.item_count,self.sample_type)
        return dataset

    def _get_val_loader(self):
        batch_size = self.args.val_batch_size
        dataset = self._get_val_dataset()
        dataloader = data_utils.DataLoader(dataset, batch_size=batch_size,
                                           shuffle=False, pin_memory=True)
        return dataloader

    def _get_val_dataset(self):
        answers = self.val 
        negative_samples = self.test_negative_samples
        dataset = SASRecvalDataset(self.train_target, answers, self.max_len, negative_samples)
        return dataset
    
    def _get_test_loader(self):
        batch_size = self.args.test_batch_size
        dataset = self._get_test_dataset()
        dataloader = data_utils.DataLoader(dataset, batch_size=batch_size,
                                           shuffle=False, pin_memory=True)
        return dataloader

    def _get_test_dataset(self):
        if self.args.split == 'leave_one_out':
            answers_pre = self.val
        else:
            answers_pre = None
        answers = self.test
        negative_samples = self.test_negative_samples
        dataset = SASRectestDataset(self.train_target, answers_pre, answers, self.max_len, negative_samples)
        return dataset

class SASRecTrainSourceDataset(data_utils.Dataset):
    def __init__(self, u2seq, den_source, den_target, max_len_s,  max_len, num_items,rng,mask_prob,sample_type):
        self.u2seq = u2seq
        self.den_source = den_source
        self.den_target = den_target
        #print(self.den_source)
        #print(self.den_target)
        self.users = sorted(self.u2seq.keys())
        self.max_len_s = max_len_s
        self.max_len = max_len
        self.num_items = num_items
        self.rng = rng
        self.mask_prob = mask_prob
        self.user_ids = []
        self.rng = rng
    
        self.sample_type = sample_type
        self.samples_multiple = {}
        
        for user_id in self.users:
            #print(user_id)
            seqs = self.u2seq[user_id]
            if len(seqs) <= 1:
                continue
            else:
                if self.sample_type in ['one']:
                    self.user_ids.append(user_id)
                    #print(self.user_ids)
                elif self.sample_type in ['random']:
                    total_seqs = len(seqs)
                    if total_seqs<=self.max_len_s:
                        self.user_ids.append(user_id)
                    else:
                        self.user_ids.extend([user_id]*((total_seqs // self.max_len_s+1)))
                elif self.sample_type in ['multiple']:
                    total_seqs = len(seqs)
                    if total_seqs <= self.max_len_s/2:
                        self.user_ids.append(f"{user_id}")
                        self.samples_multiple[f"{user_id}"] = seqs
                    else:
                        for seq in range(2*total_seqs // self.max_len_s+1):
                            
                            end = int(total_seqs - 0.5*seq* self.max_len_s)
                            #print(end)
                            start = max(0,end - self.max_len_s)
                            seqs_sample = seqs[start:end]
                            if len(seqs_sample) >= 2:
                                self.user_ids.append(f"{user_id}_{seq}")
                                self.samples_multiple[f"{user_id}_{seq}"] = seqs[start:end]
                            
                else:
        
                    print('Error:Unregonized sample type')
                    sys.exit(1)

    def __len__(self):
        return len(self.user_ids)
        

    # def __len__(self):
    #     return len(self.users)

    def __getitem__(self, index): 
        user_id = self.user_ids[index]
        sample_type = self.sample_type
        tokens = []
        labels = []
        pops_s = []
        pops_ss = []
        if sample_type in ['one']:
            seq = self.u2seq[user_id]
        elif sample_type in ['random']:
            if len(self.u2seq[user_id]) > self.max_len_s:    
                end = self.rng.randint(self.max_len_s/2, len(self.u2seq[user_id]))
                start = max(0,end - self.max_len_s)
                seq = self.u2seq[user_id][start:end]
            else:
                seq = self.u2seq[user_id]
        elif sample_type in ['multiple']:
                seq = self.samples_multiple[user_id]

        for s in seq:
            index_s = seq.index(s)
            if index_s < len(seq)-1:
                    tokens.append(s)
                    labels.append(seq[index_s+1])
                    #print('sid', s)
                    #print(sorted(self.den_source["sid"].tolist()))
                    #print(sorted(self.den_target["sid"].tolist()))
                    #sys.exit()
                    if s in self.den_source["sid"].tolist():
                        pop_s = self.den_source[self.den_source["sid"]==s]['pop'].item()
                        #print(pop_t)
                    else:
                        pop_s = 0.1

                    pops_s.append(pop_s)
                    if s in self.den_target["sid"].tolist():
                        pop_t = self.den_target[self.den_target["sid"]==s]['pop'].item()
                        #print(pop_t)
                    else:
                        pop_t = 0.1
                    #print(pop_t)
                    pops_ss.append(pop_t)
                    
        tokens = tokens[-self.max_len_s:]
        labels = labels[-self.max_len_s:]
        pops_s = pops_s[-self.max_len_s:]
        pops_ss = pops_ss[-self.max_len_s:]
        token_samples = []
        label_samples = []
        pops_samples = []
        # token_samples = [0]*(len(tokens))
        # label_samples = [0]*(len(tokens))
        mask_samples = [0]*(len(tokens))
        while len([x for x in mask_samples if x > 0]) < 2:
            mask_samples = [0]*(len(tokens))
            for s in tokens:
                index_s = tokens.index(s)
                if index_s < len(tokens)-1:
                        prob = self.rng.random()
                        if prob < self.mask_prob:
                            prob /= self.mask_prob
                            if prob < 1:
                               token_samples.append(s)
                               label_samples.append(labels[index_s])
                               pops_samples.append(pops_ss[index_s])
                            #    token_samples[index_s] = s
                            #    label_samples[index_s]= pos
                               mask_samples[index_s] = 1
                if index_s == len(tokens)-1:
                        token_samples.append(s)
                        label_samples.append(labels[index_s])
                        mask_samples[index_s] = 1 
                        pops_samples.append(pops_ss[index_s])

        token_samples = token_samples[-self.max_len:]
        label_samples = label_samples[-self.max_len:]
        pops_samples = pops_samples[-self.max_len:]
        mask_samples = mask_samples[-self.max_len:]
        
        mask_len_1 = self.max_len - len(tokens)
        mask_len_2 = self.max_len - len(token_samples)

        tokens = [0] * mask_len_1 + tokens
        labels = [0] * mask_len_1 + labels
        pops_s = [0] * mask_len_1 + pops_s
        token_samples = [0] * mask_len_2 + token_samples
        label_samples = [0] * mask_len_2 + label_samples
        pops_samples = [0] * mask_len_2 + pops_samples
        mask_samples = [0] * mask_len_1 + mask_samples

        return  torch.LongTensor(tokens), torch.LongTensor(labels), torch.LongTensor(pops_s), torch.LongTensor(token_samples), torch.LongTensor(label_samples),torch.LongTensor(pops_samples), torch.LongTensor(mask_samples)

    def _getseq(self, user):
        return self.u2seq[user]



class SASRecTrainTargetDataset(data_utils.Dataset):
    def __init__(self, u2seq, max_len, num_items,sample_type):
        self.u2seq = u2seq
        self.users = sorted(self.u2seq.keys())
        self.max_len = max_len
        self.num_items = num_items
        self.user_ids = []
        self.sample_type = sample_type
        self.samples_multiple = {}
        
        for user_id in self.users:
            #print(user_id)
            seqs = self.u2seq[user_id]
            if len(seqs) == 0:
                continue
            else:
                if self.sample_type in ['one']:
                    self.user_ids.append(user_id)
                    #print(self.user_ids)
                elif self.sample_type in ['random']:
                    total_seqs = len(seqs)
                    if total_seqs<=self.max_len:
                        self.user_ids.append(user_id)
                    else:
                        self.user_ids.extend([user_id]*((total_seqs // self.max_len+1)))
                elif self.sample_type in ['multiple']:
                    total_seqs = len(seqs)
                    if total_seqs <= self.max_len/2:
                        self.user_ids.append(f"{user_id}")
                        self.samples_multiple[f"{user_id}"] = seqs
                    else:
                        for seq in range(2*total_seqs // self.max_len+1):
                            end = int(total_seqs - 0.5*seq* self.max_len)
                            #print(end)
                            start = max(0,end - self.max_len)
                            seqs_sample = seqs[start:end]
                            if len(seqs_sample) >= 2:
                                self.user_ids.append(f"{user_id}_{seq}")
                                self.samples_multiple[f"{user_id}_{seq}"] = seqs[start:end]
                else:
        
                    print('Error:Unregonized sample type')
                    sys.exit(1)

    def __len__(self):
        return len(self.user_ids)

    # def __len__(self):
    #     return len(self.users)

    def __getitem__(self, index):
        user_id = self.user_ids[index]
        sample_type = self.sample_type
        tokens = []
        labels = []
    
        if sample_type in ['one']:
            seq = self.u2seq[user_id]
        elif sample_type in ['random']:
            if len(self.u2seq[user_id]) > self.max_len:    
                end = self.rng.randint(self.max_len/2, len(self.u2seq[user_id]))
                start = max(0,end - self.max_len)
                seq = self.u2seq[user_id][start:end]
            else:
                seq = self.u2seq[user_id]
        elif sample_type in ['multiple']:
                seq = self.samples_multiple[user_id]
        
        for s in seq:
            index_s = seq.index(s)
            if index_s < len(seq)-1:
                    tokens.append(s)
                    labels.append(seq[index_s+1])


        tokens = tokens[-self.max_len:]
        labels = labels[-self.max_len:]
        
        mask_len_1 = self.max_len - len(tokens)
      

        tokens = [0] * mask_len_1 + tokens
        labels = [0] * mask_len_1 + labels
    
        return  torch.LongTensor(tokens), torch.LongTensor(labels)#, torch.LongTensor(seq), torch.LongTensor(pos)#, torch.LongTensor(neg)

    def _getseq(self, user):
        return self.u2seq[user]

class SASRecvalDataset(data_utils.Dataset):
    def __init__(self, u2seq, u2answer, max_len,  negative_samples):
        self.u2seq = u2seq
        #self.users = sorted(self.u2seq.keys())
        self.u2answer = u2answer
        self.users = sorted(self.u2answer.keys())
        self.max_len = max_len
        self.negative_samples = negative_samples

    def __len__(self):
        return len(self.users)

    def __getitem__(self, index):
        user = self.users[index]
        seq = self.u2seq[user]

        #print('val data')
        answer = self.u2answer
        seq = seq[-self.max_len:]
        padding_len = self.max_len - len(seq)
        seq = [0] * padding_len + seq
            
            
        negs = self.negative_samples[user]    
        candidates = answer[user] + negs
        labels = [1] * len(answer[user]) + [0] * len(negs)

        return torch.LongTensor(seq), torch.LongTensor(candidates), torch.LongTensor(labels)


class SASRectestDataset(data_utils.Dataset):
    def __init__(self, u2seq, u2answer_pre, u2answer, max_len,  negative_samples):
        self.u2seq = u2seq
        #self.users = sorted(self.u2seq.keys())
        self.u2answer_pre = u2answer_pre
        self.u2answer = u2answer
        self.users = sorted(self.u2answer.keys())
        self.max_len = max_len
        self.negative_samples = negative_samples

    def __len__(self):
        return len(self.users)

    def __getitem__(self, index):
        user = self.users[index]
        seq = self.u2seq[user]
        
        #print('creating test sequence')
        answer= self.u2answer
        if self.u2answer_pre != None:
            answer_pre= self.u2answer_pre
            seq = seq + answer_pre[user]
    
        seq = seq[-self.max_len:]
        padding_len = self.max_len - len(seq)
        seq = [0] * padding_len + seq
       # print('test sequence created')
            
            
        negs = self.negative_samples[user]    
        candidates = answer[user] + negs
        labels = [1] * len(answer[user]) + [0] * len(negs)

        return torch.LongTensor(seq), torch.LongTensor(candidates), torch.LongTensor(labels)


