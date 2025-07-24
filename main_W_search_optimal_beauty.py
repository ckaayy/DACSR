import torch

from models import model_factory
from dataloaders import dataloader_factory
from trainers import trainer_factory
from utils import *

from datasets import DATASETS
from dataloaders import DATALOADERS
from models import MODELS
from trainers import TRAINERS
import json
import os
import argparse
import pickle
from optimal_search import optimal_search


def str2bool(s):
    if s not in {'false', 'true'}:
        raise ValueError('Not a valid boolean string')
    return s == 'true'

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='RecPlay')
################
# Dataset
################
    dataset_default = 'beauty'
    shift_default = 0
    parser.add_argument('--itemshift', type=int, default=shift_default, help='keep items in target not in source')
    parser.add_argument('--dataset_code', type=str, default=dataset_default, choices=['electronics','ratings_Movies_and_TV'])
    sample_type_default = 'one'
    parser.add_argument('--sample_type', type=str, default=sample_type_default,choices=['one','random','multiple'], help='Only keep users with more than min_uc ratings')
    user_split_default = 'by_timestamp'
    parser.add_argument('--user_split', type=str, default=user_split_default,choices=['by_timestamp','by_random'], help='Only keep users with more than min_uc ratings')
    parser.add_argument('--min_uc', type=int, default=4, help='Only keep users with more than min_uc ratings')
    parser.add_argument('--min_sc', type=int, default=0, help='Only keep items with more than min_sc ratings')
    split_default='random_in_target'
    parser.add_argument('--split', type=str, default=split_default, help='How to split the datasets')
    parser.add_argument('--dataset_split_seed', type=int, default=98765)#'leave_one_out'
    max_target_len_default = 4
    parser.add_argument('--max_target_len', type=int, default=max_target_len_default, help='max cold start lens for users')


################
# Dataloader
################    
    parser.add_argument('--dataloader_code', type=str, default='SASRec')
    parser.add_argument('--dataloader_random_seed', type=float, default=0.0)
    parser.add_argument('--train_target_batch_size', type=int, default=128)
    parser.add_argument('--train_source_batch_size', type=int, default=128)
    parser.add_argument('--train_combine_batch_size', type=int, default=128)
    parser.add_argument('--val_batch_size', type=int, default=128)
    parser.add_argument('--test_batch_size', type=int, default=128)
################
# NegativeSampler
################
    parser.add_argument('--test_negative_sampler_code', type=str, default='random', choices=['random'],
                    help='Method to sample negative items for testing')
    parser.add_argument('--test_negative_sample_size', type=int, default=100)
    parser.add_argument('--test_negative_sampling_seed', type=int, default=98765)
################
  #Trainer initialization  will be reset in template according to args.template
################ 
    parser.add_argument('--which_trainer', type=str, default='DASR',choices=['DASR','Source-only','Target-only'])#, choices=TRAINERS.keys()

    parser.add_argument('--trainer_code', type=str, default='CEW_trainer')#, choices=TRAINERS.keys()
# device #
    parser.add_argument('--device', type=str, default='cuda', choices=['cpu', 'cuda'])
    parser.add_argument('--num_gpu', type=int, default=1)
    parser.add_argument('--device_idx', type=str, default='0')
# optimizer #
    parser.add_argument('--optimizer', type=str, default='Adam', choices=['Adam'])
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0, help='l2 regularization')
    parser.add_argument('--momentum', type=float, default=None, help='SGD momentum')
# max epochs #
    parser.add_argument('--num_epochs', type=int, default=200, help='Number of epochs for training')
# early stop num #
    parser.add_argument('--early_stop_num', type=int, default=10, help='Stop learning is score doesnot improve for certain num of epochs')
# logger #
    parser.add_argument('--log_period_as_iter', type=int, default=12800)
# evaluation #
    parser.add_argument('--metric_ks', nargs='+', type=int, default=[1,5, 10], help='ks for Metric@k')
    parser.add_argument('--best_metric', type=str, default='NDCG@10', help='Metric for determining the best model')
###############
# Parameters for loss
#################
    alpha_default = 0.1
    beta_t_default= 0.06
    beta_s_default= 1
    beta_st_default = 0
    alpha_sst_default = 0
    parser.add_argument('--alpha', type=float, default=alpha_default, help='regularization parameter for alignment between source and sampled source')
    parser.add_argument('--alpha_sst', type=float, default=alpha_sst_default, help='regularization parameter for loss_coral for sampled source and target')
    parser.add_argument('--beta_t', type=float, default=beta_t_default, help='regularization parameter for loss_target')
    parser.add_argument('--beta_s', type=float, default=beta_s_default, help='regularization parameter for loss_source')   
    parser.add_argument('--beta_st', type=float, default=beta_st_default, help='regularization parameter for loss_source_sampled')     
################
# Model initialization will be reset in templates accordding to the args.template
################
    parser.add_argument('--model_code', type=str, default='SASRecW')#, choices=MODELS.keys()
    parser.add_argument('--model_init_seed', type=int, default=0)
    regularization_default = 'KL'
    parser.add_argument('--context_is_mean', type=str, default='true',choices=['false', 'true'],help='this is only required by Contrastive and Cosine regularizations')
    parser.add_argument('--regularization', type=str, default=regularization_default,choices=['KL', 'Cosine','Coral','MMD','Contrastive'])
    parser.add_argument('--mask_prob', type=float, default=0.7,help='sampling probablity of source data')
# Transformer#
# variable parameters that will be searched for optimal for sequence model of Transformer#
    parser.add_argument('--wt_sampling', type=str, default='pop_ratio', help='Size of hidden vectors (d_model)')
    parser.add_argument('--domain_similarity', type=str, default='jaccard100', help='Size of hidden vectors (d_model)')
    parser.add_argument('--max_len', type=int, default=30, help='Length of sequence for Transformer')
    parser.add_argument('--max_len_s', type=int, default=30, help='Length of sequence for Transformer')
    parser.add_argument('--num_items', type=int, default=None, help='Number of total items')
    parser.add_argument('--hidden_units', type=int, default=64, help='Size of hidden vectors (d_model)')

    parser.add_argument('--num_blocks', type=int, default=2, help='Number of transformer layers')
    parser.add_argument('--num_heads', type=int, default=1, help='Number of heads for multi-attention')
    parser.add_argument('--dropout_rate', type=float, default=0.1, help='Dropout probability to use throughout the model')
    parser.add_argument('--dropout_rate_emb', type=float, default=0.1, help='Dropout probability to use on the embedding')
#################
# Experimentself.beta_t = args.beta_t
################
   
    parser.add_argument('--experiment_dir', type=str, default='experiments_091324_shift%s_dataset%s_sample%s_split%s_usersplit%s_max_target_len%s'%(shift_default, dataset_default,sample_type_default,split_default,user_split_default,max_target_len_default))
    parser.add_argument('--experiment_description', type=str, default='test')

    args = parser.parse_args()
    ################
   ###################################################################
    space = {'max_len_list':[20],
             'hidden_units_list': [64],
             'learning_rate_list':[0.001], #[0.0005],
             'drop_rate_list': [0.1, 0.2, 0.3, 0.5],#[0.1],
             'drop_rate_emb_list':[0.3, 0.4, 0.5],#[0.4],
             'num_head_list':[1]#[1,2,4]
            }
    optimal_search(parser, space)