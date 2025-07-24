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
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

def train():
    export_root = setup_train(args)
    # train_loader_source, train_loader_source_sample, train_loader_target, val_loader, test_loader = dataloader_factory(args)
    # model = model_factory(args)
    # trainer = trainer_factory(args, model,train_loader_source,train_loader_source_sample, train_loader_target, val_loader, test_loader, export_root)
    train_loader_source, train_loader_target, train_loader_combine,val_loader, test_loader = dataloader_factory(args)
    model = model_factory(args)
    trainer = trainer_factory(args, model,train_loader_source, train_loader_target, train_loader_combine,val_loader, test_loader, export_root)
    trainer.train()
    trainer.test()

    # model.eval()
    # source_embs_list = []
    # target_embs_list = []

    # with torch.no_grad():
    #     # 1) collect source‐sequence embeddings
    #     for seqs_s, _, popular_s, *rest in train_loader_source:
    #         # seqs_s: [B, L], popular_s: [B, L]
    #         seqs_s      = seqs_s.to(args.device)
    #         popular_s   = popular_s.to(args.device).unsqueeze(-1)   # [B, L, 1]
    #         feats_s, _  = model.model_s(seqs_s)                     # [B, L, D]
    #         # feats_s, _  = model(seqs_s) # for sasrecW (cew)

    #         if args.weighted_mean:
    #             # weighted sum over time
    #             weighted_sum = (feats_s * popular_s).sum(dim=1)     # [B, D]
    #             weights_sum  = popular_s.sum(dim=1).clamp(min=1e-8) # [B, 1]
    #             mean_s       = weighted_sum / weights_sum          # [B, D]
    #         else:
    #             mean_s       = feats_s.mean(dim=1)                 # [B, D]

    #         source_embs_list.append(mean_s.cpu().numpy())

    #     # 2) collect target‐sequence embeddings
    #     for seqs_t, *_, in train_loader_target:
    #         seqs_t = seqs_t.to(args.device)
    #         feats_t, _ = model.model_t(seqs_t)
    #         # feats_t, _  = model(seqs_t)
    #         mean_t     = feats_t.mean(dim=1)                       # [B, D]
    #         target_embs_list.append(mean_t.cpu().numpy())
    # # stack
    # source_embs = np.vstack(source_embs_list)  # [N_s, D]
    # target_embs = np.vstack(target_embs_list)  # [N_t, D]

    # # upsample target
    # K = len(target_embs)
    # idx_t = np.random.choice(len(target_embs), K, replace=True)
    # target_embs = target_embs[idx_t]

    # # combine & label
    # # embs   = np.vstack([source_embs, target_embs])  
    # # labels = np.array([0]*len(source_embs) + [1]*len(target_embs))
    # embs   = np.vstack([source_embs, target_embs])  
    # labels = np.array([0]*len(source_embs) + [1]*int(K))

    # # run t-SNE
    # tsne = TSNE(n_components=2, perplexity=20, n_iter=1000)
    # proj = tsne.fit_transform(embs)


    # plt.figure(figsize=(7,7),dpi=300)
    # plt.scatter(proj[labels==0,0], proj[labels==0,1], s=5, label='source')
    # plt.scatter(proj[labels==1,0], proj[labels==1,1], s=5, label='target')
    # plt.legend(); plt.title("toy_bytime2_dacsr++_t-SNE")
    # plt.savefig(os.path.join(export_root, 'toy_bytime2_dacsr++_t-SNE.png'), dpi=300)
    # plt.show()

    # model.eval()
    # target_embs_list = []

    # with torch.no_grad():
    #     # collect only target embeddings
    #     for seqs_t, *rest in train_loader_target:
    #         seqs_t = seqs_t.to(args.device)
    #         feats_t, _ = model.model_t(seqs_t)            # [B, L, D]
    #         mean_t = feats_t.mean(dim=1)                       # [B, D]
    #         target_embs_list.append(mean_t.cpu().numpy())

    # # stack into [N_t, D]
    # target_embs = np.vstack(target_embs_list)

    # # run t-SNE on targets only
    # tsne = TSNE(n_components=2, perplexity=100, n_iter=1000)
    # proj_t = tsne.fit_transform(target_embs)

    # # plot
    # plt.figure(figsize=(7,7), dpi=300)
    # plt.scatter(proj_t[:, 0], proj_t[:, 1], s=5, color='orange')
    # plt.title("toy_bytime2_targetonly_dacsr++_t-SNE")
    # plt.savefig(os.path.join(export_root, 'toy_bytime2_targetonly_dacsr++_t-SNE.png'), dpi=300)
    # plt.show()


def construct_folder_save(args,val_best_score,best_parameter_before):
    'save and find the model with best validation accuracy'
    experiment_path = get_name_of_experiment_path(args.experiment_dir,
                                                  args.experiment_description,args)
    filepath_name =  os.path.join(experiment_path,'models')
    with open(os.path.join(filepath_name,"val_best_metric.json"), "r") as read_file:
        val_score = json.load(read_file)
    if val_best_score < val_score:
        val_best_score =  val_score
        best_parameter = [args.mask_prob,args.beta_t,args.beta_st,args.alpha_sst,args.alpha]
        print(best_parameter)
    else:
        best_parameter = best_parameter_before
    return best_parameter,val_best_score

def str2bool(s):
    if s not in {'false', 'true'}:
        raise ValueError('Not a valid boolean string')
    return s == 'true'

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='RecPlay')
################
# Dataset
################
    dataset_default ='toy'
    parser.add_argument('--itemshift', type=int, default=0, help='keep items in target not in source') # 0 indicate target sid all in source, those not in source are removed.
    parser.add_argument('--dataset_code', type=str, default=dataset_default, choices=['electronics','ratings_Movies_and_TV'])
    sample_type_default = 'one'
    parser.add_argument('--sample_type', type=str, default=sample_type_default,choices=['one','random','multiple'], help='Only keep users with more than min_uc ratings')
    parser.add_argument('--min_uc', type=int, default=4, help='Only keep users with more than min_uc ratings')
    parser.add_argument('--min_sc', type=int, default=0, help='Only keep items with more than min_sc ratings')
    split_default = 'random_in_target'
    parser.add_argument('--split', type=str, default=split_default, help='How to split the datasets')
    parser.add_argument('--user_split', type=str, default='by_random',choices=['by_timestamp','by_random'], help='split the users into target and source')

    parser.add_argument('--dataset_split_seed', type=int, default=98765)
    parser.add_argument('--max_target_len', type=int, default=20, help='max cold start lens for users')

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
    parser.add_argument('--trainer_code', type=str, default='CE_trainer')#, choices=TRAINERS.keys()
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
    parser.add_argument('--early_stop_num', type=int, default=20, help='Stop learning is score doesnot improve for certain num of epochs')
# logger #
    parser.add_argument('--log_period_as_iter', type=int, default=12800)
# evaluation #
    parser.add_argument('--metric_ks', nargs='+', type=int, default=[1,5, 10], help='ks for Metric@k')
    parser.add_argument('--best_metric', type=str, default='NDCG@10', help='Metric for determining the best model')
############### 
# Parameters for loss
#################
    alpha_default = 0
    beta_t_default= 0.04
    beta_s_default= 1
    beta_st_default = 0
    alpha_sst_default = 0
    parser.add_argument('--alpha', type=float, default=alpha_default, help='regularization parameter for alignment between source and sampled source')
    parser.add_argument('--alpha_sst', type=float, default=alpha_sst_default,help='regularization parameter for loss_coral for sampled source and target')
    parser.add_argument('--beta_t', type=float, default=beta_t_default, help='regularization parameter for loss_target')
    parser.add_argument('--beta_s', type=float, default=beta_s_default, help='regularization parameter for loss_source')   
    parser.add_argument('--beta_st', type=float, default=beta_st_default, help='regularization parameter for loss_source_sampled')     
################
# Model initialization will be reset in templates accordding to the args.template
################
    parser.add_argument('--model_code', type=str, default='SASRecT',choices=['SASRecT','SASRecT_1','SASRecW'])#, choices=MODELS.keys()
    parser.add_argument('--model_init_seed', type=int, default=0)
    regularization_default = 'Contrastive'
    mean_default = 'true'
    parser.add_argument('--context_is_mean', type=str, default=mean_default,choices=['false', 'true'],help='this is only required by Contrastive and Cosine regularizations')
    parser.add_argument('--weighted_mean', type=bool, default=True, help='calculate the mean weighted by popularity or not')
    temprature_default = 0.1
    KL_default = 0 
    parser.add_argument('--KL', type=int, default=KL_default)
    parser.add_argument("--static_weight",type=float,default=0.8,help="Weight on the static (SimCLR‐style) loss term")
    parser.add_argument('--warmup_epochs',type=int,default=50,help='Number of epochs over which static_weight decays from 1.0 to 0.0')

    parser.add_argument('--temprature', type=float, default=temprature_default,help='temparature parameter for contrastive loss')
    parser.add_argument('--regularization', type=str, default=regularization_default,choices=['KL', 'Cosine','Coral','MMD','Contrastive'])
    mask_prob_default = 0.5
    parser.add_argument('--mask_prob', type=float, default=mask_prob_default,help='sampling probablity of source data')
# Transformer#
# variable parameters that will be searched for optimal for sequence model of Transformer#
   
    
    parser.add_argument('--max_len_s', type=int, default=20, help='Length of sequence for Transformer')
    parser.add_argument('--max_len', type=int, default=20, help='Length of sequence for Transformer')
    parser.add_argument('--num_items', type=int, default=None, help='Number of total items')
    parser.add_argument('--hidden_units', type=int, default=64, help='Size of hidden vectors (d_model)')
    #'density_ratio'
    parser.add_argument('--wt_sampling', type=str, default='pop_ratio', help='Size of hidden vectors (d_model)')
    parser.add_argument('--domain_similarity', type=str, default='jaccard100', help='Size of hidden vectors (d_model)')
    
    parser.add_argument('--num_blocks', type=int, default=2, help='Number of transformer layers')
    parser.add_argument('--num_heads', type=int, default=1, help='Number of heads for multi-attention')
    parser.add_argument('--dropout_rate', type=float, default=0.3, help='Dropout probability to use throughout the model')
    parser.add_argument('--dropout_rate_emb', type=float, default=0.5, help='Dropout probability to use on the embedding')
    parser.add_argument('--dropout_rate_emb_t', type=float, default=0.5, help='Dropout probability to use on the embedding')
#################
# Experimentself.beta_t = args.beta_t
################
    parser.add_argument('--experiment_dir', type=str, default='experiments_itemshift_dataset%s_sample%s_KL%s_regularization%s_temp%s'%(dataset_default,sample_type_default,KL_default,regularization_default,temprature_default))
    parser.add_argument('--experiment_description', type=str, default='test_betas%s_betat%s_betast%s_betasst%s_alpha%s_mask_prop%s'%(alpha_default,mask_prob_default,beta_s_default,beta_t_default,beta_st_default,alpha_sst_default))
    args = parser.parse_args()
    if args.which_trainer == 'Source-only':
        print('source-only')
        args.mask_prob = 1
        args.beta_t = 0
        args.beta_s = 0
        args.beta_st = 1
    elif args.which_trainer == 'Target-only':
        print('target-only')
        args.mask_prob = 0.5
        args.beta_t = 1
        args.beta_s = 0
        args.beta_st = 0
        
    test_negative_sampling_seed_list = [98765]#, 987651,987652,987653,987654,987655,987656,987657,987658,987659]
    for ne_seed in test_negative_sampling_seed_list:
        print(ne_seed)
        args.test_negative_sampling_seed = ne_seed
        ################m
        mask_list = [0.5]#[0.5,0.6,0.7,0.8]
        alpha_sst_list = [0.04]#[0,0.01,0.04,0.1]
        beta_t_list = [0.01]#[0, 0.01, 0.03, 0.05]#[0.04, 0.1]
        alpha_list = [0.2]#, 0.0001, 0.0002, 0.0005,  
        beta_st_list = [0]#[0,0.01,0.04]#[0,0.04]  
        val_best_before = 0
        best_parameter_before = [0,0,0,0,0]
        for beta_t in beta_t_list:
            args.beta_t = beta_t
            print('beta_t:', beta_t)
            for beta_st in beta_st_list:
                args.beta_st = beta_st
                print('beta_st:',beta_st)
                for alpha_sst in alpha_sst_list:
                    args.alpha_sst = alpha_sst
                    print('alpha_sst:',args.alpha_sst)
                    args.experiment_dir = 'WM%s_itemshift%s_usersplit%s_maxlen%s_dataset%s_sample%s_split%s_temp%s'%(args.weighted_mean,args.itemshift, args.user_split, args.max_target_len,dataset_default,sample_type_default,args.split,temprature_default)

                    for alpha in alpha_list:
                        args.alpha = alpha 
                        
                        print('alpha:',alpha)
                        for mask in mask_list:
                            args.mask_prob = mask
                            args.experiment_description='test_mask%s_betas%s_betat%s_betast%s_betasst%s_alpha%s_negseed%s'%(mask,beta_s_default,beta_t,beta_st,alpha_sst,alpha,args.test_negative_sampling_seed)
                            train()
                            best,val_best_score = construct_folder_save(args,val_best_before,best_parameter_before)
                            best_parameter_before = best
                            val_best_before = val_best_score
                            #[args.mask_prob,args.beta_t,args.beta_st,args.alpha_sst,args.alpha]
                            
        args.experiment_description='test_mask%s_betas%s_betat%s_betast%s_betasst%s_alpha%s_negseed%s'%(best[0],beta_s_default,best[1],best[2],best[3],best[4],args.test_negative_sampling_seed)                
        filepath_name_test= os.path.join(args.experiment_dir, (args.experiment_description  
                                +"_"+ 'dataset' + args.dataset_code 
                                + "_"+ 'model' + args.model_code
                                + "_" + 'learning_rate'+str(args.lr)
                                + "_"+ 'max_len'+str(args.max_len)
                                + "_" +'num_blocks' + str(args.num_blocks) 
                                + "_"+ 'emb_dim' +str(args.hidden_units)
                                + "_" + 'weight_decay'+str(args.weight_decay)
                                +"_"+'drop_rate'+str(args.dropout_rate)
                                +"_"+'drop_rate_emb'+str(args.dropout_rate_emb)
                                +"_"+'num_heads'+str(args.num_heads)
                                ), 'logs')
        with open(os.path.join(filepath_name_test,"test_metrics.json"), "r") as read_file:
            test_best_score = json.load(read_file)
        #-----------------
        # print out the  best validation score, the test score and the searched optimal parameterset
        #-----------------
        print(' ')
        print('--------------------final results shown-------------------------------------')
        print('negative seed:',args.test_negative_sampling_seed)
        print('best model val score:',val_best_score)
        print('best model test score:',test_best_score)
        
        print('best model parameters:',best) 