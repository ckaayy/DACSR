#!/usr/bin/env python
import os
import json
from argparse import Namespace

import torch

from models import model_factory
from dataloaders import dataloader_factory
from trainers import trainer_factory
from utils import setup_train, get_name_of_experiment_path

def train(args):
    export_root = setup_train(args)
    # dataloader_factory will also set args.num_items for us
    train_src, train_tgt, train_comb, val_loader, test_loader = dataloader_factory(args)
    model    = model_factory(args)
    trainer  = trainer_factory(args, model, train_src, train_tgt, train_comb, val_loader, test_loader, export_root)
    trainer.train()
    trainer.test()

if __name__ == "__main__":

    args = Namespace(
        # data & splits
        itemshift                     = 0,
        dataset_code                  = "beauty",
        sample_type                   = "one",
        min_uc                        = 4,
        min_sc                        = 0,
        split                         = "random_in_target",
        user_split                    = "by_timestamp",
        dataset_split_seed            = 98765,
        max_target_len                = 4,

        # dataloader
        dataloader_code               = "SASRec",
        dataloader_random_seed        = 0.0,
        train_target_batch_size       = 128,   # placeholder, will be overwritten
        train_source_batch_size       = 128,   # placeholder, will be overwritten
        train_combine_batch_size      = 1000,   # placeholder, will be overwritten
        val_batch_size                = 1000,   # placeholder, will be overwritten
        test_batch_size               = 500,   # placeholder, will be overwritten

        # negative sampling for eval
        test_negative_sampler_code    = "random",
        test_negative_sample_size     = 100,
        test_negative_sampling_seed   = 98765,

        # trainer
        which_trainer                 = "DASR",    # pure Transformer on target domain
        trainer_code                  = "CEW_trainer",

        # device
        device                        = "cuda",
        num_gpu                       = 1,
        device_idx                    = "0",

        # optimizer
        optimizer                     = "Adam",
        lr                            = None,  # placeholder, will be overwritten
        weight_decay                  = 0.0,
        momentum                      = None,

        # training
        num_epochs                    = 200,
        early_stop_num                = 20,
        log_period_as_iter            = 12800,

        # evaluation metrics
        metric_ks                     = [1,5,10],
        best_metric                   = "NDCG@10",

        # domain-adaptation (turned off for pure baseline)
        alpha                         = 0.0,
        alpha_sst                     = 0.0,
        beta_t                        = 1.0,
        beta_s                        = 0.0,
        beta_st                       = 0.0,

        # model choice
        model_code                    = "linrec",
        model_init_seed               = 0,

        # contrastive / Coral losses (not used here)
        context_is_mean               = "true",
        weighted_mean                 = True,
        KL                            = 0,
        temprature                    = 0.1,
        regularization                = "Contrastive",
        mask_prob                     = 0.2,

        # Transformer architecture (LinRec) hyperparams
        max_len_s                     = 50,
        max_len                       = 50,
        num_items                     = None,  # will be set by dataloader_factory
        hidden_units                  = None,  # placeholder, will be overwritten
        wt_sampling                   = "pop_ratio",
        domain_similarity             = "jaccard100",
        num_blocks                    = None,  # placeholder, will be overwritten
        num_heads                     = None,  # placeholder, will be overwritten
        dropout_rate                  = None,  # placeholder, will be overwritten
        dropout_rate_emb              = None,  # placeholder, will be overwritten
        dropout_rate_emb_t            = 0.2,   # not used by LinRec

        # RecBole-style hyperparameters
        learning_rate                 = 0.001,
        train_batch_size              = 400,
        eval_batch_size               = 400,
        train_neg_sample_args         = None,
        neg_sampling                  = None,
        mask_ratio                    = 0.2,
        hidden_size                   = 128,
        inner_size                    = 256,
        n_layers                      = 2,
        n_heads                       = 8,
        hidden_dropout_prob           = 0.3,
        attn_dropout_prob             = 0.3,
        hidden_act                    = 'gelu',
        layer_norm_eps                = 1e-12,
        initializer_range             = 0.02,
        topk                          = 10,
        metrics                       = ['Recall','MRR','NDCG'],
        valid_metric                  = 'NDCG@10',

        # where to write logs & models
        experiment_dir                = "experiments_linrec_baseline",
        experiment_description        = "linrec_baseline"
    )

    # mapping
    args.lr                       = args.learning_rate
    args.train_target_batch_size  = args.train_batch_size
    args.train_source_batch_size  = args.train_batch_size
    args.train_combine_batch_size = args.train_batch_size
    args.val_batch_size           = args.eval_batch_size
    args.test_batch_size          = args.eval_batch_size

    args.hidden_units  = args.hidden_size
    args.num_blocks    = args.n_layers
    args.num_heads     = args.n_heads
    args.dropout_rate  = args.attn_dropout_prob 
    args.dropout_rate_emb = args.hidden_dropout_prob

    # ensure experiment folder exists
    os.makedirs(args.experiment_dir, exist_ok=True)

    # run training & testing
    train(args)

    # print out final test metrics
    best_run = get_name_of_experiment_path(args.experiment_dir, args.experiment_description, args)
    with open(os.path.join(best_run, "logs", "test_metrics.json")) as f:
        results = json.load(f)
    print("=== LinRec baseline results ===")
    print(json.dumps(results, indent=2))
