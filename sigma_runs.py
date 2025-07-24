import os
import json
import argparse
import torch

from models import model_factory
from dataloaders import dataloader_factory
from trainers import trainer_factory
from utils import setup_train, get_name_of_experiment_path


def train(args):
    # create experiment folder
    export_root = setup_train(args)

    # load DACSR-style dataloaders (we only use the combined domain loader)
    train_loader_source, train_loader_target, train_loader_combine, val_loader, test_loader = dataloader_factory(args)
    args.num_items = train_loader_combine.dataset.num_items
    print(f"num_items after dataloader: {args.num_items}")

    # instantiate SIGMA via your model_factory (args.model_code must be 'sigma')
    print("args.num_items:", args.num_items)
    print("args.hidden_units:", getattr(args, 'hidden_units', None))

    model = model_factory(args)

    # build trainer (CEW_trainer should handle SIGMA)
    trainer = trainer_factory(
        args,
        model,
        train_loader_source,
        train_loader_target,
        train_loader_combine,
        val_loader,
        test_loader,
        export_root,
    )

    # train and evaluate
    trainer.train()
    trainer.test()

    # # load and print best validation & test metrics 
    # exp_path = get_name_of_experiment_path(args.experiment_dir, args.experiment_description, args)
    # models_dir = os.path.join(exp_path, 'models')

    # # val_best_metric.json is written by the trainer
    # with open(os.path.join(models_dir, 'val_best_metric.json'), 'r') as vf:
    #     val_best = json.load(vf)
    # # test_metrics.json is written by the trainer
    # with open(os.path.join(models_dir, 'test_metrics.json'), 'r') as tf:
    #     test_best = json.load(tf)

    print('-------------------- Final Results --------------------')
    print('Dataset:', args.dataset_code)
    print('SIGMA hyperparameters:')
    print(f"  hidden_units={args.hidden_units}, num_layers={args.num_layers}, dropout_rate={args.dropout_rate}")
    print(f"  loss_type={args.loss_type}, d_state={args.d_state}, d_conv={args.d_conv}, expand={args.expand}")
    # print(f"Best validation metrics: {val_best}")
    # print(f"Test metrics:           {test_best}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train SIGMA with full compatibility')

    ################
    # Dataset
    ################
    parser.add_argument('--itemshift', type=int, default=0)
    # parser.add_argument('--dataset_code', type=str, default='kindle')
    parser.add_argument('--dataset_code', type=str, default='electronics')
    parser.add_argument('--sample_type', type=str, default='one', choices=['one','random','multiple'])
    parser.add_argument('--min_uc', type=int, default=4)
    parser.add_argument('--min_sc', type=int, default=0)
    parser.add_argument('--split', type=str, default='random_in_target')
    parser.add_argument('--user_split', type=str, default='by_random', choices=['by_timestamp','by_random'])
    parser.add_argument('--dataset_split_seed', type=int, default=98765)
    parser.add_argument('--max_target_len', type=int, default=4)


    ################
    # Dataloader
    ################
    parser.add_argument('--dataloader_code', type=str, default='SASRec')
    parser.add_argument('--dataloader_random_seed', type=float, default=0.0)
    parser.add_argument('--train_source_batch_size', type=int, default=512)
    parser.add_argument('--train_target_batch_size', type=int, default=512)
    parser.add_argument('--train_combine_batch_size', type=int, default=512)
    parser.add_argument('--val_batch_size', type=int, default=512)
    parser.add_argument('--test_batch_size', type=int, default=512)

    ################
    # NegativeSampler
    ################
    parser.add_argument('--test_negative_sampler_code', type=str, default='random', choices=['random'])
    parser.add_argument('--test_negative_sample_size', type=int, default=100)
    parser.add_argument('--test_negative_sampling_seed', type=int, default=98765)

    ################
    # Model and Trainer Selection
    ################
    parser.add_argument('--model_code', type=str, default='SIGMA', choices=['SIGMA'])
    parser.add_argument('--trainer_code', type=str, default='CEW_trainer')

    ################
    # SIGMA architecture hyperparameters
    ################
    parser.add_argument('--hidden_units', type=int, default=64)
    parser.add_argument('--num_layers', type=int, default=1)
    parser.add_argument('--dropout_rate', type=float, default=0.2)
    parser.add_argument('--loss_type', type=str, default='CE', choices=['CE','BPR'])
    parser.add_argument('--d_state', type=int, default=32)
    parser.add_argument('--d_conv', type=int, default=4)
    parser.add_argument('--expand', type=int, default=2)
    parser.add_argument('--max_len', type=int, default=50)
    parser.add_argument('--num_blocks', type=int, default=1)

    ################
    # Training Hyperparameters
    ################
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--num_gpu', type=int, default=1)
    parser.add_argument('--device_idx', type=str, default='0')
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--weight_decay', type=float, default=0.0)
    parser.add_argument('--momentum', type=float, default=None)
    parser.add_argument('--num_epochs', type=int, default=200)
    parser.add_argument('--early_stop_num', type=int, default=20)
    parser.add_argument('--log_period_as_iter', type=int, default=12800)
    parser.add_argument('--optimizer', type=str, default='Adam', choices=['Adam'])


    ################
    # Evaluation
    ################
    parser.add_argument('--metric_ks', nargs='+', type=int, default=[1,5,10])
    parser.add_argument('--best_metric', type=str, default='NDCG@10')

    ################
    # Experiment Logging
    ################
    parser.add_argument('--experiment_dir', type=str, default='experiments_sigma')
    parser.add_argument('--experiment_description', type=str, default='sigma_beauty_run')

    ################
    # Optional for domain adaptation (only if needed)
    ################
    parser.add_argument('--mask_prob', type=float, default=0.5)
    parser.add_argument('--max_len_s', type=int, default=20)
    parser.add_argument('--regularization', type=str, default='Contrastive', choices=['KL', 'Cosine','Coral','MMD','Contrastive'])
    parser.add_argument('--context_is_mean', type=str, default='true', choices=['true', 'false'])
    parser.add_argument('--weighted_mean', type=bool, default=False)
    parser.add_argument('--temprature', type=float, default=0.1)
    parser.add_argument('--KL', type=int, default=0)
    parser.add_argument('--wt_sampling', type=str, default='pop_ratio')
    parser.add_argument('--domain_similarity', type=str, default='jaccard100')

    parser.add_argument('--alpha', type=float, default=0)
    parser.add_argument('--alpha_sst', type=float, default=0)
    parser.add_argument('--beta_t', type=float, default=0.04)
    parser.add_argument('--beta_s', type=float, default=1)
    parser.add_argument('--beta_st', type=float, default=0)
    parser.add_argument('--num_heads', type=int, default=1, help='Number of heads for multi-attention')
    parser.add_argument('--dropout_rate_emb', type=float, default=0.2, help='Dropout probability to use on the embedding')

    parser.add_argument('--which_trainer', type=str, default='DASR', choices=['DASR','Source-only','Target-only'])

    args = parser.parse_args()
    train(args)
