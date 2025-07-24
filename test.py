import os
import json
import argparse
import optuna

import torch

from models import model_factory
from dataloaders import dataloader_factory
from trainers import trainer_factory
from utils import *
from datasets import DATASETS
from dataloaders import DATALOADERS
from models import MODELS
from trainers import TRAINERS


def train(args):
    """
    Runs one training+testing cycle with the given args.
    It assumes that args.experiment_dir and args.experiment_description
    have already been set appropriately.
    """
    export_root = setup_train(args)
    train_loader_source, train_loader_target, train_loader_combine, val_loader, test_loader = dataloader_factory(args)
    model = model_factory(args)
    trainer = trainer_factory(
        args,
        model,
        train_loader_source,
        train_loader_target,
        train_loader_combine,
        val_loader,
        test_loader,
        export_root
    )
    trainer.train()
    trainer.test()


def str2bool(s):
    if s not in {'false', 'true'}:
        raise ValueError('Not a valid boolean string')
    return s == 'true'


def construct_experiment_description(mask, beta_s, beta_t, beta_st, alpha_sst, alpha, neg_seed):
    """
    Replicates the naming convention used in the original grid search for the experiment_description.
    """
    return 'test_mask%s_betas%s_betat%s_betast%s_betasst%s_alpha%s_negseed%s' % (
        mask,
        beta_s,
        beta_t,
        beta_st,
        alpha_sst,
        alpha,
        neg_seed
    )


def construct_experiment_dir(weighted_mean, itemshift, user_split, max_target_len,
                             dataset_default, sample_type_default, split, temprature_default):
    """
    Replicates the naming convention used in the original grid search for experiment_dir.
    """
    return 'WM%s_itemshift%s_usersplit%s_maxlen%s_dataset%s_sample%s_split%s_temp%s' % (
        weighted_mean,
        itemshift,
        user_split,
        max_target_len,
        dataset_default,
        sample_type_default,
        split,
        temprature_default
    )


def objective(trial, base_args, search_space):
    """
    Optuna objective function. It:
    1. Suggests hyperparameters from the given search_space.
    2. Updates base_args with these hyperparameters.
    3. Constructs experiment_dir and experiment_description.
    4. Calls train().
    5. Reads val_best_metric.json to get the validation score.
    6. Returns that validation score to Optuna for maximization.
    """
    # Copy base arguments so we don't overwrite them
    args = argparse.Namespace(**vars(base_args))

    # Suggest hyperparameters
    mask = trial.suggest_categorical('mask_prob', search_space['mask_list'])
    alpha_sst = trial.suggest_categorical('alpha_sst', search_space['alpha_sst_list'])
    beta_t = trial.suggest_categorical('beta_t', search_space['beta_t_list'])
    alpha = trial.suggest_categorical('alpha', search_space['alpha_list'])
    beta_st = trial.suggest_categorical('beta_st', search_space['beta_st_list'])
    warmup_epochs   = trial.suggest_categorical('warmup_epochs', search_space['warmup_epochs_list'])

    # Set these in args
    args.mask_prob = mask
    args.alpha_sst = alpha_sst
    args.beta_t = beta_t
    args.alpha = alpha
    args.beta_st = beta_st
    args.warmup_epochs = warmup_epochs

    # We keep beta_s fixed at its default from base_args
    beta_s_default = base_args.beta_s

    # Keep negative seed fixed as in base_args
    neg_seed = base_args.test_negative_sampling_seed

    # Recompute experiment_dir and experiment_description
    args.experiment_dir = construct_experiment_dir(
        args.weighted_mean,
        args.itemshift,
        args.user_split,
        args.max_target_len,
        args.dataset_code,
        args.sample_type,
        args.split,
        args.temprature
    )
    args.experiment_description = construct_experiment_description(
        mask,
        beta_s_default,
        beta_t,
        beta_st,
        alpha_sst,
        alpha,
        neg_seed
    )

    # Run training + testing for this trial
    train(args)

    # After training, load the validation score from val_best_metric.json
    experiment_path = get_name_of_experiment_path(
        args.experiment_dir,
        args.experiment_description,
        args
    )
    val_json_path = os.path.join(experiment_path, 'models', 'val_best_metric.json')

    if not os.path.isfile(val_json_path):
        # If something went wrong and the file doesn't exist, return a very low score
        return -float('inf')

    with open(val_json_path, "r") as read_file:
        val_score = json.load(read_file)

    # Optuna tries to maximize the returned value
    return float(val_score)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='RecPlay with Optuna hyperparameter tuning')

    ################
    # Dataset
    ################
    dataset_default = 'sport'
    parser.add_argument('--itemshift', type=int, default=0,
                        help='keep items in target not in source (0 indicates target SOB in source; others removed)')
    parser.add_argument('--dataset_code', type=str, default=dataset_default,
                        choices=['electronics', 'ratings_Movies_and_TV'])
    sample_type_default = 'one'
    parser.add_argument('--sample_type', type=str, default=sample_type_default,
                        choices=['one', 'random', 'multiple'],
                        help='Only keep users with more than min_uc ratings')
    parser.add_argument('--min_uc', type=int, default=4,
                        help='Only keep users with more than min_uc ratings')
    parser.add_argument('--min_sc', type=int, default=0,
                        help='Only keep items with more than min_sc ratings')
    split_default = 'random_in_target'
    parser.add_argument('--split', type=str, default=split_default,
                        help='How to split the datasets')
    parser.add_argument('--user_split', type=str, default='by_timestamp',
                        choices=['by_timestamp', 'by_random'],
                        help='split the users into target and source')
    parser.add_argument('--dataset_split_seed', type=int, default=98765)
    parser.add_argument('--max_target_len', type=int, default=4,
                        help='max cold start length for users')

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
    parser.add_argument('--test_negative_sampler_code', type=str, default='random',
                        choices=['random'],
                        help='Method to sample negative items for testing')
    parser.add_argument('--test_negative_sample_size', type=int, default=100)
    parser.add_argument('--test_negative_sampling_seed', type=int, default=98765)

    ################
    # Trainer initialization (will be reset according to args.which_trainer)
    ################
    parser.add_argument('--which_trainer', type=str, default='DASR',
                        choices=['DASR', 'Source-only', 'Target-only'])
    parser.add_argument('--trainer_code', type=str, default='CE_trainer')

    # device #
    parser.add_argument('--device', type=str, default='cuda', choices=['cpu', 'cuda'])
    parser.add_argument('--num_gpu', type=int, default=1)
    parser.add_argument('--device_idx', type=str, default='0')

    # optimizer #
    parser.add_argument('--optimizer', type=str, default='Adam', choices=['Adam'])
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0,
                        help='l2 regularization')
    parser.add_argument('--momentum', type=float, default=None,
                        help='SGD momentum')

    # max epochs #
    parser.add_argument('--num_epochs', type=int, default=200,
                        help='Number of epochs for training')

    # early stop num #
    parser.add_argument('--early_stop_num', type=int, default=20,
                        help='Stop if score does not improve for this many epochs')

    # logger #
    parser.add_argument('--log_period_as_iter', type=int, default=12800)

    # evaluation #
    parser.add_argument('--metric_ks', nargs='+', type=int, default=[1, 5, 10],
                        help='ks for Metric@k')
    parser.add_argument('--best_metric', type=str, default='NDCG@10',
                        help='Metric for determining the best model')

    ###############
    # Parameters for loss
    #################
    alpha_default = 0
    beta_t_default = 0.04
    beta_s_default = 1
    beta_st_default = 0
    alpha_sst_default = 0
    parser.add_argument('--alpha', type=float, default=alpha_default,
                        help='Regularization parameter for alignment between source and sampled source')
    parser.add_argument('--alpha_sst', type=float, default=alpha_sst_default,
                        help='Regularization parameter for loss_coral for sampled source and target')
    parser.add_argument('--beta_t', type=float, default=beta_t_default,
                        help='Regularization parameter for loss_target')
    parser.add_argument('--beta_s', type=float, default=beta_s_default,
                        help='Regularization parameter for loss_source')
    parser.add_argument('--beta_st', type=float, default=beta_st_default,
                        help='Regularization parameter for loss_source_sampled')

    parser.add_argument('--mask_prob', type=float, default=0.5,
                        help='Sampling probability of source data')

    ################
    # Model initialization (reset according to args.model_code)
    ################
    parser.add_argument('--model_code', type=str, default='SASRecT',
                        choices=['SASRecT', 'SASRecT_1', 'SASRecW', 'SIGMA'])
    parser.add_argument('--model_init_seed', type=int, default=0)
    regularization_default = 'Contrastive'
    mean_default = 'true'
    parser.add_argument('--context_is_mean', type=str, default=mean_default,
                        choices=['false', 'true'],
                        help='Only required by Contrastive and Cosine regularizations')
    parser.add_argument('--weighted_mean', type=bool, default=True,
                        help='Calculate the mean weighted by popularity or not')
    temprature_default = 0.1
    KL_default = 0
    parser.add_argument('--KL', type=int, default=KL_default)
    parser.add_argument('--temprature', type=float, default=temprature_default,
                        help='Temperature parameter for contrastive loss')
    parser.add_argument("--static_weight",type=float,default=0.7,help="Weight on the static (SimCLR‐style) loss term")
    parser.add_argument('--warmup_epochs',type=int,default=30,help='Number of epochs over which static_weight decays from 1.0 to 0.0')

    parser.add_argument('--regularization', type=str, default=regularization_default,
                        choices=['KL', 'Cosine', 'Coral', 'MMD', 'Contrastive'])

    mask_prob_default = 0.5

    # Transformer #
    parser.add_argument('--max_len_s', type=int, default=20,
                        help='Length of sequence for Transformer (source)')
    parser.add_argument('--max_len', type=int, default=20,
                        help='Length of sequence for Transformer (target)')
    parser.add_argument('--num_items', type=int, default=None,
                        help='Number of total items')
    parser.add_argument('--hidden_units', type=int, default=64,
                        help='Size of hidden vectors (d_model)')
    parser.add_argument('--wt_sampling', type=str, default='pop_ratio',
                        help='Sampling strategy (pop_ratio, etc.)')
    parser.add_argument('--domain_similarity', type=str, default='jaccard100',
                        help='Domain similarity measure (e.g., jaccard100)')
    parser.add_argument('--num_blocks', type=int, default=2,
                        help='Number of transformer layers')
    parser.add_argument('--num_heads', type=int, default=1,
                        help='Number of heads for multi-attention')
    parser.add_argument('--dropout_rate', type=float, default=0.3,
                        help='Dropout probability throughout the model')
    parser.add_argument('--dropout_rate_emb', type=float, default=0.5,
                        help='Dropout probability on the embedding (source)')
    parser.add_argument('--dropout_rate_emb_t', type=float, default=0.5,
                        help='Dropout probability on the embedding (target)')

    #################
    # Experiments
    ################
    parser.add_argument(
        '--experiment_dir',
        type=str,
        default='experiments_itemshift_dataset%s_sample%s_KL%s_regularization%s_temp%s' % (
            dataset_default, sample_type_default, KL_default,
            regularization_default, temprature_default
        )
    )
    parser.add_argument(
        '--experiment_description',
        type=str,
        default='test_betas%s_betat%s_betast%s_betasst%s_alpha%s_mask_prop%s' % (
            alpha_default, mask_prob_default, beta_s_default,
            beta_t_default, beta_st_default, alpha_sst_default
        )
    )

    ####################
    # Optuna-specific arguments
    ####################
    parser.add_argument('--n_trials', type=int, default=200,
                        help='Number of Optuna trials to run')

    args = parser.parse_args()

    # Adjust args if using Source-only or Target-only trainer
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

    # Define the discrete search space (replicating the original grid)
    search_space = {
        'mask_list': [0.5, 0.6, 0.7, 0.8],
        'alpha_sst_list': [0,0.01, 0.02, 0.04, 0.07, 0.1],        
        'beta_t_list': [0, 0.01, 0.03, 0.05, 0.07, 0.1],        
        'alpha_list': [0, 0.01, 0.03, 0.05, 0.07, 0.1, 0.2],         
        'beta_st_list': [0, 0.01, 0.03, 0.05, 0.07, 0.1, 0.2],
        'warmup_epochs_list': [10,20,30,40,50],         
    }

    # Keep track of the total number of combinations in the grid
    total_combinations = (
        len(search_space['mask_list']) *
        len(search_space['alpha_sst_list']) *
        len(search_space['beta_t_list']) *
        len(search_space['alpha_list']) *
        len(search_space['beta_st_list']) *
        len(search_space['warmup_epochs_list'])
    )

    # If n_trials exceeds the total grid size, we cap it to avoid redundant evaluations
    n_trials = min(args.n_trials, total_combinations)

    # Create an Optuna study (maximize validation score)
    study = optuna.create_study(direction='maximize')

    # Run the optimization
    study.optimize(
        lambda trial: objective(trial, args, search_space),
        n_trials=n_trials
    )

    # After optimization, retrieve the best trial
    best_trial = study.best_trial
    best_params = best_trial.params
    best_val_score = best_trial.value

    # Print out best validation score and best hyperparameters
    print(' ')
    print('-------------------- Best Validation Results --------------------')
    print(f'Best validation score: {best_val_score}')
    print(f'Best hyperparameters: {best_params}')

    # Now, re-run train/test (or locate existing logs) to fetch the test score for the best params
    # We need to set args to these best params and reconstruct the directory names
    beta_s_default = args.beta_s
    neg_seed = args.test_negative_sampling_seed

    # Update args with best hyperparameters
    args.mask_prob = best_params['mask_prob']
    args.alpha_sst = best_params['alpha_sst']
    args.beta_t = best_params['beta_t']
    args.alpha = best_params['alpha']
    args.beta_st = best_params['beta_st']

    # Reconstruct experiment_dir and experiment_description exactly as in objective
    args.experiment_dir = construct_experiment_dir(
        args.weighted_mean,
        args.itemshift,
        args.user_split,
        args.max_target_len,
        args.dataset_code,
        args.sample_type,
        args.split,
        args.temprature
    )
    args.experiment_description = construct_experiment_description(
        args.mask_prob,
        beta_s_default,
        args.beta_t,
        args.beta_st,
        args.alpha_sst,
        args.alpha,
        neg_seed
    )

    # Construct the path to test_metrics.json following the original naming convention
    filepath_name_test = os.path.join(
        args.experiment_dir,
        (
            args.experiment_description
            + "_"
            + 'dataset' + args.dataset_code
            + "_"
            + 'model' + args.model_code
            + "_"
            + 'learning_rate' + str(args.lr)
            + "_"
            + 'max_len' + str(args.max_len)
            + "_"
            + 'num_blocks' + str(args.num_blocks)
            + "_"
            + 'emb_dim' + str(args.hidden_units)
            + "_"
            + 'weight_decay' + str(args.weight_decay)
            + "_"
            + 'drop_rate' + str(args.dropout_rate)
            + "_"
            + 'drop_rate_emb' + str(args.dropout_rate_emb)
            + "_"
            + 'num_heads' + str(args.num_heads)
        ),
        'logs'
    )
    test_json_path = os.path.join(filepath_name_test, "test_metrics.json")

    if os.path.isfile(test_json_path):
        with open(test_json_path, "r") as read_file:
            test_best_score = json.load(read_file)
    else:
        test_best_score = None
        print(f'Warning: Could not find test_metrics.json at {test_json_path}')

    # Print out the final test score
    print(' ')
    print('-------------------- Final Test Results --------------------')
    print(f'Best model test score: {test_best_score}')
