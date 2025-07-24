from config import *

import json
import os
import pprint as pp
import random
from datetime import date
from pathlib import Path

import numpy as np
import torch
import torch.backends.cudnn as cudnn
from torch import optim as optim


def setup_train(args):
    set_up_gpu(args)

    export_root = create_experiment_export_folder(args)
    export_experiments_config_as_json(args, export_root)

    #pp.pprint({k: v for k, v in vars(args).items() if v is not None}, width=1)
    return export_root


def create_experiment_export_folder(args):
    experiment_dir, experiment_description = args.experiment_dir, args.experiment_description
    if not os.path.exists(experiment_dir):
        os.mkdir(experiment_dir)
    experiment_path = get_name_of_experiment_path(experiment_dir, experiment_description,args)
    if not os.path.exists(experiment_path):
        os.mkdir(experiment_path)
        print('Folder created: ' + os.path.abspath(experiment_path))
    else:
        print('Folder existed and will be rewritten')
    return experiment_path


def get_name_of_experiment_path(experiment_dir, experiment_description, args):
    #experiment_path = os.path.join(experiment_dir, (experiment_description + "_" + str(date.today())))
    experiment_path = os.path.join(experiment_dir, (experiment_description 
                              +"_"+ 'dataset' + args.dataset_code 
                              + "_"+ 'model' + args.model_code
                               + "_" + 'learning_rate'+str(args.lr)
                              + "_"+ 'max_len'+str(args.max_len)
                              + "_" +'num_blocks' + str(args.num_blocks) 
                              + "_"+ 'emb_dim' +str(args.hidden_units)
                              + "_" + 'weight_decay'+str(args.weight_decay)
                              +"_"+'drop_rate'+str(args.dropout_rate)
                              +"_"+'drop_rate_emb'+str(args.dropout_rate_emb)
                              +"_"+'num_heads'+str(args.num_heads)))
#    idx = _get_experiment_index(experiment_path)
#    experiment_path = experiment_path + "_" + str(idx)
    return experiment_path


#def _get_experiment_index(experiment_path):
#    idx = 0
#    while os.path.exists(experiment_path + "_" + str(idx)):
#        idx += 1
#    return idx


def load_weights(model, path):
    pass


def save_test_result(export_root, result):
    filepath = Path(export_root).joinpath('test_result.txt')
    with filepath.open('w') as f:
        json.dump(result, f, indent=2)


def export_experiments_config_as_json(args, experiment_path):
    with open(os.path.join(experiment_path, 'config.json'), 'w') as outfile:
        json.dump(vars(args), outfile, indent=2)


def fix_random_seed_as(random_seed):
    random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    np.random.seed(random_seed)
    cudnn.deterministic = True
    cudnn.benchmark = False


def set_up_gpu(args):
    os.environ['CUDA_VISIBLE_DEVICES'] = args.device_idx
    args.num_gpu = len(args.device_idx.split(","))


def load_pretrained_weights(model, path):
    chk_dict = torch.load(os.path.abspath(path))
    model_state_dict = chk_dict[STATE_DICT_KEY] if STATE_DICT_KEY in chk_dict else chk_dict['state_dict']
    model.load_state_dict(model_state_dict)


def setup_to_resume(args, model, optimizer):
    chk_dict = torch.load(os.path.join(os.path.abspath(args.resume_training), 'models/checkpoint-recent.pth'))
    model.load_state_dict(chk_dict[STATE_DICT_KEY])
    optimizer.load_state_dict(chk_dict[OPTIMIZER_STATE_DICT_KEY])


def create_optimizer(model, args):
    if args.optimizer == 'Adam':
        return optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    return optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, momentum=args.momentum)


class AverageMeterSet(object):
    def __init__(self, meters=None):
        self.meters = meters if meters else {}

    def __getitem__(self, key):
        if key not in self.meters:
            meter = AverageMeter()
            meter.update(0)
            return meter
        return self.meters[key]

    def update(self, name, value, n=1):
        if name not in self.meters:
            self.meters[name] = AverageMeter()
        self.meters[name].update(value, n)

    def reset(self):
        for meter in self.meters.values():
            meter.reset()

    def values(self, format_string='{}'):
        return {format_string.format(name): meter.val for name, meter in self.meters.items()}

    def averages(self, format_string='{}'):
        return {format_string.format(name): meter.avg for name, meter in self.meters.items()}

    def sums(self, format_string='{}'):
        return {format_string.format(name): meter.sum for name, meter in self.meters.items()}

    def counts(self, format_string='{}'):
        return {format_string.format(name): meter.count for name, meter in self.meters.items()}


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val
        self.count += n
        self.avg = self.sum / self.count

    def __format__(self, format):
        return "{self.val:{format}} ({self.avg:{format}})".format(self=self, format=format)
