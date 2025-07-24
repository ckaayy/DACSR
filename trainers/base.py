from loggers import *
from config import STATE_DICT_KEY, OPTIMIZER_STATE_DICT_KEY
from utils import AverageMeterSet
from dataloaders import dataloader_factory
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from itertools import cycle
import json
from abc import *
from pathlib import Path
import os

class AbstractTrainer(metaclass=ABCMeta):
    #def __init__(self, args, model, train_source_loader, train_source_sample_loader,train_target_loader, val_loader, test_loader, export_root):#
    def __init__(self, args, model, train_source_loader,train_target_loader, train_combine_loader, val_loader, test_loader, export_root):#
        self.args = args
        self.which_trainer = args.which_trainer
        self.device = args.device
        self.model = model.to(self.device)
        self.early_stop_num=args.early_stop_num
        self.is_parallel = args.num_gpu > 1
        
        if self.is_parallel:
            self.model = nn.DataParallel(self.model)

        self.train_source_loader = train_source_loader
        self.train_combine_loader = train_combine_loader
        self.train_target_loader = train_target_loader
        self.val_loader = val_loader
        self.test_loader = test_loader

        self.optimizer = self._create_optimizer()

        self.num_epochs = args.num_epochs
        self.metric_ks = args.metric_ks
        self.best_metric = args.best_metric

        self.export_root = export_root
        self.writer, self.train_loggers, self.val_loggers = self._create_loggers()
        self.add_extra_loggers()
        self.logger_service = LoggerService(self.train_loggers, self.val_loggers)
        self.log_period_as_iter = args.log_period_as_iter

    @abstractmethod
    def add_extra_loggers(self):
        pass

    @abstractmethod
    def log_extra_train_info(self, log_data):
        pass

    @abstractmethod
    def log_extra_val_info(self, log_data):
        pass

    @classmethod
    @abstractmethod
    def code(cls):
        pass

    @abstractmethod
    def calculate_loss(self, batch):
        pass

    @abstractmethod
    def calculate_metrics(self, batch):
        pass

    def train(self):
        accum_iter = 0

        best_score = 0
        num_count = 0
        for epoch in range(self.num_epochs):
            accum_iter = self.train_one_epoch(epoch, accum_iter)
            average_score=self.validate(epoch, accum_iter)
            score_current = average_score[self.best_metric]
            if best_score < score_current:
               best_score =  score_current
               num_count = 0
            else:
               num_count = num_count+1 
               print(num_count)
               if num_count> self.early_stop_num:
                   break
        self.logger_service.complete({
            'state_dict': (self._create_state_dict()),
        })
        self.writer.close()

    def train_one_epoch(self, epoch, accum_iter):
        self.model.train()

        average_meter_set = AverageMeterSet()
        target_dataloader = self.train_target_loader
        source_dataloader = self.train_source_loader
        combine_dataloader = self.train_combine_loader
        #tqdm_dataloader = tqdm(self.train_target_loader)
        # total = min(len(source_dataloader),len(source_sample_dataloader), len(target_dataloader))
        # tqdm_dataloader = tqdm(zip(cycle(source_dataloader), source_sample_dataloader,target_dataloader),total=total)
        #total = min(len(source_dataloader), len(target_dataloader))
        #tqdm_dataloader = tqdm(zip(cycle(source_dataloader),target_dataloader),total=total)
        #self.optimizer.zero_grad()

        if self.args.model_code in ['SASRecW', 'SIGMA', 'linrec']:
            #print('model code:', self.args.model_code)
            tqdm_dataloader = tqdm(combine_dataloader)
            for batch_idx, combine_batch in enumerate(tqdm_dataloader):
            
                batch_size = combine_batch[0].size(0)
                combine_batch = [x.to(self.device) for x in combine_batch]
                self.optimizer.zero_grad()
                loss,loss_t,loss_s,loss_s_s,loss_c,loss_sst = self.calculate_loss(combine_batch)
                loss.backward()

                self.optimizer.step() 

                average_meter_set.update('loss', loss.item())
                if loss_s_s == 0:
                    average_meter_set.update('loss_t', loss_t)
                    average_meter_set.update('loss_s', loss_s)
                    average_meter_set.update('loss_s_s', loss_s_s)
                    average_meter_set.update('loss_c', loss_c)
                    average_meter_set.update('loss_sst', loss_sst)
                else:
                    average_meter_set.update('loss_t', loss_t.item())
                    average_meter_set.update('loss_s', loss_s.item())
                    average_meter_set.update('loss_s_s', loss_s_s.item())
                    average_meter_set.update('loss_c', loss_c.item())
                    average_meter_set.update('loss_sst', loss_sst.item())
                tqdm_dataloader.set_description(
                    'Epoch {}, loss {:.3f}, loss_t {:.3f}, loss_s {:.3f}, loss_s_s {:.3f} , loss_c {:.3f}, loss_sst {:.3f}'.format(epoch+1, average_meter_set['loss'].avg,average_meter_set['loss_t'].avg,average_meter_set['loss_s'].avg,average_meter_set['loss_s_s'].avg,average_meter_set['loss_c'].avg,average_meter_set['loss_sst'].avg))

                accum_iter += batch_size

                if self._needs_to_log(accum_iter):
                    tqdm_dataloader.set_description('Logging to Tensorboard')
                    log_data = {
                        'state_dict': (self._create_state_dict()),
                        'epoch': epoch+1,
                        'accum_iter': accum_iter,
                    }
                    log_data.update(average_meter_set.averages())
                    self.log_extra_train_info(log_data)
                    self.logger_service.log_train(log_data)
        else:
            #print('model code:', self.args.model_code)
            # tqdm_dataloader = tqdm(source_dataloader)
            # dataloader_iterator = iter(target_dataloader)
            # for batch_idx, source_batch in enumerate(tqdm_dataloader):
                
            #     try:
            #         target_batch = next(dataloader_iterator)
            #     except StopIteration:
            #         dataloader_iterator = iter(target_dataloader)
            #         target_batch = next(dataloader_iterator)

            tqdm_dataloader = tqdm(target_dataloader)
            dataloader_iterator = iter(source_dataloader)
            for batch_idx, target_batch in enumerate(tqdm_dataloader):
        
                try:
                    source_batch = next(dataloader_iterator)
                except StopIteration:
                    dataloader_iterator = iter(source_dataloader)
                    source_batch = next(dataloader_iterator)
                batch_size = target_batch[0].size(0)
                target_batch = [x.to(self.device) for x in target_batch]
                source_batch = [x.to(self.device) for x in source_batch]
                # print(source_batch[0][0,:])
                # print(source_batch[2][0,:])
                # print(target_batch[0][0,:])
                self.optimizer.zero_grad()
                loss,loss_t,loss_s,loss_s_s,loss_c,loss_sst = self.calculate_loss(source_batch, target_batch)
                 
                loss.backward()

                self.optimizer.step() 

                average_meter_set.update('loss', loss.item())
                if loss_s_s == 0:
                    average_meter_set.update('loss_t', loss_t)
                    average_meter_set.update('loss_s', loss_s)
                    average_meter_set.update('loss_s_s', loss_s_s)
                    average_meter_set.update('loss_c', loss_c)
                    average_meter_set.update('loss_sst', loss_sst)
                else:
                    average_meter_set.update('loss_t', loss_t.item())
                    average_meter_set.update('loss_s', loss_s.item())
                    average_meter_set.update('loss_s_s', loss_s_s.item())
                    average_meter_set.update('loss_c', loss_c.item())
                if loss_sst == 0:
                    average_meter_set.update('loss_sst', loss_sst)
                else:
                    average_meter_set.update('loss_sst', loss_sst.item())
                tqdm_dataloader.set_description(
                    'Epoch {}, loss {:.3f}, loss_t {:.3f}, loss_s {:.3f}, loss_s_s {:.3f} , loss_c {:.3f}, loss_sst {:.3f}'.format(epoch+1, average_meter_set['loss'].avg,average_meter_set['loss_t'].avg,average_meter_set['loss_s'].avg,average_meter_set['loss_s_s'].avg,average_meter_set['loss_c'].avg,average_meter_set['loss_sst'].avg))

                accum_iter += batch_size

                if self._needs_to_log(accum_iter):
                    tqdm_dataloader.set_description('Logging to Tensorboard')
                    log_data = {
                        'state_dict': (self._create_state_dict()),
                        'epoch': epoch+1,
                        'accum_iter': accum_iter,
                    }
                    log_data.update(average_meter_set.averages())
                    self.log_extra_train_info(log_data)
                    self.logger_service.log_train(log_data)
            #source_sample_batch = [x.to(self.device) for x in source_sample_batch]

            # if epoch in [0,1,2,3,4,5] and batch_idx==0:
            #     print(source_batch[0][0,:])
            #     print(target_batch[0][0,:])
               
            # print(source_batch[0][0,:])
            # print(source_batch[1][0,:])
            # print(source_sample_batch[0][0,:])
            # print(source_sample_batch[1][0,:])
               
            #     print(target_batch[0][1,:])
            #     print(target_batch[1][1,:])
               
            #     print(source_batch[0][1,:])
            #     print(source_batch[1][1,:])
            # loss.backward()

            # self.optimizer.step() 

            # average_meter_set.update('loss', loss.item())
            # if loss_s_s == 0:
            #     average_meter_set.update('loss_t', loss_t)
            #     average_meter_set.update('loss_s', loss_s)
            #     average_meter_set.update('loss_s_s', loss_s_s)
            #     average_meter_set.update('loss_c', loss_c)
            #     average_meter_set.update('loss_sst', loss_sst)
            # else:
            #     average_meter_set.update('loss_t', loss_t.item())
            #     average_meter_set.update('loss_s', loss_s.item())
            #     average_meter_set.update('loss_s_s', loss_s_s.item())
            #     average_meter_set.update('loss_c', loss_c.item())
            #     average_meter_set.update('loss_sst', loss_sst.item())
            # tqdm_dataloader.set_description(
            #     'Epoch {}, loss {:.3f}, loss_t {:.3f}, loss_s {:.3f}, loss_s_s {:.3f} , loss_c {:.3f}, loss_sst {:.3f}'.format(epoch+1, average_meter_set['loss'].avg,average_meter_set['loss_t'].avg,average_meter_set['loss_s'].avg,average_meter_set['loss_s_s'].avg,average_meter_set['loss_c'].avg,average_meter_set['loss_sst'].avg))

            # accum_iter += batch_size

            # if self._needs_to_log(accum_iter):
            #     tqdm_dataloader.set_description('Logging to Tensorboard')
            #     log_data = {
            #         'state_dict': (self._create_state_dict()),
            #         'epoch': epoch+1,
            #         'accum_iter': accum_iter,
            #     }
            #     log_data.update(average_meter_set.averages())
            #     self.log_extra_train_info(log_data)
            #     self.logger_service.log_train(log_data)

        return accum_iter

    def validate(self, epoch, accum_iter):
        self.model.eval()

        average_meter_set = AverageMeterSet()

        with torch.no_grad():
            tqdm_dataloader = tqdm(self.val_loader)
            for batch_idx, batch in enumerate(tqdm_dataloader):
                batch = [x.to(self.device) for x in batch]

                metrics = self.calculate_metrics(batch, self.which_trainer)

                for k, v in metrics.items():
                    average_meter_set.update(k, v)
                description_metrics = ['NDCG@%d' % k for k in self.metric_ks[:5]] +\
                                      ['Recall@%d' % k for k in self.metric_ks[:5]]+\
                                      ['MRR']
                description = 'Val: ' + ', '.join(s + ' {:.3f}' for s in description_metrics)
                description = description.replace('NDCG', 'N').replace('Recall', 'R')
                description = description.format(*(average_meter_set[k].avg for k in description_metrics))
                tqdm_dataloader.set_description(description)

            log_data = {
                'state_dict': (self._create_state_dict()),
                'epoch': epoch+1,
                'accum_iter': accum_iter,
            }
            log_data.update(average_meter_set.averages())
            self.log_extra_val_info(log_data)
            self.logger_service.log_val(log_data)
            average_score = average_meter_set.averages()
        return average_score

    def test(self):
        print('Test best model with test set!')

        best_model = torch.load(os.path.join(self.export_root, 'models', 'best_acc_model.pth')).get('model_state_dict')
        self.model.load_state_dict(best_model)
        self.model.eval()

        average_meter_set = AverageMeterSet()

        with torch.no_grad():
            tqdm_dataloader = tqdm(self.test_loader)
            for batch_idx, batch in enumerate(tqdm_dataloader):
                batch = [x.to(self.device) for x in batch]

                metrics = self.calculate_metrics(batch, self.which_trainer)

                for k, v in metrics.items():
                    average_meter_set.update(k, v)
                description_metrics = ['NDCG@%d' % k for k in self.metric_ks[:5]] +\
                                      ['Recall@%d' % k for k in self.metric_ks[:5]] +\
                                      ['MRR']
                description = 'Test: ' + ', '.join(s + ' {:.3f}' for s in description_metrics)
                description = description.replace('NDCG', 'N').replace('Recall', 'R')
                description = description.format(*(average_meter_set[k].avg for k in description_metrics))
                tqdm_dataloader.set_description(description)

            average_metrics = average_meter_set.averages()
            with open(os.path.join(self.export_root, 'logs', 'test_metrics.json'), 'w') as f:
                json.dump(average_metrics, f, indent=4)
            print(average_metrics)
            
        


    def _create_optimizer(self):
        args = self.args
        if args.optimizer.lower() == 'adam':
            return optim.Adam(self.model.parameters(), lr=args.lr, betas=(0.9, 0.999), weight_decay=args.weight_decay)#optim.Adam(self.model.parameters(), lr=args.lr, betas=(0.9, 0.98))
        #
        elif args.optimizer.lower() == 'sgd':
            return optim.SGD(self.model.parameters(), lr=args.lr, weight_decay=args.weight_decay, momentum=args.momentum)
        else:
            raise ValueError

    def _create_loggers(self):
        root = Path(self.export_root)
        writer = SummaryWriter(root.joinpath('logs'))
        model_checkpoint = root.joinpath('models')

        train_loggers = [
            MetricGraphPrinter(writer, key='epoch', graph_name='Epoch', group_name='Train'),
            MetricGraphPrinter(writer, key='loss', graph_name='Loss', group_name='Train'),
            MetricGraphPrinter(writer, key='loss_t', graph_name='Loss_t', group_name='Train'),
            MetricGraphPrinter(writer, key='loss_s', graph_name='Loss_s', group_name='Train'),
            MetricGraphPrinter(writer, key='loss_s_s', graph_name='Loss_s_s', group_name='Train'),
            MetricGraphPrinter(writer, key='loss_c', graph_name='Loss_c', group_name='Train'),
            MetricGraphPrinter(writer, key='loss_sst', graph_name='Loss_sst', group_name='Train'),
        ]

        val_loggers = []
        for k in self.metric_ks:
            val_loggers.append(
                MetricGraphPrinter(writer, key='NDCG@%d' % k, graph_name='NDCG@%d' % k, group_name='Validation'))
            val_loggers.append(
                MetricGraphPrinter(writer, key='Recall@%d' % k, graph_name='Recall@%d' % k, group_name='Validation'))
            val_loggers.append(
                MetricGraphPrinter(writer, key='MRR', graph_name='MRR', group_name='Validation'))
        val_loggers.append(RecentModelLogger(model_checkpoint))
        val_loggers.append(BestModelLogger(model_checkpoint, metric_key=self.best_metric))
        return writer, train_loggers, val_loggers

    def _create_state_dict(self):
        return {
            STATE_DICT_KEY: self.model.module.state_dict() if self.is_parallel else self.model.state_dict(),
            OPTIMIZER_STATE_DICT_KEY: self.optimizer.state_dict(),
        }

    def _needs_to_log(self, accum_iter):
        return accum_iter % self.log_period_as_iter < self.args.train_target_batch_size and accum_iter != 0


