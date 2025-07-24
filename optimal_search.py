
from models import model_factory
from dataloaders import dataloader_factory
from trainers import trainer_factory
from utils import *
import json
import os
import argparse
from dataloaders import DATALOADERS
from models import MODELS
from trainers import TRAINERS

def train(args):
    export_root = setup_train(args)
    # train_loader_source, train_loader_source_sample, train_loader_target, val_loader, test_loader = dataloader_factory(args)
    # model = model_factory(args)
    # trainer = trainer_factory(args, model,train_loader_source,train_loader_source_sample, train_loader_target, val_loader, test_loader, export_root)
    train_loader_source, train_loader_target,train_loader_combine, val_loader, test_loader = dataloader_factory(args)
    model = model_factory(args)
    trainer = trainer_factory(args, model,train_loader_source, train_loader_target,train_loader_combine, val_loader, test_loader, export_root)
    trainer.train()
    trainer.test()

    
def construct_folder_save(args,search_parameter,val_best_score,best_parameter_before):
    best_parameter = best_parameter_before
    filepath_name= os.path.join(args.experiment_dir, (args.experiment_description  
                              + "_"+ 'dataset' + args.dataset_code
                              +"_"+ 'model' + args.model_code
                              + "_" + 'learning_rate'+str(args.lr)
                              + "_"+ 'max_len'+str(args.max_len)
                               + "_" +'num_blocks' + str(args.num_blocks) 
                              + "_"+ 'emb_dim' +str(args.hidden_units)
                              + "_" + 'weight_decay'+str(args.weight_decay)
                              +"_"+'drop_rate'+str(args.dropout_rate)
                              +"_"+'drop_rate_emb'+str(args.dropout_rate_emb)
                              +"_"+'num_heads'+str(args.num_heads)
                              ),'models')
                    
    with open(os.path.join(filepath_name,"val_best_metric.json"), "r") as read_file:
            val_score = json.load(read_file)
    if val_best_score < val_score:
            val_best_score =  val_score
            best_parameter = search_parameter
    return best_parameter,val_best_score

def optimal_search(parser, space):
  

    args = parser.parse_args()
    #### reset some of the args accordding to some model requirements and constraints.   check the template.py for details.  

    layer_num_default = 2
    
    max_len_list = space['max_len_list']
    hidden_units_list = space['hidden_units_list']
    learning_rate_list = space['learning_rate_list']
    drop_rate_list = space['drop_rate_list']
    drop_rate_emb_list = space['drop_rate_emb_list']
    num_head_list = space['num_head_list']
    
    
    
    if len(hidden_units_list) == 1:
        hidden_units_default = hidden_units_list[0]
        print('hidden_units_default:', hidden_units_default)
    else:
        hidden_units_default = 64
    if len(learning_rate_list) == 1:
        learning_rate_default = learning_rate_list[0]
        print('learning_rate_default:', learning_rate_default)
    else:
        learning_rate_default = learning_rate_list[0]
    if len(drop_rate_list) == 1:
        dropout_rate_default = drop_rate_list[0]
        print('dropout_rate_default:', dropout_rate_default)
    else:
        dropout_rate_default = 0.1
    if len(drop_rate_emb_list) == 1:
        dropout_rate_emb_default = drop_rate_emb_list[0]
        print('dropout_rate_emb_default:', dropout_rate_emb_default)
    else:
        dropout_rate_emb_default = 0.1
        

    val_best_dlen=[]
    test_best_dlen=[]
    # optimal_parameter_sets= []
    # test_result_sets = []
    
    args.num_blocks = layer_num_default
    args.hidden_units = hidden_units_default
    args.weight_decay = 0
    
    num_heads_default = num_head_list[0]           
    args.num_heads = num_heads_default
    
    for leng in max_len_list:
        args.max_len = leng
        
        #########searching for best hidden units
        val_best_score = 0
        args.dropout_rate = dropout_rate_default
        args.dropout_rate_emb = dropout_rate_emb_default
        args.lr = learning_rate_default        
        print('searching for the best hidden units')
        best_hidden_units_before = hidden_units_default

        for hidden_units in hidden_units_list:    
            args.hidden_units = hidden_units
            
            train(args)
            best_hidden_units, val_best_score = construct_folder_save(args,args.hidden_units,val_best_score,best_hidden_units_before)
            best_hidden_units_before = best_hidden_units
        print('best hidden_units:', best_hidden_units)

        ################search best learning rate
        print('searching for the best learning rate')
        val_best_score = 0 
        best_learning_rate_before = learning_rate_default            
        for lr in learning_rate_list:
            args.lr = lr 
            print('learning rate:', lr)      
            args.hidden_units= best_hidden_units
        
            if args.lr == learning_rate_default:
                best_learning_rate, val_best_score = construct_folder_save(args,args.lr,val_best_score,best_learning_rate_before)
            else:
                train(args)
                best_learning_rate, val_best_score = construct_folder_save(args,args.lr,val_best_score,best_learning_rate_before)
            best_learning_rate_before = best_learning_rate
        print('best learning rate:', best_learning_rate)
        ################search best dropout rate
        print('searching for the best dropout rate')
        val_best_score = 0 
        best_dropout_rate_before = dropout_rate_default           
        for drop_rate in drop_rate_list:        
            args.dropout_rate = drop_rate
            args.hidden_units= best_hidden_units
            args.lr = best_learning_rate
            if args.dropout_rate == dropout_rate_default:
                best_dropout_rate, val_best_score = construct_folder_save(args,args.dropout_rate,val_best_score,best_dropout_rate_before)
            else:
                train(args)
                best_dropout_rate, val_best_score = construct_folder_save(args,args.dropout_rate,val_best_score,best_dropout_rate_before)
            best_dropout_rate_before = best_dropout_rate
        print('best dropout rate:', best_dropout_rate)
        ################search best dropout rate
        print('searching for the best embedding dropout rate')
        val_best_score = 0
        best_dropout_rate_emb_before = dropout_rate_emb_default             
        for drop_rate_emb in drop_rate_emb_list:      
            args.dropout_rate_emb = drop_rate_emb
            args.hidden_units= best_hidden_units
            args.dropout_rate = best_dropout_rate
            args.lr = best_learning_rate
            if args.dropout_rate_emb == dropout_rate_emb_default:
                best_dropout_rate_emb, val_best_score = construct_folder_save(args,args.dropout_rate_emb,val_best_score,best_dropout_rate_emb_before)
            else:
                train(args)
                best_dropout_rate_emb, val_best_score = construct_folder_save(args,args.dropout_rate_emb,val_best_score,best_dropout_rate_emb_before)
            best_dropout_rate_emb_before = best_dropout_rate_emb
        print('best dropout rate emb:', best_dropout_rate_emb)
        ################search best number heads for Multihead attention based models
        val_best_score = 0
        best_num_heads_before = num_heads_default
        for num_head in num_head_list:
            args.dropout_rate_emb = best_dropout_rate_emb
            args.dropout_rate = best_dropout_rate
            args.hidden_units = best_hidden_units
            args.lr = best_learning_rate

            args.num_heads = num_head
           
            if args.num_heads == num_heads_default:    
                best_num_heads, val_best_score = construct_folder_save(args,args.num_heads,val_best_score,best_num_heads_before)
            else:
                train(args)
                best_num_heads, val_best_score = construct_folder_save(args,args.num_heads,val_best_score,best_num_heads_before)
            best_num_heads_before = best_num_heads
           
        print('best num of heads:',best_num_heads)
        #############################optimal model search complete
        print('optimal model parameter searching completed!')
        val_best_dlen.append(val_best_score)
        filepath_name_test= os.path.join(args.experiment_dir, (args.experiment_description  
                              + "_"+ 'dataset' + args.dataset_code
                              +"_"+ 'model' + args.model_code
                              + "_" + 'learning_rate'+str(best_learning_rate)
                              + "_"+ 'max_len'+str(args.max_len)
                              + "_" +'num_blocks' + str(args.num_blocks) 
                              + "_"+ 'emb_dim' +str(best_hidden_units)
                              + "_" + 'weight_decay'+str(args.weight_decay)                     
                              +"_"+'drop_rate'+str(best_dropout_rate)
                              +"_"+'drop_rate_emb'+str(best_dropout_rate_emb)
                              +"_"+'num_heads'+str(best_num_heads)
                              ), 'logs')
        with open(os.path.join(filepath_name_test,"test_metrics.json"), "r") as read_file:
            test_best_score = json.load(read_file)
                                 
        test_best_dlen.append(test_best_score)

        print('best model val score:',val_best_score)
        print('best model test score:',test_best_score)
        print('best model parameters:',[best_hidden_units,best_learning_rate, best_dropout_rate, best_dropout_rate_emb, best_num_heads])

    #     optimal_parameter_sets = optimal_parameter_sets.extend([best_hidden_units,best_learning_rate, best_dropout_rate, best_dropout_rate_emb, best_num_heads])
    #     test_result_sets = test_result_sets.extend(test_best_score)
    # return optimal_parameter_sets,test_result_sets
         
    
    
