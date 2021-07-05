from myModel import myModel, NTM, myModelStat
from myTrain import myTrain
from myTest import myTest
import myDataset
from myUtils import cp, time_since
import myConfig
import numpy as np
import random
import torch
import torch.nn.functional as F
from torch.optim import Adam
from itertools import chain
import os
import json
import time
import argparse
import logging

def main(opt):
        
    #Dataset
    start_time = time.time()
    train_loader, val_loader, test_loader, train_bow_loader, \
    valid_bow_loader, test_bow_loader, bow_dictionary, opt = myDataset.load_data(opt) 
    load_data_time = time_since(start_time)
    logging.info('Time for loading the data: %.1f' %load_data_time)
    
    #Model
    start_time = time.time()
    model = myModel(opt).to(opt.device)
    ntm_model = NTM(opt).to(opt.device)
    optimizer_fe, optimizer_ntm, optimizer_whole = init_optimizers(model, ntm_model, opt)
    myModelStat(model)
    myModelStat(ntm_model)

    #Train
    if opt.only_train_ntm:
        myTrain(model, ntm_model, optimizer_fe, optimizer_ntm, optimizer_whole, 
                train_loader, val_loader, bow_dictionary, train_bow_loader,
                valid_bow_loader, opt)          
        return 
    if opt.joint_train:
        check_pt_ntm_model_path, check_pt_model_path = myTrain(model, ntm_model, optimizer_fe, optimizer_ntm, optimizer_whole, 
                train_loader, val_loader, bow_dictionary, train_bow_loader,
                valid_bow_loader, opt) 
     
    if not opt.joint_train:
        check_pt_model_path = myTrain(model, ntm_model, optimizer_fe, optimizer_ntm, optimizer_whole, 
                train_loader, val_loader, bow_dictionary, train_bow_loader,
                valid_bow_loader, opt)  
        check_pt_ntm_model_path = None

    cp(check_pt_model_path, 'check_pt_model_path')
    cp(check_pt_ntm_model_path, 'check_pt_ntm_model_path')
    training_time = time_since(start_time)
    logging.info('Time for training: %.1f' % training_time)
    
    #Test
    start_time = time.time()
    myTest(model, ntm_model, test_loader ,bow_dictionary, test_bow_loader, check_pt_ntm_model_path, check_pt_model_path, opt) 
    test_time = time_since(start_time)
    logging.info('Time for testing: %.1f' % test_time)

def init_optimizers(model, ntm_model, opt):
    optimizer_fe = Adam(params=filter(lambda p: p.requires_grad, model.parameters()), lr=opt.learning_rate)
    optimizer_ntm = Adam(params=filter(lambda p: p.requires_grad, ntm_model.parameters()), lr=opt.learning_rate)
    whole_params = chain(model.parameters(), ntm_model.parameters())
    optimizer_whole = Adam(params=filter(lambda p: p.requires_grad, whole_params), lr=opt.learning_rate)

    return optimizer_fe, optimizer_ntm, optimizer_whole 
  
def process_opt(opt):
    if opt.seed > 0:
        torch.manual_seed(opt.seed)
        np.random.seed(opt.seed)
        random.seed(opt.seed)

    # only train ntm
    if opt.only_train_ntm:
        assert opt.ntm_warm_up_epochs > 0 and not opt.load_pretrain_ntm
        opt.exp += '.topic_num{}'.format(opt.topic_num)
        opt.exp += '.ntm_warm_up_%d' % opt.ntm_warm_up_epochs
        opt.model_path = opt.model_path % (opt.exp, opt.timemark)
        if not os.path.exists(opt.model_path):
            os.makedirs(opt.model_path)
        print("Only training the ntm for %d epochs and save it to %s" % (opt.ntm_warm_up_epochs, opt.model_path))
        return opt
        
    # joint train settings
    if opt.joint_train:
        opt.exp += '.joint_train'
        #if opt.add_two_loss:
        #    opt.exp += '.add_two_loss'
        if opt.joint_train_strategy != 'p_1_joint':
            opt.exp += '.' + opt.joint_train_strategy
            opt.p_fe_e = int(opt.joint_train_strategy.split('_')[1])
            if opt.joint_train_strategy.split('_')[-1] != 'joint':
                opt.iterate_train_ntm = True

    # adding topic settings
    if opt.use_topic_represent:
        if opt.load_pretrain_ntm:
            has_topic_num = [t for t in opt.check_pt_ntm_model_path.split('.') if 'topic_num' in t]
            if len(has_topic_num) != 0:
                assert opt.topic_num == int(has_topic_num[0].replace('topic_num', ''))

            ntm_tag = '.'.join(opt.check_pt_ntm_model_path.split('/')[-1].split('.')[:-1])
            # opt.exp += '.ntm_%s' % ntm_tag
        else:
            opt.exp += '.ntm_warm_up_%d' % opt.ntm_warm_up_epochs
    
    size_tag = ".emb{}".format(opt.emb_size) + ".dec{}".format(opt.hidden_size)
    opt.exp += '.seed{}'.format(opt.seed)
    opt.exp += size_tag

    # fill time into the name
    if opt.model_path.find('%s') > 0:
        opt.model_path = opt.model_path % (opt.exp, opt.timemark)

    if not os.path.exists(opt.model_path):
        os.makedirs(opt.model_path)

    logging.info('Model_PATH : ' + opt.model_path)
    
    # dump the setting (opt) to disk in order to reuse easily
    if opt.train_from:
        opt = torch.load(
            open(os.path.join(opt.model_path, 'initial.config'), 'rb')
        )
    else:
        torch.save(opt,
                   open(os.path.join(opt.model_path, 'initial.config'), 'wb')
                   )
        json.dump(vars(opt), open(os.path.join(opt.model_path, 'initial.json'), 'w'))

    return opt        
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch Training')
    myConfig.ntm_opts(parser)
    myConfig.model_opts(parser)
    opt = parser.parse_args()        
    opt = process_opt(opt)
    
    if torch.cuda.is_available():
        if not opt.gpuid:
            opt.gpuid = 0
        opt.device = torch.device("cuda:%d" % opt.gpuid)
    else:
        opt.device = torch.device("cpu")
        opt.gpuid = -4
        print("CUDA is not available, fall back to CPU.") 
           
    logging = myConfig.init_logging(log_file=opt.model_path + '/output.log', stdout=True)
    logging.info('Parameters:')
    [logging.info('%s    :    %s' % (k, str(v))) for k, v in opt.__dict__.items()]
    
    main(opt)
       


    
    
    
    
    