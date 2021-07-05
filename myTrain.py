import numpy as np
from tqdm import tqdm
from myUtils import decode_one_hot, fp, cp, time_since, convert_time2str
from myMetric import precision_k, recall_k, f1_score_k
from myModel import myLoss
from myTest import evaluate_loss
import torch
import torch.nn as nn
import torch.nn.functional as F
import os 
import time
import sys
import logging
import math

EPS = 1e-6

def myTrain(model, ntm_model, optimizer_fe, optimizer_ntm, optimizer_whole, 
            train_data_loader, valid_data_loader, bow_dictionary, train_bow_loader,
            valid_bow_loader, opt):    
    logging.info('======================  Start Training  =========================')
    if opt.only_train_ntm or (opt.use_topic_represent and not opt.load_pretrain_ntm):
        print("\nWarming up ntm for %d epochs" % opt.ntm_warm_up_epochs)
        for epoch in range(1, opt.ntm_warm_up_epochs + 1):
            sparsity = train_ntm_one_epoch(ntm_model, train_bow_loader, optimizer_ntm, opt, epoch)
            val_loss = test_ntm_one_epoch(ntm_model, valid_bow_loader, opt, epoch)
            if epoch % 10 == 0:
                ntm_model.print_topic_words(bow_dictionary, os.path.join(opt.model_path, 'topwords_e%d.txt' % epoch))
                best_ntm_model_path = os.path.join(opt.model_path, 'e%d.val_loss=%.3f.sparsity=%.3f.ntm_model' %
                                                   (epoch, val_loss, sparsity))
                logging.info("\nSaving warm up ntm model into %s" % best_ntm_model_path)
                torch.save(ntm_model.state_dict(), open(best_ntm_model_path, 'wb'))
    elif opt.use_topic_represent:
        print("Loading ntm model from %s" % opt.check_pt_ntm_model_path)
        ntm_model.load_state_dict(torch.load(opt.check_pt_ntm_model_path))

    if opt.only_train_ntm:
        print('==========================only_train_ntm completed===========================')
        return 
            
    total_batch = 0
    total_train_loss_statistics = []
    report_train_loss_statistics = []
    report_train_loss = []
    report_valid_loss = []
    best_valid_loss = float('inf')
    best_ntm_valid_loss = float('inf')
    joint_train_patience = 1
    ntm_train_patience = 1
    global_patience = 5
    num_stop_dropping = 0
    num_stop_dropping_ntm = 0
    num_stop_dropping_global = 0

    t0 = time.time()
    Train_fe = True
    begin_iterate_train_ntm = opt.iterate_train_ntm
    check_pt_model_path = ""
    print("\nEntering main training for %d epochs" % opt.epochs)
    for epoch in range(opt.start_epoch, opt.epochs + 1):
        if Train_fe:
            if epoch <= opt.p_fe_e or not opt.joint_train:#warm up train
                optimizer = optimizer_fe
                model.train()
                ntm_model.eval()
                logging.info("\nTraining fe epoch: {}/{}".format(epoch, opt.epochs))
            elif begin_iterate_train_ntm:
                optimizer = optimizer_ntm
                model.train()
                ntm_model.train()
                fix_model(model)
                logging.info("\nTraining ntm epoch: {}/{}".format(epoch, opt.epochs))
                begin_iterate_train_ntm = False
            else:#train whole
                optimizer = optimizer_whole
                unfix_model(model)
                model.train()
                ntm_model.train()
                logging.info("\nTraining fe+ntm epoch: {}/{}".format(epoch, opt.epochs))
                if opt.iterate_train_ntm:
                    begin_iterate_train_ntm = True

            logging.info("The total num of batches: %d, current learning rate:%.6f" %
                         (len(train_bow_loader), optimizer.param_groups[0]['lr']))

            for batch_i, batch in enumerate(train_data_loader):
                total_batch += 1
                #cp(total_batch,'total_batch')
                batch_loss_stat, _ = train_one_batch(batch, model, ntm_model, optimizer, opt, batch_i)
                report_train_loss_statistics.append(batch_loss_stat)

                if (batch_i + 1) % (len(train_bow_loader) // 10) == 0:
                    print("Train: %d/%d batches, current avg loss: %.3f" %
                          ((batch_i + 1), len(train_bow_loader), batch_loss_stat))
            
            #cp(total_batch,'total_batch')
            current_train_loss = np.mean(report_train_loss_statistics)

            # test the model on the validation dataset for one epoch
            model.eval()
            valid_loss_stat = evaluate_loss(valid_data_loader, valid_bow_loader, model, ntm_model, opt)
            
            current_valid_loss = valid_loss_stat

            # debug
            if math.isnan(current_valid_loss) or math.isnan(current_train_loss):
                logging.info(
                    "NaN valid loss. Epoch: %d; batch_i: %d, total_batch: %d" % (epoch, batch_i, total_batch))
                exit()

            if current_valid_loss < best_valid_loss:  # update the best valid loss and save the model parameters
                print("Valid loss drops")
                sys.stdout.flush()
                best_valid_loss = current_valid_loss
                num_stop_dropping = 0
                num_stop_dropping_global = 0
                
                if epoch >= opt.start_checkpoint_at and epoch > opt.p_fe_e and not opt.save_each_epoch:
                    check_pt_model_path = os.path.join(opt.model_path, 'e%d.val_loss=%.3f.model-%s' %
                                                       (epoch, current_valid_loss, convert_time2str(time.time() - t0)))
                    # save model parameters
                    torch.save(
                        model.state_dict(),
                        open(check_pt_model_path, 'wb')
                    )
                    logging.info('Saving fe checkpoints to %s' % check_pt_model_path)

                    if opt.joint_train:
                        check_pt_ntm_model_path = check_pt_model_path.replace('.model-', '.model_ntm-')
                        # save model parameters
                        torch.save(
                            ntm_model.state_dict(),
                            open(check_pt_ntm_model_path, 'wb')
                        )
                        logging.info('Saving ntm checkpoints to %s' % check_pt_ntm_model_path)
            else:
                print("Valid loss does not drop")
                sys.stdout.flush()
                num_stop_dropping += 1
                num_stop_dropping_global += 1
                # decay the learning rate by a factor
                for i, param_group in enumerate(optimizer.param_groups):
                    old_lr = float(param_group['lr'])
                    new_lr = old_lr * opt.learning_rate_decay
                    if old_lr - new_lr > EPS:
                        param_group['lr'] = new_lr
                        print("The new learning rate for fe is decayed to %.6f" % new_lr)

            if opt.save_each_epoch:
                check_pt_model_path = os.path.join(opt.model_path, 'e%d.train_loss=%.3f.val_loss=%.3f.model-%s' %
                                                   (epoch, current_train_loss, current_valid_loss,
                                                    convert_time2str(time.time() - t0)))
                torch.save(  # save model parameters
                    model.state_dict(),
                    open(check_pt_model_path, 'wb')
                )
                logging.info('Saving fe checkpoints to %s' % check_pt_model_path)

                if opt.joint_train:
                    check_pt_ntm_model_path = check_pt_model_path.replace('.model-', '.model_ntm-')
                    torch.save(  # save model parameters
                        ntm_model.state_dict(),
                        open(check_pt_ntm_model_path, 'wb')
                    )
                    logging.info('Saving ntm checkpoints to %s' % check_pt_ntm_model_path)

            # log loss, ppl, and time

            logging.info('Epoch: %d; Time spent: %.2f' % (epoch, time.time() - t0))
            logging.info(
                'avg training loss: %.3f; avg validation loss: %.3f; best validation loss: %.3f' % (
                    current_train_loss, current_valid_loss, best_valid_loss))

            report_train_loss.append(current_train_loss)
            report_valid_loss.append(current_valid_loss)

            report_train_loss_statistics = []

            if not opt.save_each_epoch and num_stop_dropping >= opt.early_stop_tolerance:  # not opt.joint_train or
                logging.info('Have not increased for %d check points, early stop training' % num_stop_dropping)

                break
    #plenty of code 
    if opt.joint_train:
        return check_pt_ntm_model_path, check_pt_model_path
     
    if not opt.joint_train:
        return check_pt_model_path        
    
def train_one_batch(batch, model, ntm_model, optimizer, opt, batch_i):
    # train for one batch
    src_bow, src, trg = batch
    #trg_lens = [int(sum(l)) for l in trg.data.cpu().int().numpy()]
    # move data to GPU if available
    src = src.to(opt.device)
    trg = trg.to(opt.device)

    # model.train()
    optimizer.zero_grad()

    if opt.use_topic_represent:
        src_bow = src_bow.to(opt.device)
        src_bow_norm = F.normalize(src_bow)
        if opt.topic_type == 'z':
            topic_represent, _, recon_batch, mu, logvar = ntm_model(src_bow_norm)
        else:
            _, topic_represent, recon_batch, mu, logvar = ntm_model(src_bow_norm)

        if opt.add_two_loss:
            ntm_loss = loss_function(recon_batch, src_bow, mu, logvar)
    else:
        topic_represent = None

    start_time = time.time()

    # for one2one setting
    y_pred = model(src, topic_represent)
    forward_time = time_since(start_time)
    start_time = time.time()
    ##cp(y_pred.shape, 'y_pred')
    ##cp(trg.float().shape, 'trg.float()')
    loss = myLoss(y_pred, trg.float())

    loss_compute_time = time_since(start_time)

    #total_trg_tokens = sum(trg_lens)

    ##cp(trg_lens, 'trg_lens')
    ##cp(total_trg_tokens, 'total_trg_tokens')
    ##print(type(total_trg_tokens))
    
    if math.isnan(loss.item()):
        print("Batch i: %d" % batch_i)
        print("src")
        print(src)
        print("trg")
        print(trg)
        raise ValueError("Loss is NaN")
    '''
    if opt.loss_normalization == "tokens":  # use number of target tokens to normalize the loss
        normalization = total_trg_tokens
    elif opt.loss_normalization == 'batches':  # use batch_size to normalize the loss
        normalization = src.size(0)
    else:
        raise ValueError('The type of loss normalization is invalid.')
    assert normalization > 0, 'normalization should be a positive number'
    '''

    start_time = time.time()
    if opt.add_two_loss:
        loss = loss+ opt.delta * ntm_loss
    # back propagation on the normalized loss
    #loss.div(normalization).backward()
    loss.backward()
    backward_time = time_since(start_time)
  
    #if opt.max_grad_norm > 0:
    #    grad_norm_before_clipping = nn.utils.clip_grad_norm_(model.parameters(), opt.max_grad_norm)

    optimizer.step()
    
    # construct a statistic object for the loss
    stat = loss.item()                
    return stat, y_pred.detach()
        
def train_ntm_one_epoch(model, dataloader, optimizer, opt, epoch):
    model.train()
    train_loss = 0
    for batch_idx, data_bow in enumerate(dataloader):
        ##cp(data_bow,'data_bow')
        [data_bow] = data_bow
        data_bow = data_bow.to(opt.device)
        # normalize data
        data_bow_norm = F.normalize(data_bow)
        optimizer.zero_grad()
        
        ##cp(data_bow,'data_bow')
        ##cp(data_bow_norm,'data_bow_norm')
        
        _, _, recon_batch, mu, logvar = model(data_bow_norm)
        
        #cp(recon_batch.shape,'recon_batch.shape')
        #cp(data_bow.shape,'data_bow.shape')
        #cp(data_bow_norm.shape,'data_bow_norm.shape')
        #cp(mu.shape,'mu.shape')
        
        loss = loss_function(recon_batch, data_bow, mu, logvar)
        loss = loss + model.l1_strength * l1_penalty(model.fcd1.weight)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        if batch_idx % 100 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data_bow), len(dataloader.dataset),
                       100. * batch_idx / len(dataloader),
                       loss.item() / len(data_bow)))

    logging.info('====>Train epoch: {} Average loss: {:.4f}'.format(
        epoch, train_loss / len(dataloader.dataset)))
    sparsity = check_sparsity(model.fcd1.weight.data)
    logging.info("Overall sparsity = %.3f, l1 strength = %.5f" % (sparsity, model.l1_strength))
    logging.info("Target sparsity = %.3f" % opt.target_sparsity)
    update_l1(model.l1_strength, sparsity, opt.target_sparsity)
    return sparsity

def test_ntm_one_epoch(model, dataloader, opt, epoch):#no use 
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for i, data_bow in enumerate(dataloader):
            [data_bow] =  data_bow
            data_bow = data_bow.to(opt.device)
            data_bow_norm = F.normalize(data_bow)

            _, _, recon_batch, mu, logvar = model(data_bow_norm)
            test_loss += loss_function(recon_batch, data_bow, mu, logvar).item()

    avg_loss = test_loss / len(dataloader.dataset)
    logging.info('====> Val epoch: {} Average loss:  {:.4f}'.format(epoch, avg_loss))
    return avg_loss

# Reconstruction + KL divergence losses summed over all elements and batch
def loss_function(recon_x, x, mu, logvar):
    #cp(recon_x.shape,'recon_x.shape')
    #cp(x.shape,'x.shape')
    BCE = F.binary_cross_entropy(recon_x, x, size_average=False)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD

def l1_penalty(para):
    return nn.L1Loss()(para, torch.zeros_like(para))

def check_sparsity(para, sparsity_threshold=1e-3):
    num_weights = para.shape[0] * para.shape[1]
    num_zero = (para.abs() < sparsity_threshold).sum().float()
    return num_zero / float(num_weights)

def update_l1(cur_l1, cur_sparsity, sparsity_target):
    diff = sparsity_target - cur_sparsity
    cur_l1.mul_(2.0 ** diff)

def fix_model(model):
    for param in model.parameters():
        param.requires_grad = False

def unfix_model(model):
    for param in model.parameters():
        param.requires_grad = True












