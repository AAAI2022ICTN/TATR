import numpy as np
from tqdm import tqdm
from myModel import myLoss, NTM
from myUtils import decode_one_hot, fp, cp, time_since
from myMetric import precision_k, recall_k, f1_score_k, evaluator
import torch
import torch.nn.functional as F
import time

def evaluate_loss(data_loader, valid_bow_loader, model, ntm_model, opt):

    model.to(opt.device)
    ntm_model.to(opt.device)
    model.eval()
    ntm_model.eval()
    
    evaluation_loss_sum = 0.0
    total_trg_tokens = 0
    n_batch = 0
    loss_compute_time_total = 0.0
    forward_time_total = 0.0
    print("Evaluate loss for %d batches" % len(valid_bow_loader))
    with torch.no_grad():
        for batch_i, batch in enumerate(data_loader):
            src_bow, src, trg = batch
            #trg_lens = [int(sum(l)) for l in trg.data.cpu().int().numpy()]
            batch_size = src.size(0)
            n_batch += batch_size

            # move data to GPU if available
            src = src.to(opt.device)
            trg = trg.to(opt.device)
            
            if opt.use_topic_represent:
                src_bow = src_bow.to(opt.device)
                src_bow_norm = F.normalize(src_bow)
                if opt.topic_type == 'z':
                    topic_represent, _, _, _, _ = ntm_model(src_bow_norm)
                else:
                    _, topic_represent, _, _, _ = ntm_model(src_bow_norm)
            else:
                topic_represent = None

            start_time = time.time()

            # one2one setting
            y_pred = model(src, topic_represent)

            forward_time = time_since(start_time)
            forward_time_total += forward_time

            start_time = time.time()
            loss = myLoss(y_pred, trg.float())
            loss_compute_time = time_since(start_time)
            loss_compute_time_total += loss_compute_time

            evaluation_loss_sum += loss.item()/len(data_loader) 
            #total_trg_tokens += sum(trg_lens)
               
            ##cp(batch_i,'batch_i')
            ##cp(len(valid_bow_loader),'len(valid_bow_loader)')
            ##cp((batch_i + 1) % (len(valid_bow_loader),'(batch_i + 1) % (len(valid_bow_loader)')
            ##cp(len(valid_bow_loader) // 5,'len(valid_bow_loader) // 5')
            
            if (batch_i + 1) % (len(valid_bow_loader) // 5) == 0:
                print("Valid: %d/%d batches, current avg loss: %.3f" %
                      ((batch_i + 1), len(valid_bow_loader), evaluation_loss_sum ))

    #eval_loss_stat = LossStatistics(evaluation_loss_sum, total_trg_tokens, n_batch, forward_time=forward_time_total,
                                    #loss_compute_time=loss_compute_time_total)
    return evaluation_loss_sum

def myTest(model, ntm_model, test_data_loader ,bow_dictionary, test_bow_loader, check_pt_ntm_model_path, check_pt_model_path, opt):

    y_test = []
    y_pred = []
    
    print("**************************************")
    print("**************************************")
    print("this is the final results")
    
    #load best model 
    model.load_state_dict(torch.load(check_pt_model_path))
    model.to(opt.device)
    model.eval()
    if opt.use_topic_represent:
        ntm_model = NTM(opt)
        ntm_model.load_state_dict(torch.load(check_pt_ntm_model_path))
        ntm_model.to(opt.device)
        ntm_model.eval()
    else:
        ntm_model = None
    model.eval()
    #ntm_model.eval()
    
    #test
    print("Evaluate loss for %d batches" % len(test_data_loader))
    with torch.no_grad():
        for batch_i, batch in enumerate(test_data_loader):
            src_bow, src, trg = batch
            trg_lens = [int(sum(l)) for l in trg.data.cpu().int().numpy()]
            # move data to GPU if available
            src = src.to(opt.device)
            trg = trg.to(opt.device)
            #NTM
            if opt.use_topic_represent:
                src_bow = src_bow.to(opt.device)
                src_bow_norm = F.normalize(src_bow)
                if opt.topic_type == 'z':
                    topic_represent, _, recon_batch, mu, logvar = ntm_model(src_bow_norm)
                else:
                    _, topic_represent, recon_batch, mu, logvar = ntm_model(src_bow_norm)
            else:
                topic_represent = None
            #main model
            pred = model(src, topic_represent)

            labels_cpu = trg.data.cpu().float().numpy()
            pred_cpu = pred.data.cpu().numpy()
            pred_cpu = np.exp(pred_cpu)
            
            y_test.append(labels_cpu)
            y_pred.append(pred_cpu)
    
    y_test = np.array(y_test)
    y_test = np.reshape(y_test,(-1,y_test.shape[-1]))
    y_pred = np.array(y_pred)
    y_pred = np.reshape(y_pred,(-1,y_pred.shape[-1]))    
       
    top_K = 1
    acc, precision, recall, f1 = evaluator(y_test, y_pred, top_K)
    print('pre@%d,re@%d,f1@%d'%(top_K,top_K,top_K))
    print(round(precision,3),round( recall,3),round( f1,3) )
    top_K = 3
    acc, precision, recall, f1 = evaluator(y_test, y_pred, top_K)
    print('pre@%d,re@%d,f1@%d'%(top_K,top_K,top_K))
    print(round(precision,3),round( recall,5),round( f1,3) )
    top_K = 5
    acc, precision, recall, f1 = evaluator(y_test, y_pred, top_K)
    print('pre@%d,re@%d,f1@%d'%(top_K,top_K,top_K))
    print(round(precision,3),round( recall,3),round( f1,3) )    
            
