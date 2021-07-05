import numpy as np
import torch.utils.data as data_utils
import torch
import random
from myUtils import fp, cp, time_since, load_dict, encode_one_hot, load_npy
import pickle
import time
import logging
import gensim
import re

def load_data(opt):
    base_path = opt.load_path
    opt.data_path = base_path+'processed_data'
    batch_size = opt.batch_size

    bow_dictionary = load_dict(base_path, 'bow_dictionary')
    word2idx = load_dict(base_path, 'word2idx')
    tag2idx = load_dict(base_path, 'tag2idx')
    opt.label_num = len(tag2idx)
    
    X_bow_train = load_npy('train_bow', opt)
    X_train = load_npy('train_text', opt)
    y_train = load_npy('train_label', opt)
    X_bow_valid = load_npy('valid_bow', opt)
    X_valid = load_npy('valid_text', opt)
    y_valid = load_npy('valid_label', opt)
    X_bow_test = load_npy('test_bow', opt)
    X_test = load_npy('test_text', opt)
    y_test = load_npy('test_label', opt)    
    
    #BoW feature
    train_bow_data = data_utils.TensorDataset(torch.from_numpy(X_bow_train).type(torch.float32))
    val_bow_data = data_utils.TensorDataset(torch.from_numpy(X_bow_valid).type(torch.float32))
                         
    test_bow_data = data_utils.TensorDataset(torch.from_numpy(X_bow_test).type(torch.float32))
    
    train_bow_loader = data_utils.DataLoader(train_bow_data, batch_size, shuffle=True, drop_last=True)
    valid_bow_loader = data_utils.DataLoader(val_bow_data, batch_size, shuffle=True, drop_last=True)
    test_bow_loader = data_utils.DataLoader(test_bow_data, batch_size, drop_last=True)
    
    #Nomral feature and label
    train_data = data_utils.TensorDataset(torch.from_numpy(X_bow_train).type(torch.float32),
                                          torch.from_numpy(X_train).type(torch.LongTensor),
                                          torch.from_numpy(y_train).type(torch.LongTensor))
    val_data = data_utils.TensorDataset(torch.from_numpy(X_bow_valid).type(torch.float32),
                                        torch.from_numpy(X_valid).type(torch.LongTensor),
                                          torch.from_numpy(y_valid).type(torch.LongTensor))                                          
    test_data = data_utils.TensorDataset(torch.from_numpy(X_bow_test).type(torch.float32),
                                         torch.from_numpy(X_test).type(torch.LongTensor),
                                         torch.from_numpy(y_test).type(torch.LongTensor))
    
    train_loader = data_utils.DataLoader(train_data, batch_size, shuffle=True, drop_last=True)
    val_loader = data_utils.DataLoader(val_data, batch_size, shuffle=True, drop_last=True)
    test_loader = data_utils.DataLoader(test_data, batch_size, drop_last=True)
    
    #label_num = int(train_label.max())
    #vocab_size = int(train_text.max())+2# +2 Don't Know Why
    #fp('label_num')
    #fp('vocab_size')
    print("load done")
    
    return train_loader, val_loader, test_loader, train_bow_loader, \
    valid_bow_loader, test_bow_loader, bow_dictionary, opt
    

