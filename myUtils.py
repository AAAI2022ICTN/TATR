import os
import sys
import csv
import codecs
import time
import numpy as np
import math
import pickle

def load_dict(base_path, dict_name):
    file_name=base_path+dict_name
    f=open(file_name,'rb')
    mydict=pickle.load(f)
    f.close()
    return mydict
    
def load_npy(name, opt):
    path = opt.data_path+'/%s.npy'%name
    data = np.load(path, allow_pickle=True)
    print('%s readed, shape:'%name, data.shape)
    return data

def time_since(start_time):
    return time.time()-start_time
    
def convert_time2str(seconds):
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    return "%dh-%02dm" % (h, m)
    
def cp(var, var_name='begin'):
    '''
    clearly print, in other words, let us clearly find the print
    '''
    print('===================%s===================='%var_name)
    print(var)    
    print('===================%s===================='%var_name)
       
def fp(name):
    '''
    fast print the str and related variable
    '''
    print(name+'\n', sys._getframe().f_back.f_locals[name])

def decode_one_hot(one_hot_label):
    '''
    Input: one-hot form label
    Output: value form label
    '''
    label_value = one_hot_label.argsort()[-int(sum(one_hot_label)):]
    return label_value

def encode_one_hot(inst, vocab_size, label_from):
    '''
    one hot for a value x, int, x>=1
    '''
    one_hots = np.zeros(vocab_size, dtype=np.float32)
    for value in inst:
        one_hots[value-label_from]=1
    return one_hots

