import torch
from itertools import combinations

eps4log = 1e-16#eps to avoid log(0)

def log_prb_labels(x, ls):
    """
    x=log(P) is the log probability tensor of shape 
        (number_of_classes + 1, other dimensions like height, width, time, ...)
    ls is a list of distinct labels in range [0, number_of_classes)
    Return log Prb(ls|P)
    
    Note: considering truncated beta series in our paper if ls is too long. 
    This function uses the alternating series for calculation and have complexity 2^L.  
    """
    if ls==[]:
        return torch.sum(x[-1])
    
    P = torch.exp(x[ls + [-1]])
    log_p_ls_n = torch.sum(torch.log(torch.sum(P, dim=0)))#log Prb(labels or NULL)
    sign = (-1)**(len(ls)%2)
    term_positive = 1 + sign*torch.exp(torch.sum(x[-1]) - log_p_ls_n)#this term is supposed to be positive
    for i in range(1, len(ls)):
        sign = -sign
        for comb in combinations(list(range(len(ls))), i):
            log_p = torch.sum(torch.log(torch.sum(P[list(comb)+[-1]], dim=0)))
            term_positive += sign*torch.exp(log_p - log_p_ls_n)
    return log_p_ls_n + torch.log(torch.abs(term_positive) + eps4log)    
    

def log_prb_1l_per_smpl(x, ls):
    """
    x=log(P) is a list of log probability tensors, and it has shape
        (batch_size, number_of_classes + 1, other dimensions like time, ...)
    ls is a list of labels with the same length as x. 
    Each sample has one label in range [0, number_of_classes)
    Return a list of log Prb(label|P)
    """
    B = x.shape[0]#batch size
    dims_sum = list(range(1, len(x.shape) - 1))#sum over these dimensions of x[:, label]
    log_p_n = torch.sum(x[:, -1], dims_sum)
    log_p_l_n = torch.sum(torch.log(torch.exp(x[torch.arange(B), ls]) + torch.exp(x[:, -1])), dims_sum)
    term_positive = 1 - torch.exp(log_p_n - log_p_l_n)
    return log_p_l_n + torch.log(torch.abs(term_positive) + eps4log)


"""Functions below are less important. They are just for special purposes
"""


def remove_repetitive_labels(labels):
    # e.g., convert [2,2,3] to [2,3]
    y = []
    for label in labels:
        if label not in y:
            y.append(label)
    return y


def log_prb_0label(x):
    # return log Prb({}|P=exp(x))
    return torch.sum(x[-1])


def log_prb_1label(x, l):
    # return log Prb({l}|P=exp(x))
    log_p_l_n = torch.sum(torch.log(torch.exp(x[l]) + torch.exp(x[-1])))
    log_p = torch.sum(x[-1])
    term_positive = 1 - torch.exp(log_p - log_p_l_n)
    return log_p_l_n + torch.log(torch.abs(term_positive) + eps4log)


def log_prb_2label(x, ls):
    # return log Prb({l1, l2}|P=exp(x))
    P = torch.exp(x[ls + [-1]])
    log_p_ls_n = torch.sum(torch.log(torch.sum(P, dim=0)))#log Prb alpha
    
    log_p = torch.sum(x[-1])
    term_positive = 1 + torch.exp(log_p - log_p_ls_n)
    for i in range(2):
        log_p = torch.sum(torch.log(P[i] + P[-1]))
        term_positive -= torch.exp(log_p - log_p_ls_n)
    return log_p_ls_n + torch.log(torch.abs(term_positive) + eps4log)


def log_prb_3label(x, ls):
    # return log Prb({l1, l2, l3}|P=exp(x))
    P = torch.exp(x[ls + [-1]])
    log_p_ls_n = torch.sum(torch.log(torch.sum(P, dim=0)))#log Prb alpha
    
    log_p = torch.sum(x[-1])
    term_positive = 1 - torch.exp(log_p - log_p_ls_n)
    for i in range(3):
        log_p = torch.sum(torch.log(P[i] + P[-1]))
        term_positive += torch.exp(log_p - log_p_ls_n)
        for j in range(i+1, 3):
            log_p = torch.sum(torch.log(P[i]+P[j] + P[-1]))
            term_positive -= torch.exp(log_p - log_p_ls_n) 
    return log_p_ls_n + torch.log(torch.abs(term_positive) + eps4log)


def log_prb_4label(x, ls):
    # return log Prb({l1, l2, l3, l4}|P=exp(x))
    P = torch.exp(x[ls + [-1]])
    log_p_ls_n = torch.sum(torch.log(torch.sum(P, dim=0)))#log Prb alpha
    
    log_p = torch.sum(x[-1])
    term_positive = 1 + torch.exp(log_p - log_p_ls_n)
    for i in range(4):
        log_p = torch.sum(torch.log(P[i] + P[-1]))
        term_positive -= torch.exp(log_p - log_p_ls_n)
        for j in range(i+1, 4):
            log_p = torch.sum(torch.log(P[i]+P[j] + P[-1]))
            term_positive += torch.exp(log_p - log_p_ls_n)
            for k in range(j+1, 4):
                log_p = torch.sum(torch.log(P[i]+P[j]+P[k] + P[-1]))
                term_positive -= torch.exp(log_p - log_p_ls_n)    
    return log_p_ls_n + torch.log(torch.abs(term_positive) + eps4log)


def log_prb_5label(x, ls):
    # x is log(P)
    # ls is {l1,l2,l3,l4,l5}, excluding NULL
    # return log Prb(ls|P)    
    P = torch.exp(x[ls + [-1]])
    log_p_ls_n = torch.sum(torch.log(torch.sum(P, dim=0)))#log Prb alpha
    
    log_p = torch.sum(x[-1])
    term_positive = 1 - torch.exp(log_p - log_p_ls_n)
    for i in range(5):
        log_p = torch.sum(torch.log(P[i] + P[-1]))
        term_positive += torch.exp(log_p - log_p_ls_n)
        for j in range(i+1, 5):
            log_p = torch.sum(torch.log(P[i]+P[j] + P[-1]))
            term_positive -= torch.exp(log_p - log_p_ls_n)
            for k in range(j+1, 5):
                log_p = torch.sum(torch.log(P[i]+P[j]+P[k] + P[-1]))
                term_positive += torch.exp(log_p - log_p_ls_n)
                for m in range(k+1, 5):
                    log_p = torch.sum(torch.log(P[i]+P[j]+P[k]+P[m] + P[-1]))
                    term_positive -= torch.exp(log_p - log_p_ls_n)#the last sign is -       
    return log_p_ls_n + torch.log(torch.abs(term_positive) + eps4log)


def log_prb_6label(x, ls): 
    # return log Prb({l1, l2, l3, l4, l5, l6}|P=exp(x))
    P = torch.exp(x[ls + [-1]])
    log_p_ls_n = torch.sum(torch.log(torch.sum(P, dim=0)))#log Prb alpha
    
    log_p = torch.sum(x[-1])
    term_positive = 1 + torch.exp(log_p - log_p_ls_n)
    for i in range(6):
        log_p = torch.sum(torch.log(P[i] + P[-1]))
        term_positive -= torch.exp(log_p - log_p_ls_n)
        for j in range(i+1, 6):
            log_p = torch.sum(torch.log(P[i]+P[j] + P[-1]))
            term_positive += torch.exp(log_p - log_p_ls_n)
            for k in range(j+1, 6):
                log_p = torch.sum(torch.log(P[i]+P[j]+P[k] + P[-1]))
                term_positive -= torch.exp(log_p - log_p_ls_n)
                for m in range(k+1, 6):
                    log_p = torch.sum(torch.log(P[i]+P[j]+P[k]+P[m] + P[-1]))
                    term_positive += torch.exp(log_p - log_p_ls_n)
                    for n in range(m+1, 6):
                        log_p = torch.sum(torch.log(P[i]+P[j]+P[k]+P[m]+P[n] + P[-1]))
                        term_positive -= torch.exp(log_p - log_p_ls_n)
    return log_p_ls_n + torch.log(torch.abs(term_positive) + eps4log)


# other even less important functions
import random
import scipy.io
import numpy as np


def nest_images(images, nestH, nestW):
    iH, iW = images.shape[2], images.shape[3]
    y = torch.zeros(images.shape[1], nestH, nestW)
    for image in images:
        m = random.randint(0, nestH-iH)
        n = random.randint(0, nestW-iW)
        y[:,m:m+iH,n:n+iW] += image
        
    y[y>1] = 1
    return y


def read_svhn_mat(file):
    # see ***_preprocessing.m for detailed steps to get these mat files
    data = scipy.io.loadmat(file)
    images = np.transpose(data['Images'], [3,2,0,1])
    labels = []
    for mat_label in data['Labels']:
        label = []
        for l in mat_label[0][0]:
            label.append(int(l))
        labels.append(label)
    return images, np.array(labels)