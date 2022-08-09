#!/usr/bin/env python
# coding: utf-8
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.svm import OneClassSVM, SVC
from sklearn.neighbors import LocalOutlierFactor
from sklearn.ensemble import IsolationForest
import numpy as np
import matplotlib.pyplot as plt

from sklearn import svm, datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels

import os
import torch as tc
from conf_set.utils import *

import sys
import pickle
import math
import pdb
import seaborn as sns
import iid_Q1 as iid

# ## Generate a toy dataset, with label '0' indicating normal and label '1' indicating abnormal 

# Generate and plot a synthetic imbalanced classification dataset
from collections import Counter
from sklearn.datasets import make_classification
from matplotlib import pyplot
from numpy import where
from utils import dataLoading

# def split_data(normalDataX, normalDatay, abnormalDataX, abnormalDatay, testsize, valsize, synth=True):

#     if synth == True:
#         testindiciesNormal = np.random.randint(low=0, 
#                                             high=normalDataX.shape[0],
#                                             size=testsize)
#         testindiciesAbnormal = np.random.randint(low=0,
#                                                 high=abnormalDataX.shape[0],
#                                                 size=testsize)
#         valindiciesNormal = np.random.randint(low=0, 
#                                             high=normalDataX.shape[0],
#                                             size=valsize)
#         valindiciesAbnormal = np.random.randint(low=0,
#                                                 high=abnormalDataX.shape[0],
#                                                 size=valsize)
#         testX = np.hstack([normalDataX[testindiciesNormal,], abnormalDataX[testindiciesAbnormal,]]) 
#         testy = np.hstack([normalDatay[testindiciesNormal,], abnormalDatay[testindiciesAbnormal,]])
#         valX_5 = np.hstack([normalDataX[valindiciesNormal[:int(valsize*0.5),],], abnormalDataX[valindiciesAbnormal[:int(valsize*0.1),],]]) 
#         valy_5 = np.hstack([normalDatay[valindiciesNormal[:int(valsize*0.5),],], abnormalDatay[valindiciesAbnormal[:int(valsize*0.1),],]])
#         valX_75 = np.hstack([normalDataX[valindiciesNormal[:int(valsize*0.75),],], abnormalDataX[valindiciesAbnormal[:int(valsize*0.5),],]]) 
#         valy_75 = np.hstack([normalDatay[valindiciesNormal[:int(valsize*0.75),],], abnormalDatay[valindiciesAbnormal[:int(valsize*0.5),],]]) 
#         valX = np.hstack([normalDataX[valindiciesNormal,], abnormalDataX[valindiciesAbnormal,]]) 
#         valy = np.hstack([normalDatay[valindiciesNormal,], abnormalDatay[valindiciesAbnormal,]]) 
#         valX = [valX_5, valX_75, valX]
#         valy = [valy_5, valy_75, valy]
#     else:
#         # testsize = int(normalDataX.shape[0] - trainsize)
#         testsize = abnormalDataX.shape[0]
#         indices = np.arange(normalDataX.shape[0])
#         np.random.shuffle(indices)
#         indices_normal = indices[:testsize]
#         indices_abnormal = np.arange(abnormalDataX.shape[0])
#         np.random.shuffle(indices_abnormal)
#         testX = np.hstack([normalDataX[indices,], abnormalDataX[indices_abnormal,]])
#         testy = np.hstack([normalDatay[indices,], abnormalDatay[indices_abnormal,]])
#         testX, valX, testy, valy = train_test_split(testX, testy, test_size=0.5, random_state=30, stratify=testy)
#         valX = [valX[:int(0.5*valX.shape[0])], valX[:int(0.75*valX.shape[0])], valX]
#         valy = [valy[:int(0.5*valy.shape[0])], valy[:int(0.75*valy.shape[0])], valy]
#     return testX, testy, valX, valy


# def train(model, trainX, trainy):
    
#     trainX = trainX[trainy==0]
#     model.fit(trainX)
#     # detect outliers in the test set
#     # yhat = model.predict(testX)
#     # yhat[yhat==1] = 0
#     # yhat[yhat==-1] = 1
#     # probas = -model.score_samples(testX)
#     # normalprobas = np.sort(probas[testy==0])
#     # abnormalprobas = np.sort(probas[testy==1])
#     # # breakpoint()
#     # normal_max = normalprobas[-int(normalprobas.shape[0]*epsilon)]
#     # abnorm_min = abnormalprobas[int(abnormalprobas.shape[0]*epsilon)]
#     # # print('normal max', normal_max)
#     # # print('abnormal min', abnorm_min)
#     # min_tmp = np.min([abnorm_min, normal_max])
#     # max_tmp = np.max([abnorm_min, normal_max])
#     # uncertainty = np.sum((probas>min_tmp) * (probas<max_tmp))
#     # frac = uncertainty / probas.shape[0]
    
#     # plt.boxplot(x=[normalprobas, abnormalprobas], positions = [1,2], labels=['normal', 'abnormal'] )
#     # plt.savefig('Box.png')
#     # error = np.mean(testy != yhat)

#     # return model, abnorm_min, normal_max, frac, epsilon, epsilon
#     return model

# def find_maximum_train_error_allow(eps, delta, n):

#         k_min = 0
#         k_max = n
#         bnd_min = half_line_bound_upto_k(n, k_min, eps)
#         # breakpoint()
#         if bnd_min > delta:
#             return None
#         assert(bnd_min <= delta)
#         k = n
#         while True:
#             # choose new k
#             k_prev = k
#             k = (T(k_min + k_max).float()/2.0).round().long().item()
        
#             # terinate condition
#             if k == k_prev:
#                 break
        
#             # check whether the current k satisfies the condition
#             bnd = half_line_bound_upto_k(n, k, eps)
#             # bnd_p = _half_line_bound_upto_k(n, k, eps)
#             # breakpoint()
#             if bnd <= delta:
#                 k_min = k
#             else:
#                 k_max = k

#         # confirm that the solution satisfies the condition
#         k_best = k_min
#         # tmp = half_line_bound_upto_k(n, k_best, eps)
#         # breakpoint()
#         assert(half_line_bound_upto_k(n, k_best, eps) <= delta)
#         error_allow = float(k_best) / float(n)
#         return error_allow


# # ### Second, empirical_cs_error_fn is used to compute the error based on the current threshold T. empirical_cs_error_fn function computes the false negative rate as the error. False negative rate is the rate that the classifier classifier an unsafe state as safe.
# def empirical_cs_error_fn(T, valX, valy, n=None, device=tc.device("cpu")):
#     # probas = -model.score_samples(valX) 
#     yhat = np.zeros_like(valy)
#     # yhat[probas > T] = 1
#     # yhat[probas < T] = 0
#     yhat[valX > T] = 1
#     yhat[valX < T] = 0
#     tn, fp, fn, tp = confusion_matrix(valy, yhat).ravel()
#     error = fn / (fn + tp)
#     return error, [], []


# # ## empirical_cs_error_fn function computes the false positive rate as the error. False positive rate is the rate that the classifier classifier a safe state as unsafe.
# def empirical_cs_error_fp(T, valX, valy, n=None, device=tc.device("cpu")):
#     # probas = -model.score_samples(valX)
#     yhat = np.zeros_like(valy)
#     # yhat[probas > T] = 1
#     # yhat[probas < T] = 0
#     # breakpoint()
#     yhat[valX > T] = 1
#     yhat[valX < T] = 0
#     tn, fp, fn, tp = confusion_matrix(valy, yhat).ravel()
#     error = fp / (fp + tn)
#     return error, [], []

# def geb_VC(delta, n, d=1.0):
#     n = float(n)
#     g = np.sqrt(((np.log(2*n / d) + 1) * d + np.log(4 / delta)) / n)
#     return g

# def compute_tr_error_allow_VC(eps, delta, n):
#     g = geb_VC(delta, n)
    
#     error_allow = eps - g
#     if error_allow < 0.0:
#         return None
#     else:
#         return error_allow

# def find_cs_level(valX, valy, n, train_error_allow, fp, T_min, T_max, T_diff):
#         if fp == True:
#             T_best = 1.0
#         else:
#             T_best = 0.0

#         while True:
            
#             # update
#             T_cur = (T_max + T_min)/2.0
            
#             # terminate condition
#             if abs(T_min-T_max) <= T_diff:
#                 break
            
#             # update lower and max bounds
#             if fp:
#                 error, _, _ = empirical_cs_error_fp(T_cur, valX, valy)
#             else:
#                 error, _, _ = empirical_cs_error_fn(T_cur, valX, valy)
                
#             if error <= train_error_allow:
#                 T_best = T_cur
#                 if fp:
#                     T_max = T_cur
#                 else:
#                     T_min = T_cur

#                 # print("[best threshold] error = %f, train_error_allow = %f, T = %f"%(error, train_error_allow, T_best))
#             else:
#                 if fp:
#                     T_min = T_cur
#                 else:
#                     T_max = T_cur
#         return T_best

# def evaluate(probas, testy, valX, valy, epsilon, delta):
#     train_error_allow_fp = find_maximum_train_error_allow(epsilon, delta, np.sum(valy==0))
#     train_error_allow_fn = find_maximum_train_error_allow(epsilon, delta, np.sum(valy==1))
#     # train_error_allow_fp = compute_tr_error_allow_VC(epsilon, delta, np.sum(valy==0))
#     # train_error_allow_fn = compute_tr_error_allow_VC(epsilon, delta, np.sum(valy==1))
#     # pdb.set_trace()
#     T_best_fp = find_cs_level(valX, valy, n=np.sum(valy==0), train_error_allow=train_error_allow_fp, fp=True, T_min=0.0, T_max=1.0, T_diff=1e-7)
#     T_best_fn = find_cs_level(valX, valy, n=np.sum(valy==1), train_error_allow=train_error_allow_fn, fp=False, T_min=0.0, T_max=1.0, T_diff=1e-7)

#     # X_val_0 = np.sort(valX[valy==0,])
#     # X_val_1 = np.sort(valX[valy==1,])
#     # T_best_fn = X_val_1[math.floor(X_val_1.shape[0] * train_error_allow_fn)]
#     # T_best_fp = X_val_0[-math.floor(X_val_0.shape[0] * train_error_allow_fp)]

#     yhat = np.zeros_like(testy)
#     yhat[probas > T_best_fn] = 1
#     yhat[probas < T_best_fn] = 0
#     tn, fp, fn, tp = confusion_matrix(testy, yhat).ravel()
#     fnr = fn / (fn + tp)
#     yhat = np.zeros_like(testy)
#     yhat[probas > T_best_fp] = 1
#     yhat[probas < T_best_fp] = 0
#     tn, fp, fn, tp = confusion_matrix(testy, yhat).ravel()
#     fpr = fp / (tn + fp)

#     normalprobas = probas[testy==0]
#     abnormalprobas = probas[testy==1]
#     min_tmp = np.min([T_best_fn, T_best_fp])
#     max_tmp = np.max([T_best_fn, T_best_fp])
#     uncertainty = np.sum((probas<max_tmp) * (probas>min_tmp))
#     frac = uncertainty / probas.shape[0]

#     # filter out uncertain pts
#     # remain = probas[(probas <= min_tmp) | (probas >= max_tmp)]
    
    
#     remain = probas[(probas >= max_tmp)]
#     positive = probas[testy == 1]

#     yhat = np.zeros_like(positive)
#     yhat[positive >= max_tmp] = 1
#     error = np.mean(yhat != testy[testy == 1])

#     return T_best_fp, T_best_fn, frac, fpr, fnr

# def evaluate_tau(probas, testy, tau):
#     yhat = np.zeros_like(testy)
#     yhat[probas > tau] = 1
#     yhat[probas < tau] = 0
#     tn, fp, fn, tp = confusion_matrix(testy, yhat).ravel()
#     fnr = fn / (fn + tp)
#     fpr = fp / (tn + fp)
#     error = np.mean(yhat != testy)
#     return fnr, fpr, error

# def sigmoid(x):
#     z = 1 / (1 + np.exp(-x))
#     return z

# def strategy(T_fn, T_fp, testX, testy, valX, valy, epsilon, delta):
#     tau = -1.0
#     if T_fn > T_fp:
#         tau = (T_fn + T_fp) / 2.0
#     else:
#         while True:
#             T_fp, T_fn, frac, fpr, fnr = evaluate(testX, testy, valX, valy, epsilon, delta)
#             if T_fn > T_fp:
#                 tau = (T_fn + T_fp) / 2.0
#                 break
#             else:
#                 epsilon += 0.1
#     return tau, T_fn, T_fp, epsilon


if __name__ == "__main__":

    plt.rcParams.update({'font.size':20})

    np.random.seed(seed=42)
    synth = True
    trainsize = int(5e4)
    valsize = int(2e3)
    testsize = int(5e4)

    X, y = make_classification(n_samples=1000000, n_features=6, n_informative=5, n_redundant=0,
        n_clusters_per_class=1, weights=[0.99], class_sep=5.0, flip_y=0, random_state=42)

    length = X.shape[0]
    normalDataX = X[y==0,:]
    normalDatay = y[y==0]
    abnormalDataX = X[y==1, :]
    abnormalDatay = y[y==1]

    trainX, trainy = normalDataX[:trainsize,], normalDatay[:trainsize,]
    print('Training!')
    gts = []
    model = IsolationForest()
    model = iid.train(model, trainX, trainy)

    f = open('log', 'w')

    # Delta verification
    epsilons = [0.05, 0.1, 0.15, 0.2]
    deltas = [0.05, 0.1, 0.15, 0.2]

    ambiguous = np.zeros([4,4])
    probas_normalDataX = -model.score_samples(normalDataX)
    probas_normalDataX = iid.sigmoid(probas_normalDataX)
    probas_abnormalDataX = -model.score_samples(abnormalDataX)
    probas_abnormalDataX = iid.sigmoid(probas_abnormalDataX)
    for i_epsilon, epsilon in enumerate(epsilons):
        for i_delta, delta in enumerate(deltas):
            trials = 100
            fracs = []

            for i in range(trials):
                print("Trial", i)
                testX, testy, [valX_5, valX_75, valX], [valy_5, valy_75, valy] = iid.split_data(probas_normalDataX, 
                                                                                normalDatay,
                                                                                probas_abnormalDataX,
                                                                                abnormalDatay,
                                                                                testsize=testsize,
                                                                                valsize=valsize,
                                                                                synth=synth)
                T_best_fp, T_best_fn, frac, fpr, fnr = iid.evaluate(testX, testy, valX, valy, epsilon, delta)
                fracs.append(frac)

            ambiguous[i_epsilon, i_delta] = np.mean(fracs)
            np.save(f'Q4.npy', ambiguous)
    
    plt.figure(figsize=[6.5,6.5])
    ambiguous = np.load('Q4.npy')
    ambiguous = ambiguous[:, ::-1]
    ax = sns.heatmap(ambiguous, linewidth=0.5)
    plt.yticks(np.arange(4)+0.5, ['0.2', '0.15', '0.10', '0.05'])
    plt.xticks(np.arange(4)+0.5, ['0.05', '0.1', '0.15', '0.2'])
    plt.xlabel(r'$\delta$')
    plt.ylabel(r'$\epsilon$')
    plt.savefig('ambiguous.png')
        

        
