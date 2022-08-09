import torch 
import torchvision
import numpy as np
from torch import nn
import sklearn
from sklearn.linear_model import LogisticRegression
import iid_Q1 as iid
from sklearn.metrics import confusion_matrix
from typing import Any, Callable, Dict, List, Optional, Tuple

import time
import matplotlib.pyplot as plt
import numpy as np

from sklearn.datasets import fetch_openml
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils import check_random_state
from sklearn.svm import OneClassSVM
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from analysis import clopper_pearson

import pdb

random_seeds = [10, 20, 30]

fprs_CP = []
fnrs_CP = []
fracs_CP = []

fprs_PAC = []
fnrs_PAC = []
fracs_PAC = []
"""
for random_seed in random_seeds:
    np.random.seed(random_seed)

    # Turn down for faster convergence
    t0 = time.time()
    f = open('log', 'w')

    # Load data from https://www.openml.org/d/554
    X, y = fetch_openml("mnist_784", version=1, return_X_y=True, as_frame=False)

    train_data = X[np.isin(y, ['0', '6','8','9']),]
    train_targets = y[np.isin(y, ['0', '6','8','9']),]
    train_targets[np.isin(train_targets, ['0','6','9'])] = 0
    train_targets[np.isin(train_targets, ['8'])] = 1

    random_state = check_random_state(random_seed)
    trials = 100
    epsilon = 0.05
    delta = 0.05
    permutation = random_state.permutation(train_data.shape[0])
    train_data = train_data[permutation]
    train_targets = train_targets[permutation]

    train_data_0 = train_data[train_targets==0,][:2502,]
    train_data_1 = train_data[train_targets==1,][:542]

    test_data_0 = train_data[train_targets==0,][2502:3204,]
    test_data_1 = train_data[train_targets==1,][542:708]

    X_train = np.vstack([train_data_0, train_data_1])
    y_train = np.zeros(X_train.shape[0])
    y_train[-542:] = 1
    X_test = np.vstack([test_data_0, test_data_1])
    y_test= np.zeros(X_test.shape[0])
    y_test[-166:] = 1

    X_train, X_val, y_train, y_val = sklearn.model_selection.train_test_split(X_train, y_train, train_size=0.66)

    normal_points = np.sum(y_val==0)
    abnormal_points = np.sum(y_val==1)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val  = scaler.fit_transform(X_val)
    X_test = scaler.transform(X_test)

    X_test = np.vstack([X_val, X_test])
    y_test = np.hstack([y_val, y_test])

    # Turn up tolerance for faster convergence
    clf = LogisticRegression(C=25.0 / 5000, penalty="l1", solver="saga", tol=0.1)
    clf.fit(X_train, y_train)
    sparsity = np.mean(clf.coef_ == 0) * 100
    score = clf.score(X_test, y_test)
    # print('Best C % .4f' % clf.C_)
    print("Sparsity with L1 penalty: %.2f%%" % sparsity)
    print("Test score with L1 penalty: %.4f" % score)

    probas = clf.predict_proba(X_test)[:, 1] 
    testy = y_test

    for i in range(trials):

        normal_indices = np.nonzero(y_test==0)[0]
        abnormal_indices = np.nonzero(y_test==1)[0]
        np.random.shuffle(normal_indices)
        np.random.shuffle(abnormal_indices)
        normal_indices = normal_indices[:normal_points]
        abnormal_indices = abnormal_indices[:abnormal_points]
        X_val = np.hstack([probas[normal_indices], probas[abnormal_indices]])
        y_val = np.hstack([y_test[normal_indices], y_test[abnormal_indices]])

        X_val_0 = np.sort(X_val[y_val==0])
        X_val_1 = np.sort(X_val[y_val==1])
        T_best_fn = X_val_1[int(X_val_1.shape[0] * epsilon)]
        T_best_fp = X_val_0[int(-X_val_0.shape[0]*epsilon)]

        yhat = np.zeros_like(testy)
        yhat[probas > T_best_fn] = 1
        yhat[probas < T_best_fn] = 0
        tn, fp, fn, tp = confusion_matrix(testy, yhat).ravel()
        baseline_fnr = fn / (fn + tp)
        yhat = np.zeros_like(testy)
        yhat[probas > T_best_fp] = 1
        yhat[probas < T_best_fp] = 0
        tn, fp, fn, tp = confusion_matrix(testy, yhat).ravel()
        baseline_fpr = fp / (tn + fp)

        normalprobas = probas[testy==0]
        abnormalprobas = probas[testy==1]
        min_tmp = np.min([T_best_fn, T_best_fp])
        max_tmp = np.max([T_best_fn, T_best_fp])
        uncertainty = np.sum((probas<max_tmp) * (probas>min_tmp))
        baseline_frac = uncertainty / probas.shape[0]

        print('CP fpr', baseline_fpr)
        print('CP fnr', baseline_fnr)
        print('CP frac', baseline_frac)

        fprs_CP.append(baseline_fpr)
        fnrs_CP.append(baseline_fnr)
        fracs_CP.append(baseline_frac)

        T_best_fp_PAC, T_best_fn_PAC, frac, fpr, fnr = iid.evaluate(probas, y_test, X_val, y_val, epsilon, delta)
        print('PAC fpr', fpr)
        print('PAC fnr', fnr)
        print('PAC frac', frac)

        fprs_PAC.append(fpr)
        fnrs_PAC.append(fnr)
        fracs_PAC.append(frac)

fprs_CP = np.array(fprs_CP).reshape(-1, 1)
fnrs_CP = np.array(fnrs_CP).reshape(-1, 1)
fracs_CP = np.array(fracs_CP).reshape(-1, 1)
results_CP = np.hstack([fprs_CP, fnrs_CP, fracs_CP])
np.save(f'CP.npy', results_CP)
fprs_CP = np.sum(fprs_CP<epsilon) / trials
fnrs_CP = np.sum(fnrs_CP<epsilon) / trials
fracs_CP = np.sum(fracs_CP) / trials
print('CP average fpr', fprs_CP, file=f)
print('CP average fnr', fnrs_CP, file=f)
print('CP average fracs', fracs_CP, file=f)

fprs_PAC = np.array(fprs_PAC).reshape(-1, 1)
fnrs_PAC = np.array(fnrs_PAC).reshape(-1, 1)
fracs_PAC = np.array(fracs_PAC).reshape(-1, 1)
results_PAC = np.hstack([fprs_PAC, fnrs_PAC, fracs_PAC])
np.save(f'PAC.npy', results_PAC)
fprs_PAC = np.sum(fprs_PAC<epsilon) / trials
fnrs_PAC = np.sum(fnrs_PAC<epsilon) / trials
fracs_PAC = np.sum(fracs_PAC) / trials
print('PAC fpr', fprs_PAC, file=f)
print('PAC fnr', fnrs_PAC, file=f)
print('PAC fracs', fracs_PAC, file=f)
"""

method = 'CP'

trials = 4000
data = np.load(f'{method}.npy', allow_pickle=True)[:trials, :]
batch_size = 25
epsilon = 0.05
iters = int(data.shape[0] / batch_size)

total = 300
successes = data > epsilon
rates = ['fpr', 'fnr', 'frac']
for i in range(3):
    lo, hi = clopper_pearson(np.sum(successes[:, i]), total)
    print('CP {:s} 95% confidence interval: {:.3f}-{:.3f}'.format(rates[i], lo, hi))

print('CP average FPR', np.mean(data[:, 0]))
print('CP average FNR', np.mean(data[:, 1]))
print('CP average frac', np.mean(data[:, 2]))

# fprs_list = []
# fnrs_list = []
# fracs_list =  []

# for i in range(iters):
#     fprs = data[i*batch_size:(i+1)*batch_size, 0]
#     fnrs = data[i*batch_size:(i+1)*batch_size, 1]
#     fracs = data[i*batch_size:(i+1)*batch_size, 2]

#     fprs_mean = np.mean(fprs)
#     fprs_std = np.std(fprs)
#     fprs_025 = np.quantile(fprs, 0.025)
#     fprs_975 = np.quantile(fprs, 0.975)
#     fnrs_mean = np.mean(fnrs)
#     fnrs_std = np.std(fnrs)
#     fnrs_025 = np.quantile(fnrs, 0.025)
#     fnrs_975 = np.quantile(fnrs, 0.975)
#     fracs_mean = np.mean(fracs)
#     fracs_std = np.std(fracs)
#     fracs_025 = np.quantile(fracs, 0.025)
#     fracs_975 = np.quantile(fracs, 0.975)

#     # print('CP fprs 0.025', fprs_025)
#     # print('CP fprs 0.975', fprs_975)
#     # print('CP fnrs 0.025', fnrs_025)
#     # print('CP fnrs 0.975', fnrs_975)
#     # print('CP fracs 0.025', fracs_025)
#     # print('CP fracs 0.975', fracs_975)

#     # pdb.set_trace()

#     fprs = np.sum(fprs<epsilon) / batch_size
#     fnrs = np.sum(fnrs<epsilon) / batch_size
#     fracs = np.mean(fracs)

#     fprs_list.append(fprs)
#     fnrs_list.append(fnrs)
#     fracs_list.append(fracs)

# fprs = np.array(fprs_list).reshape([1, iters])
# fnrs = np.array(fnrs_list).reshape([1, iters])
# fracs = np.array(fracs_list).reshape([1, iters])

# results = np.vstack([fprs,
#                      fnrs,
#                      fracs])

# means = np.mean(results, axis=1)
# stds = np.std(results, axis=1)

# CI = 1.96 * np.sqrt(means*(1-means)/iters) 
# low = means - CI
# high = means + CI
# print('CP Low bound', low)
# print('CP High bound', high)
# print('CP CI', CI)

# # print('detector:', detector)
# print('CP means:', 1-means)
# print('CP stds:', stds)

# means_bion_upper = means + np.sqrt(means*(1-means) / trials) * 2.575
# means_bion_lower = means - np.sqrt(means*(1-means) / trials) * 2.575

# print('upper', means_bion_upper)
# print('lower', means_bion_lower)

method = 'PAC'

data = np.load(f'{method}.npy', allow_pickle=True)[:trials, :]
total = 300
successes = data > epsilon
rates = ['fpr', 'fnr', 'frac']
for i in range(3):
    lo, hi = clopper_pearson(np.sum(successes[:, i]), total)
    print('PAC {:s} 95% confidence interval: {:.3f}-{:.3f}'.format(rates[i], lo, hi))
print('PAC average FPR', np.mean(data[:, 0]))
print('PAC average FNR', np.mean(data[:, 1]))
print('PAC average frac', np.mean(data[:, 2]))
# fprs_list = []
# fnrs_list = []
# fracs_list =  []

# for i in range(iters):
#     fprs = data[i*batch_size:(i+1)*batch_size, 0]
#     fnrs = data[i*batch_size:(i+1)*batch_size, 1]
#     fracs = data[i*batch_size:(i+1)*batch_size, 2]

#     fprs_mean = np.mean(fprs)
#     fprs_std = np.std(fprs)
#     fprs_025 = np.quantile(fprs, 0.025)
#     fprs_975 = np.quantile(fprs, 0.975)
#     fnrs_mean = np.mean(fnrs)
#     fnrs_std = np.std(fnrs)
#     fnrs_025 = np.quantile(fnrs, 0.025)
#     fnrs_975 = np.quantile(fnrs, 0.975)
#     fracs_mean = np.mean(fracs)
#     fracs_std = np.std(fracs)
#     fracs_025 = np.quantile(fracs, 0.025)
#     fracs_975 = np.quantile(fracs, 0.975)

#     # print('PAC fprs 0.025', fprs_025)
#     # print('PAC fprs 0.975', fprs_975)
#     # print('PAC fnrs 0.025', fnrs_025)
#     # print('PAC fnrs 0.975', fnrs_975)
#     # print('PAC fracs 0.025', fracs_025)
#     # print('PAC fracs 0.975', fracs_975)

#     # pdb.set_trace()

#     fprs = np.sum(fprs<epsilon) / batch_size
#     fnrs = np.sum(fnrs<epsilon) / batch_size
#     fracs = np.mean(fracs)

#     fprs_list.append(fprs)
#     fnrs_list.append(fnrs)
#     fracs_list.append(fracs)

# fprs = np.array(fprs_list).reshape([1, iters])
# fnrs = np.array(fnrs_list).reshape([1, iters])
# fracs = np.array(fracs_list).reshape([1, iters])

# results = np.vstack([fprs,
#                      fnrs,
#                      fracs])

# means = np.mean(results, axis=1)
# stds = np.std(results, axis=1)

# CI = 1.96 * np.sqrt(means*(1-means)/iters) 
# low = means - CI
# high = means + CI
# print('PAC Low bound', low)
# print('PAC High bound', high)
# print('PAC CI', CI)

# # print('detector:', detector)
# print('PAC means:', 1-means)
# print('PAC stds:', stds)