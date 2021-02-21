#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 10 17:21:39 2021

@author: mder
"""

import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix, f1_score


def save_metrics(X_test, y_test, model, network_weights_name, clasfier, **kwargs):

    label_name = kwargs['label_name']
    network = kwargs['network']
    results_path = kwargs['results_path']
   
    y_pred_test = model.predict(X_test)
    y_test_round = np.round(y_pred_test)
    
    f1_macro = f1_score(y_test, y_test_round,average="macro")
    f1_micro = f1_score(y_test, y_test_round,average="micro")
 
    con_mat=confusion_matrix(y_test,y_test_round)
    tp=con_mat[1][1]
    tn=con_mat[0][0]
    fp=con_mat[0][1]
    fn=con_mat[1][0]
    recall=tp/(tp+fn)
    specificity=tn/(tn+fp)
    #precision=tp/(tp+fp)
    acc=(tp+tn)/(tp+tn+fp+fn)
    
    fpr, tpr, threshold=roc_curve(y_test, y_pred_test)
    auc=roc_auc_score(y_test, y_pred_test)
    
    df = pd.DataFrame([[network, clasfier, auc, recall, specificity, acc, f1_macro, f1_micro, network_weights_name, kwargs['warmup_lr'], kwargs['warmup_epochs'], kwargs['warmup_decay'], kwargs['learning_rate'], kwargs['epochs'], kwargs['decay'],]])
        
    with open('{}/{}_{}_results.csv'.format(results_path + label_name + '/csv', label_name, network), 'a') as f:
        df.to_csv(f, header =['network', 'classifier', 'auc', 'recall', 'specificity', 'acc', 'f1_macro', 'f1_micro', 'network_weights_name', 'warmup_lr', 'warmup_epochs', 'warmup_decay', 'train_lr', 'train_epochs', 'train_decay',])
 
    plt.plot(fpr, tpr, 'b', label = 'AUC = %0.4f' % auc)
    plt.legend(loc='lower right')
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate', fontsize = 20.0)
    plt.xlabel('False Positive Rate', fontsize = 20.0)
    plt.xticks(fontsize=15, rotation=0)
    plt.yticks(fontsize=15, rotation=0)

    plt.savefig('{}/{}/AUC_{}_{}_{}_{}.png'.format(results_path + label_name, 'images', auc, network, clasfier, network_weights_name))  

def save_train_history():
    pass





















