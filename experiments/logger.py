#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 26 18:45:50 2021

@author: mder
"""

import pandas as _pd
import os as _os
import numpy as _np
import matplotlib.pyplot as _plt
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix, f1_score 
import csv



class Logger:
    #TODOC
    def __init__(
            self,
            dataset_name = 'datasetname', 
            evaluable_datasets = [
                "messidor2"
                ],
            metric_names = ['network', 'classifier', 'auc', 'recall', 'specificity', 
                            'acc', 'f1_macro', 'f1_micro', 'network_weights_name', 
                            'warmup_lr', 'warmup_epochs', 'warmup_decay', 'train_lr', 
                            'train_epochs', 'train_decay']
            ):
        
        self.__dataset_name = dataset_name
        self.__evaluable_datasets = evaluable_datasets
        self.__metric_names = metric_names
        self.__make_dataframes()
        
    
    #def add(self, models, metrics, datasets):
   #     self.__models = models
   #     self.__metrics = metrics
   #     self.__datasets = datasets

    def __make_dataframes(self):
        #TODOC
        self.__dfs = {}
        for dataset in self.__evaluable_datasets:
            self.__dfs[dataset] = _pd.DataFrame(
                    columns=[
                        *self.__metric_names
                        ]
                    )
            
    def __call__(
        self, X_test, y_test, model, network_weights_name, clasfier, **kwargs
        ):
        #TODOC
        #model = self.__models[model_name]
        label_name = kwargs['label_name']
        network = kwargs['network']
        results_path = kwargs['results_path']
       
        y_pred_test = model.predict(X_test)
        y_test_round = _np.round(y_pred_test)
        
        f1_macro = f1_score(y_test, y_test_round,average="macro")
        f1_micro = f1_score(y_test, y_test_round,average="micro")
     
        con_mat = confusion_matrix(y_test,y_test_round)
        tp = con_mat[1][1]
        tn = con_mat[0][0]
        fp = con_mat[0][1]
        fn = con_mat[1][0]
        recall = tp/(tp+fn)
        specificity = tn/(tn+fp)
        #precision=tp/(tp+fp)
        acc = (tp+tn)/(tp+tn+fp+fn)
        
        fpr, tpr, threshold = roc_curve(y_test, y_pred_test)
        auc = roc_auc_score(y_test, y_pred_test)
        
        df = _pd.DataFrame([[network, clasfier, auc, recall, specificity, acc, f1_macro, f1_micro, network_weights_name, kwargs['warmup_lr'], kwargs['warmup_epochs'], kwargs['warmup_decay'], kwargs['learning_rate'], kwargs['epochs'], kwargs['decay'],]])
    
        if not _os.path.exists('{}'.format(results_path + label_name + '/csv')):
            _os.makedirs(results_path)
        
        
        with open('{}/{}_{}_results.csv'.format(results_path + label_name + '/csv', label_name, network), 'a') as f:
            if f.tell() == 0:
               writer = csv.writer(f)
               writer.writerow(['network', 'classifier', 'auc', 'recall', 'specificity', 'acc', 'f1_macro', 'f1_micro', 'network_weights_name', 'warmup_lr', 'warmup_epochs', 'warmup_decay', 'train_lr', 'train_epochs', 'train_decay'])
    
    
            #df.to_csv(f, header =['network', 'classifier', 'auc', 'recall', 'specificity', 'acc', 'f1_macro', 'f1_micro', 'network_weights_name', 'warmup_lr', 'warmup_epochs', 'warmup_decay', 'train_lr', 'train_epochs', 'train_decay'])
            df.to_csv(f, header = False)
     

        _plt.plot(fpr, tpr, 'b', label = 'AUC = %0.4f' % auc)
        _plt.legend(loc='lower right')
        _plt.plot([0, 1], [0, 1],'r--')
        _plt.xlim([0, 1])
        _plt.ylim([0, 1])
        _plt.ylabel('True Positive Rate', fontsize = 20.0)
        _plt.xlabel('False Positive Rate', fontsize = 20.0)
        _plt.xticks(fontsize=15, rotation=0)
        _plt.yticks(fontsize=15, rotation=0)

        _plt.savefig('{}/{}/AUC_{}_{}_{}_{}.png'.format(results_path + label_name, 'images', auc, network, clasfier, network_weights_name))  

            
    