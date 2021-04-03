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
            
    def __call__(self, X_test, y_true, model, network_weights_name, clasfier, test_steps, **kwargs):
        #TODOC
        #model = self.__models[model_name]
        label_name = kwargs['label_name']
        network = kwargs['network']
        results_path = kwargs['results_path']
        y_pred_test = []
        y_pred_round = []
        y_true = []
        
        if type(X_test)==list:
            y_pred_test.extend(model.predict(X_test))
            y_pred_round.extend(_np.round(y_pred_test))
        else:
            for x,y in X_test.take(test_steps):
                prediction = model.predict(x)
                y_pred_test.extend(prediction)
                y_pred_round.extend(_np.round(prediction))
                y_true.extend(y)
                
        
        f1_macro = f1_score(y_true, y_pred_round,average="macro")
        f1_micro = f1_score(y_true, y_pred_round,average="micro")
     
        con_mat = confusion_matrix(y_true, y_pred_round)
        tp = con_mat[1][1]
        tn = con_mat[0][0]
        fp = con_mat[0][1]
        fn = con_mat[1][0]
        recall = tp/(tp+fn)
        specificity = tn/(tn+fp)
        #precision=tp/(tp+fp)
        acc = (tp+tn)/(tp+tn+fp+fn)
        
        fpr, tpr, threshold = roc_curve(y_true, y_pred_test)
        auc = roc_auc_score(y_true, y_pred_test)
        
        df = _pd.DataFrame([[network, clasfier, auc, recall, specificity, acc, f1_macro, f1_micro, network_weights_name, kwargs['optimizer'], kwargs['dropout'], kwargs['learning_rate'], kwargs['epochs']]])
    
        if not _os.path.exists('{}'.format(results_path + label_name + '/csv')):
            _os.makedirs(results_path)
        
        
        with open('{}/{}_results.csv'.format(results_path + label_name + '/csv', label_name), 'a') as f:
            if f.tell() == 0:
               writer = csv.writer(f)
               writer.writerow(['network', 'classifier', 'auc', 'recall', 'specificity', 'acc', 'f1_macro', 'f1_micro', 'network_weights_name', 'optimizer', 'dropout', 'train_lr', 'train_epochs'])
    
    
            #df.to_csv(f, header =['network', 'classifier', 'auc', 'recall', 'specificity', 'acc', 'f1_macro', 'f1_micro', 'network_weights_name', 'warmup_lr', 'warmup_epochs', 'warmup_decay', 'train_lr', 'train_epochs', 'train_decay'])
            df.to_csv(f, header = False, index=False)
     
