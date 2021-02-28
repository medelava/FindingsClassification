#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  8 14:14:05 2021

@author: mder
"""
import tensorflow as _tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from copy import deepcopy as _deepcopy
import h5py
from PIL import Image
import numpy as np


def augmentation_func(params):
    #TODOC
    params = _deepcopy(params)
    params["augmentation"]["preprocessing_function"] = lambda i: i
    return params 


def calculate_steps(**kwargs):
    """
    Calculate steps for trianing, validation and test
    
    Parameters
    ----------
    kwargs['path_train'] : str
         path of hdf5 file with training images.
    kwargs['path_val'] : str
         path of hdf5 file with validation images.
    kwargs['path_test'] : str
         path of hdf5 file with test images.
         
    Returns
    -------
    number of steps for training, validation and test
    """
    
    with h5py.File(kwargs['path_train'], 'r') as df:
        num_train = df[kwargs['label_name']][:].shape[0]
        
    with h5py.File(kwargs['path_val'], 'r') as df:
        num_val = df[kwargs['label_name']][:].shape[0]
        
    with h5py.File(kwargs['path_test'], 'r') as df:
        num_test = df[kwargs['label_name']][:].shape[0]
        
    return num_train//kwargs['batch_size'], num_val//kwargs['batch_size'], num_test//kwargs['batch_size']



def callbacks(model_path, file_name, callback_params): 
    """
    Returns two callbacks, model checlpoint and eaerlystopping, to be used during training. 
    
    Parameters
    ----------
        model_path : str
            path to save the trained model weights.
        filename : str
            name of the .hdf5 weights file.
            
    Returns
    -------
    checkpoint
        callback to save the best model based on the specified monitor.
    early stop
        earlystopping callback.
    """
    
    checkpoint = ModelCheckpoint(model_path + file_name, 
                                 monitor = callback_params['monitor'], 
                                 verbose = callback_params['verbose'], 
                                 save_best_only = callback_params['save_best_only'], 
                                 mode = callback_params['mode'])

    early_stop = EarlyStopping(monitor = callback_params['monitor'], 
                               min_delta = callback_params['min_delta'], 
                               patience = callback_params['patience'], 
                               verbose = callback_params['verbose'], 
                               mode = callback_params['mode'], 
                               restore_best_weights = callback_params['restore_best_weights'])
    return checkpoint, early_stop




def features_extraction(model, train_ds, num_img, batch_size):
    # tf.keras.models.Model(inputs=model.input, outputs=model.get_layer('inception_v3').layers[-1].output)
    #model_notop = _tf.keras.models.Model(inputs=model.input, outputs=model.layers[-4].get_output_at(0))
    features = []
    labels = []
    N = np.round(num_img/batch_size)
    for X_batch, y_batch in train_ds.take(N):
        batch_features = model.predict(X_batch)
        features.extend(batch_features)
        labels.extend(y_batch.numpy())
    return features, labels
        
     
def image_from_generator(dataset):  
    """
    Shows an image from the tf dataset. 
    
    Parameters
    ----------
    dataset : tf data dataset
        tf data dataset of images.
            
    Returns
    -------
    shows an image stored in the dataset.
    """
    
    img = next(iter(dataset))
    imagen=img[0].numpy()
    formatted = (imagen * 255 / np.max(imagen)).astype('uint8')
    img = Image.fromarray(formatted[0], 'RGB')
    img.show()



