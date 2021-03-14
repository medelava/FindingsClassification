# importar clase generator y clase convnnet
#import generator, convnet, logger
#%% 
import argparse
import json 
import numpy as np

import tensorflow as tf
#%% 
from retfindings.datasets.generator import make_generator
from retfindings.models.convnet import CNNFindings
from logger import Logger
from utils import augmentation_func, calculate_steps, callbacks, features_extraction, image_from_generator
from inspect import getmembers, isfunction
from sklearn.model_selection import GridSearchCV
from sklearn import preprocessing
import classifiers
from sklearn.gaussian_process.kernels import RBF as _RBF
from sklearn.gaussian_process.kernels import WhiteKernel as _WhiteKernel
from sklearn.gaussian_process.kernels import Matern as _Matern
#%% 
def make_model(**kwargs):
    
    """
    A model is build using one of the following options:
        inceptionV3, resnet50, efficientnetb5 or xception.
    A top of a dense of 1024 neurons, a dropout and a output dense is added.
    Finally the model is compiled.
    
    Parameters
    ----------
    network : str
        name of the network to use as base model.
    learning_rate : float
        learning rate used to compile the model.
    decay : float
        decay used to compile the model.
    dropout : float
        dropout used in the top model.
    img_shape : list
        shape of the imput images of the model.
    outputs : int
        number of outputs in the last dense.
    epochs : int
        number of epochs used to compile the model.
    train_mode : str
        'tl' if transfer learning is going to be performed.'ft' if fine tuning is going to be performed.
                      
    Returns
    -------
        model : keras model
            compiled keras model.
    """
   
    img_shape = kwargs["img_shape"]
    network = kwargs['network']
    

    if network == 'inceptionv3':
        base_model = tf.keras.applications.InceptionV3(weights = 'imagenet', 
                                                       include_top = False,
                                                       input_shape = img_shape,
                                                       pooling = 'avg'
                                                       )
    elif network == 'resnet50':
        base_model = tf.keras.applications.ResNet50(weights = "imagenet",
                                                    include_top = True,
                                                    input_shape = img_shape,
                                                    pooling = 'avg'
                                                    )

    elif network == 'efficientnetb5': 
        base_model = tf.keras.applications.EfficientNetB5(weights = "imagenet",
                                                          include_top = False,
                                                          input_shape = img_shape,
                                                          pooling = 'avg'
                                                          )

    elif network == 'xception':
        base_model = tf.keras.applications.Xception(weights="imagenet",
                                                    include_top=True,
                                                    input_shape=img_shape,
                                                    pooling='avg'
                                                    )
    input_tensor = tf.keras.Input(shape = img_shape)
    model = CNNFindings(base_model, kwargs["outputs"])
    predictions = model(input_tensor) 
    features = model.features_extraction(input_tensor)
    #model(img_shape)
    
    #return model
    return {'full_model': model, 
            'features_extractor': tf.keras.Model(inputs = [input_tensor], outputs = [features])}
    #return {'full_model': tf.keras.Model(inputs = [input_tensor], outputs = [predictions]), 
    #        'features_extractor': }


def make_datasets(augmentation, **kwargs):
    """
    Builds tf dataset for training, validation and test partitions from .hdf5 files.
    
    Parameters
    ----------
    path_train : str
        path of the .hdf5 file with the training images.
    path_val : str
        path of the .hdf5 file with the validation images.
    path_test : str
        path of the .hdf5 file with the test images.
    batch_size : int
        number of images per batch produces by the generator and save in the tf dataset.
    label_name : str
        name of the key for the lables to be used during training.
    sampling : str
        king of samplig to be used in the generator. The options are: 'rand', 'oversampling', 'batch'.
    augmentation : dict
        dictionary with the augmentation parametest to be used during training.
    img_shape : list
        shape of the mages to be used during training.
        
    Returns
    -------
    tran_ds : dataset
        tf dataset for the training examples.
    val_ds : dataset
        tf dataset for the validation examples.
    test_ds : dataset
        tf dataset for the test exafeatures(model, train_ds, num_img, batch_size)mples.
    """
    
    
    path_train = kwargs["path_train"]
    path_val = kwargs["path_val"]
    path_test = kwargs["path_test"]
    batch_size = kwargs["batch_size"] 
    label_name = kwargs["label_name"] 
    sampling = kwargs["sampling"]
    img_shape = kwargs["img_shape"]
    
    train_ds = make_generator(path_train, batch_size, label_name, sampling, augmentation, img_shape, 'train')
    val_ds = make_generator(path_val, batch_size, label_name, sampling, augmentation, img_shape, 'validation')
    test_ds = make_generator(path_test, batch_size, label_name, sampling, augmentation, img_shape, 'test')
    return train_ds, val_ds, test_ds
   

# guardar curva con logger



def transfer_learning(model, train_generator, val_generator, train_steps, 
                      val_steps, callback_params, **kwargs):
    """
    Trains the top model, keepping the base model layers freezed, which are only used as feature extractors. 
    
    Parameters
    ----------
    model : keras model
        compiled model that has a standard convolutional neural network and specified top fully connected classifier.
    trian_generator : generator
        td data generator for trian images each sample batch size is fixed in make_datasets.
    val_generator : generator
        td data generator for validation images each sample batch size is fixed in make_datasets.
    trian_steps : int
        number of steps during training: num_img/batch_size.
    val_steps : int
        number of steps during validation: num_img/batch_size.
    model_path : str
        path to save the transfer learning trained model.
     kwargs['label_name'] : str
        name to save the trained model: label_tl_randomNumber.
    kwargs['epochs'] : int
        number of epochs to train the model.
        
    Returns
    -------
    model : keras model
        trained model.
    history
        information of the training process.
    """
    
    # model.layers[1].freeze()
    model.freeze()
    
    file_name = '{}_tl_{}.hdf5'. format(kwargs['label_name'], np.random.random())
    checkpoint, early_stop = callbacks(kwargs['model_path'], file_name, callback_params)    


    optimizer = tf.keras.optimizers.RMSprop(learning_rate = kwargs["learning_rate"],
                                            decay = kwargs["decay"])
    
    model.compile(
        loss = tf.keras.losses.BinaryCrossentropy(),
        optimizer = optimizer,
        metrics = ['accuracy', tf.keras.metrics.AUC()]
    )
     
    print(model.input)
    
    
    history = model.fit(
        train_generator,
        steps_per_epoch = train_steps,
        epochs = kwargs['epochs'], 
        validation_data = val_generator,
        validation_steps = val_steps,
        callbacks = [checkpoint, early_stop],
        #class_weight=d_class_weights
    )
    
    return model, history, file_name


def fine_tuning(model, train_generator, val_generator, train_steps, 
                val_steps, callback_params, **kwargs):
    """
    Trains the top model, keepping the base model layers freezed, which are only used as feature extractors. 
    
    Parameters
    ----------
    model : keras model
        compiled model that has a standard convolutional neural network and specified top fully connected classifier.
    trian_generator : generator
        td data generator for trian images each sample batch size is fixed in make_datasets.
    val_generator : generator
        td data generator for validation images each sample batch size is fixed in make_datasets.
    trian_steps : int
        number of steps during training: num_img/batch_size.
    val_steps : int
        number of steps during validation: num_img/batch_size.
    model_path : str
        path to save the fine tuned model.
     kwargs['label_name'] : str
        name to save the trained model: label_tl_randomNumber.
    kwargs['epochs'] : int
        number of epochs to train the model.
        
    Returns
    -------
    model : keras model
        trained model.
    history
        information of the training process.
    """
    
    if kwargs['ft_mode'] == 'all':
        #model.layers[1].unfreeze()
        model.unfreeze()
    else:
        for layer in model.layers[0].layers[:kwargs['num_non_trainable']]:
            layer.trainable = False
    
    file_name = '{}_ft_{}.hdf5'. format(kwargs['label_name'], np.random.random())
    checkpoint, early_stop = callbacks(kwargs['model_path'], file_name, callback_params)    
    
    optimizer = tf.keras.optimizers.RMSprop(learning_rate = kwargs["learning_rate"],
                                            decay = kwargs["decay"])
    
    model.compile(
        loss = tf.keras.losses.BinaryCrossentropy(),
        optimizer = optimizer,
        metrics = ['accuracy', tf.keras.metrics.AUC()]
    )
         
    
    history = model.fit(
        train_generator,
        steps_per_epoch = train_steps,
        epochs = kwargs['epochs'], 
        validation_data = val_generator,
        validation_steps = val_steps,
        callbacks = [checkpoint, early_stop],
        #class_weight=d_class_weights
    )
    
    return model, history, file_name



if __name__=='__main__':   
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_info", type=str, default="data_info.json")
    parser.add_argument("--hyperparams", type=str, default="hyperparams.json")
    parser.add_argument("--callbacks", type=str, default="callbacks.json")
    args = parser.parse_args()
    
    # read parameters from json
    with open(args.callbacks, "r") as f:
        callback_params = json.load(f)

    with open(args.data_info, "r") as f:
        data_info = json.load(f)
        
    with open(args.hyperparams, "r") as f:
        hyperparams = json.load(f)
    
    hyperparams = augmentation_func(hyperparams)

    # model definition
    models = make_model(**data_info, **hyperparams)

    train_ds, val_ds, test_ds = make_datasets(augmentation=hyperparams["augmentation"], **data_info)
     
    image_from_generator(train_ds)
  
    
    
    
    train_steps, val_steps, test_steps = calculate_steps(**data_info)
    
    
    # train the model using transfer learning 'tf' or fine tunning 'ft'
    if hyperparams["train_mode"] == 'tl':
        model, history, file_name = transfer_learning(models['full_model'], 
                                           train_ds, 
                                           val_ds, 
                                           train_steps, 
                                           val_steps,
                                           callback_params,
                                           **hyperparams,
                                           **data_info)
        
        
    else:
        
        model, history, file_name = fine_tuning(models['full_model'], 
                                    train_ds, 
                                    val_ds, 
                                    train_steps, 
                                    val_steps,
                                    callback_params,
                                    **hyperparams,
                                    **data_info)



    num_img = 50
    batch_size=3
    
    train_features, train_labels = features_extraction(models['features_extractor'], train_ds, num_img, batch_size)
    val_features, val_labels = features_extraction(models['features_extractor'], train_ds, num_img, batch_size)
    
    train_features = train_features + val_features
    train_labels = train_labels + val_labels
    test_features, test_labels = features_extraction(models['features_extractor'], train_ds, num_img, batch_size)
    
    scaler = preprocessing.StandardScaler().fit(train_features)
    
    train_scaled = scaler.transform(train_features)
    test_scaled = scaler.transform(test_features)
    
    models_names = getmembers(classifiers, isfunction)

#%%     
    
    lower_ls = hyperparams["lower length_scale bound"]
    upper_ls = hyperparams["upper length_scale bound"]
    lower_nl = hyperparams["lower noise_level"]
    upper_nl = hyperparams["upper noise_level"]
    
    ls_samples = np.random.uniform(
        lower_ls, upper_ls, hyperparams["num_samples"])
    nl_samples = np.random.uniform(
        lower_nl, upper_nl, hyperparams["num_samples"])
    
    ls_nl = np.vstack((ls_samples, nl_samples)).T
    scores = ["roc_auc"]
    
    param_grid = [{
                    "kernel": [
                        1.0 * _RBF(
                            length_scale=i, 
                            length_scale_bounds=(lower_ls, upper_ls)) 
                        + _WhiteKernel(noise_level=j, 
                           noise_level_bounds=(lower_nl, upper_nl)) for i,j in ls_nl]
                        
                }, {
                    "kernel": [1.0 * _Matern(length_scale=i, 
                                             length_scale_bounds=(lower_ls, upper_ls)) \
                                   + _WhiteKernel(noise_level=j, 
                                                  noise_level_bounds=(lower_nl, upper_nl)) for i,j in ls_nl]
                }]
    
#%%     
    for name, make_clf in models_names[:1]:
        classif = make_clf()
        
        clf = GridSearchCV(estimator=classif, param_grid=param_grid, cv=2,
                       scoring="roc_auc")
        clf.fit(train_scaled, train_labels)
        classif  = clf.best_estimator_     
        
        #classif = function().fit(train_scaled, train_labels)
        save_results = Logger('datasetname')
        save_results(X_test=test_scaled, y_test=test_labels, model=classif, 
                     network_weights_name = file_name, clasfier=name, **data_info,
                     **hyperparams)
        
        
        
        
        
        
        
        #save_metrics(test_features, test_labels, model=classif, network_weights_name = file_name, clasfier=name, **data_info, **hyperparams)
        #network, model, results_path, test_hdf5, finding, warmup_lr, warmup_epochs, warmup_decay, train_lr, train_epochs, train_decay, history
     
        
    #gp = GP_rbf_classifier(features, labels)
    #svm= smv(features, labels)
    #logger.save_results(model)



# regresion logistica se puede sacar incertidumbre seria desvacione standar de la prediccion, mirar si es igual en GP
# explicar whitekernel y como funciona GP.
# gradient boosting, random forest, SVM, logistic regression, capas densas. 
# funcion de orden superior
# pep 8, pypfleig
# uuid4 para gusrdar modelos
# parametros del optimizador
# hyperparams["augmentation"]["preprocessing_function"](next(iter(train_ds))[0]).numpy().min()




#"tf.keras.applications.inception_v3.preprocess_input()"


