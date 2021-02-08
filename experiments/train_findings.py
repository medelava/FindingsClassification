# importar clase generator y clase convnnet
#import generator, convnet, logger
import argparse
import json 
import numpy as np
#
import tensorflow as tf
import h5py
from PIL import Image
from retfindings.datasets.generator import make_generator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from retfindings.models.convnet import CNNFindings



def make_model(network, learning_rate, decay, dropout, img_shape, outputs, epochs, train_mode):
    
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

    model = CNNFindings(base_model, outputs)
    input_tensor=tf.keras.Input(shape=img_shape)
    model(input_tensor)
    
    
    optimizer= tf.keras.optimizers.RMSprop(learning_rate=learning_rate, decay=decay)

    model.compile(
        loss=tf.keras.losses.BinaryCrossentropy(),
        optimizer=optimizer,
        metrics=['accuracy', tf.keras.metrics.AUC()]
    )
    
    return model


def make_datasets(path_train, path_val, path_test, batch_size, label_name, sampling, augmentation, img_shape):
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
        tf dataset for the test examples.
    """
    
    
    train_ds = make_generator(path_train, batch_size, label_name, sampling, augmentation, img_shape, 'train')
    val_ds = make_generator(path_val, batch_size, label_name, sampling, augmentation, img_shape, 'validation')
    test_ds = make_generator(path_test, batch_size, label_name, sampling, augmentation, img_shape, 'test')
    return train_ds, val_ds, test_ds
   

# guardar curva con logger

def callbacks(model_path, file_name): 
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
                                 monitor = 'val_loss', 
                                 verbose = 1, 
                                 save_best_only = True, 
                                 mode = 'auto')

    early_stop = EarlyStopping(monitor = 'val_loss', 
                               min_delta = 0, 
                               patience = 10, 
                               verbose = 1, 
                               mode = 'auto', 
                               baseline = None, 
                               restore_best_weights = True)
    return checkpoint, early_stop

    

def transfer_learning(model, train_generator, val_generator, train_steps, 
                      val_steps, model_path,  **kwargs):
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
    
    model.freeze()
    
    file_name = '{}_tl_{}.hdf5'. format(kwargs['label_name'], np.random.random())
    checkpoint, early_stop = callbacks(model_path, file_name)    
    
    history = model.fit(
        train_generator,
        steps_per_epoch = train_steps,
        epochs = kwargs['epochs'], 
        validation_data = val_generator,
        validation_steps = val_steps,
        callbacks = [checkpoint, early_stop],
        #class_weight=d_class_weights
    )
    
    return model, history


def fine_tuning(model, train_generator, val_generator, train_steps, 
                val_steps, model_path, **kwargs):
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
    
    model.unfreeze()
    
    file_name = '{}_ft_{}.hdf5'. format(kwargs['label_name'], np.random.random())
    checkpoint, early_stop = callbacks(model_path, file_name)    
    
    history = model.fit(
        train_generator,
        steps_per_epoch = train_steps,
        epochs = kwargs['epochs'], 
        validation_data = val_generator,
        validation_steps = val_steps,
        callbacks = [checkpoint, early_stop],
        #class_weight=d_class_weights
    )
    
    return model, history


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



if __name__=='__main__':
    # parameters not included in json
    model_path = "/home/mder/repositories/FindingsClassification/retfindings/models/"
    augmentation = {"preprocessing_function": lambda i:i, 
                    "apply":False,
                    "random_brightness": {"max_delta": 0.5},
                    "random_contrast": {"lower":0.6, "upper":1},
                    "random_hue": {"max_delta": 0.1},
                    "random_saturation": {"lower": 0.6, "upper":1},
                    "random_rotation": {"minval": 0, "maxval": 2*np.pi},
                    "horizontal_flip": True, "vertical_flip": True,
                    "rotation_range": {"minval": -0.3, "maxval": 0.3},
                    "width_shift_range": {"minval": -2, "maxval": 2},
                    "height_shift_range": {"minval": -2, "maxval": 2},}
    
    
    
    # argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_info", type=str, default="data_info.json")
    parser.add_argument("--hyperparams", type=str, default="hyperparams.json")
    args = parser.parse_args()
    
 
    # read parameters from json
    with open(args.data_info, "r") as f:
        data_info = json.load(f)
        
    with open(args.hyperparams, "r") as f:
        hyperparams = json.load(f)
   
        
    print('BBBatchsize', data_info['batch_size'])
   
    
   
    # model definition
    model = make_model(img_shape = data_info['img_shape'], **hyperparams)
    train_ds, val_ds, test_ds = make_datasets(**data_info, augmentation=augmentation)
    image_from_generator(train_ds)
    train_steps, val_steps, test_steps = calculate_steps(**data_info)
    
    # train the model using transfer learning 'tf' or fine tunning 'ft'
    if hyperparams["train_mode"] == 'tf':
        model, history = transfer_learning(model, 
                                           train_ds, 
                                           val_ds, 
                                           train_steps, 
                                           val_steps,
                                           model_path,
                                           **hyperparams,
                                           **data_info)
    else:
        model, history = fine_tuning(model, 
                                    train_ds, 
                                    val_ds, 
                                    train_steps, 
                                    val_steps,
                                    model_path,
                                    **hyperparams,
                                    **data_info)

    #logger.save_results(model)

















