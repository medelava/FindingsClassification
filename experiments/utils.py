import tensorflow as _tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from copy import deepcopy as _deepcopy
import h5py
#from PIL import Image
import numpy as np
from retfindings.datasets.generator import make_generator


def preprocess_network(params):
    params = _deepcopy(params)
    network = params["network"]
    if network == 'inceptionv3':
       params["augmentation"]["preprocessing_function"] = _tf.keras.applications.inception_v3.preprocess_input
    elif network == 'resnet50':
       params["augmentation"]["preprocessing_function"] = _tf.keras.applications.resnet.preprocess_input
    elif network == 'efficientnetb5': 
        params["augmentation"]["preprocessing_function"] = _tf.keras.applications.efficientnet.preprocess_input
    elif network == 'xception':
       params["augmentation"]["preprocessing_function"] = _tf.keras.applications.xception.preprocess_input
    else:
       raise NameError('no preprocessing function for this network')
    return params 


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
        tf dataset for the test exafeatures(model, train_ds, num_img, batch_size samples.
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
    from PIL import Image
    img = next(iter(dataset))
    imagen=img[0].numpy()
    print(imagen.max(), imagen.min())
    #formatted = (imagen * 255 / np.max(imagen)).astype('uint8')
    img = Image.fromarray(imagen.astype('uint8'), 'RGB')
    img.show()



