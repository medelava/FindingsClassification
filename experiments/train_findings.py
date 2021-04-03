import argparse
import json
import uuid
import tensorflow as tf
from retfindings.models.convnet import compile_model, make_model
from logger import   Logger
from utils import preprocess_network, calculate_steps, callbacks, image_from_generator, make_datasets

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
    
    if kwargs["train_mode"] == 'tl':
        epochs = kwargs['epochs']
    else:
        epochs = kwargs['epochs_warmup']
    
    model.freeze()

    file_uuid = uuid.uuid4()
    file_name = '{}_tl_{}.hdf5'. format(kwargs['label_name'], file_uuid)
    checkpoint, early_stop = callbacks(kwargs['model_path'], file_name, callback_params)    

    if kwargs['optimizer'] == 'adam':
        optimizer = tf.keras.optimizers.Adam(learning_rate=kwargs["learning_rate"], 
                                             beta_1=kwargs["beta_1"],
                                             beta_2=kwargs["beta_2"], 
                                             epsilon=kwargs["epsilon"])
    elif kwargs['optimizer'] == 'sgd':
        optimizer =tf.keras.optimizers.SGD(learning_rate=kwargs["learning_rate"],
                                           momentum=kwargs["momentum"])
    
    model = compile_model(model, optimizer)

    history = model.fit(
        train_generator,
        steps_per_epoch = train_steps,
        epochs = epochs, 
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
    
    
    file_uuid = uuid.uuid4()
    file_name = '{}_ft_{}.hdf5'. format(kwargs['label_name'], file_uuid)
    checkpoint, early_stop = callbacks(kwargs['model_path'], file_name, callback_params)    
   
    optimizer = tf.keras.optimizers.Adam(learning_rate=kwargs["learning_rate"], 
                                         beta_1=kwargs["beta_1"],
                                         beta_2=kwargs["beta_2"], 
                                         epsilon=kwargs["epsilon"])
    
    model = compile_model(model, optimizer) 
    
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
    parser.add_argument("--data_info", type=str, default="./parameters/data_info.json")
    parser.add_argument("--hyperparams", type=str, default="./parameters/hyperparams.json")
    parser.add_argument("--callbacks", type=str, default="./parameters/callbacks.json")
    args = parser.parse_args()
    
    # read parameters from json
    with open(args.callbacks, "r") as f:
        callback_params = json.load(f)

    with open(args.data_info, "r") as f:
        data_info = json.load(f)
        
    with open(args.hyperparams, "r") as f:
        hyperparams = json.load(f)
   
    hyperparams = preprocess_network(hyperparams)
    models = make_model(**data_info, **hyperparams)     # model definition
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
        
        save_results = Logger(dataset_name='messidor2')
        save_results(X_test=test_ds, y_true=[], model=model, 
                     network_weights_name = file_name, clasfier='multilayer perceptron', test_steps=test_steps, **data_info,
                     **hyperparams)
    else:
        model, history, file_name_warming = transfer_learning(models['full_model'], 
                                           train_ds, 
                                           val_ds, 
                                           train_steps, 
                                           val_steps,
                                           callback_params,
                                           **hyperparams,
                                           **data_info)

        model, history, file_name = fine_tuning(models['full_model'], 
                                    train_ds, 
                                    val_ds, 
                                    train_steps, 
                                    val_steps,
                                    callback_params,
                                    **hyperparams,
                                    **data_info)

        save_results = Logger('datasetname')
        save_results(X_test=test_ds, y_true=[], model=model, 
                     network_weights_name = file_name, clasfier='multilayer perceptron', test_steps=test_steps, **data_info,
                     **hyperparams)
        

# regresion logistica se puede sacar incertidumbre seria desvacione standar de la prediccion, mirar si es igual en GP
# explicar whitekernel y como funciona GP.
# gradient boosting, random forest, SVM, logistic regression, capas densas. 
# funcion de orden superior
# pep 8, pypfleig
# uuid4 para gusrdar modelos
# parametros del optimizador
# hyperparams["augmentation"]["preprocessing_function"](next(iter(train_ds))[0]).numpy().min()


