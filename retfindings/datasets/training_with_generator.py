import h5py
import tensorflow as tf
import numpy as np
import tensorflow_addons as tfa
import os
import csv
import pandas as pd
import numpy as np
import h5py
from sklearn import metrics
import matplotlib.pyplot as plt
import sys
import tensorflow as tf
#from keras.applications.vgg16 import VGG16
#from keras.applications.resnet50 import ResNet50
from tensorflow.keras.applications import InceptionV3
from sklearn.metrics import roc_auc_score, roc_curve, auc, confusion_matrix, accuracy_score, f1_score

from tensorflow.keras.callbacks import Callback, EarlyStopping, ModelCheckpoint
from tensorflow.keras.layers import (
    Dense,
    Dropout,
)

from tensorflow.keras.models import Model, Sequential, model_from_json
#from keras_preprocessing.image import ImageDataGenerator
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score

print(f"Numpy version: {np.__version__}")
print(f"Tensorflow version: {tf.__version__}")


class Generator:
    def __init__(self, file, batch_size, finding, sampling = 'batch'):
        self.file = file
        self.finding = finding
        self.batch_size = batch_size
        self.sampling = sampling
        self.init_generator()
        
    def batch_sample(self):
        def generator():
            i = 0
            with h5py.File(self.file, 'r') as df:
                num_img = df[self.finding].shape[0]
            while True:
                if i + self.batch_size > num_img:
                    i=0
                with h5py.File(self.file, 'r') as df:
                    img = df['images'][i:i+self.batch_size]
                    label = df[self.finding][i:i+self.batch_size]
                    yield img, label
                i = i + self.batch_size
                
        self.generator = generator()
        
    def random_sample(self):
        def generator(): 
            with h5py.File(self.file, 'r') as df:
                num_img = df[self.finding].shape[0]
            while True:
                with h5py.File(self.file, 'r') as df:
                    i = np.random.randint(0, num_img- self.batch_size) 
                    img = df['images'][i:i + self.batch_size]
                    label = df[self.finding][i:i + self.batch_size]
                    yield img, label
        self.generator = generator()
   
    def oversampling(self):
        def generator(): 
            with h5py.File(self.file, 'r') as df:
                y = df[self.finding][:]
                num_img = df[self.finding].shape[0]
            idx = np.arange(num_img)
            idx1 = idx[y==1]
            idx0 = idx[y==0]
            batch1 = self.batch_size//2
            batch0 = self.batch_size - batch1
            while True:
                with h5py.File(self.file, 'r') as df:
                    i1 = np.random.randint(0, idx1.size-batch1) 
                    i0 = np.random.randint(0, idx0.size-batch0)
                    img1 = df['images'][idx1[i1:i1+batch1]]
                    img0 = df['images'][idx0[i0:i0+batch0]]
                    label1 = df[self.finding][idx1[i1:i1+batch1]]
                    label0 = df[self.finding][idx0[i0:i0+batch0]]
                    yield np.concatenate([img1, img0], axis=0), np.concatenate([label1, label0], axis=0)
        self.generator = generator()
        
    def init_generator(self):
        if self.sampling == 'batch':
            self.batch_sample()
        elif self.sampling == 'rand':
            self.random_sample()
        elif self.sampling == 'oversampling':
            self.oversampling()
        
    def __call__(self):
        yield next(self.generator)

def apply_transform(batch, label, augmentation): 
    batch = tf.image.random_brightness(batch, **augmentation["random_brightness"])
    batch = tf.image.random_contrast(batch, **augmentation["random_contrast"])
    batch = tf.image.random_hue(batch, **augmentation["random_hue"])
    batch = tf.image.random_saturation(batch, **augmentation["random_saturation"])
    
    random_angles = tf.random.uniform(shape = (batch.shape[0], ), **augmentation["rotation_range"])
    batch = tfa.image.transform(batch,
                                tfa.image.transform_ops.angles_to_projective_transforms(
                                random_angles, tf.cast(batch.shape[1], tf.float32),
                                tf.cast(batch.shape[2], tf.float32)),
                                interpolation="BILINEAR")
    
    if augmentation["horizontal_flip"]:
        batch = tf.image.random_flip_left_right(batch)
    if augmentation["vertical_flip"]:
        batch = tf.image.random_flip_up_down(batch)
    return augmentation["preprocessing_function"](batch), label

    random_x = tf.random.uniform(shape = (batch.shape[0], 1), **augmentation["width_shift_range"])
    random_y = tf.random.uniform(shape = (batch.shape[0], 1), **augmentation["height_shift_range"])
    translate = tf.concat([random_x, random_y], axis=1)
    batch = tfa.image.translate(batch, translations = translate, interpolation="BILINEAR")
    return batch, label



def make_generator(path, batch_size, finding, sampling, augmentation, img_shape):          
        _gen = Generator(path,batch_size, finding, sampling)

        findings_data = tf.data.Dataset.from_generator(
            _gen,
            output_types = ((tf.float32), (tf.float32)),
            output_shapes = ((batch_size, *img_shape), (batch_size,)) )

        findings_data = findings_data.map(lambda batch, label: apply_transform(batch, label, augmentation) )
        return findings_data




def build_model(network, warmup_learning_rate, warmup_decay, dropout):
    if network == 'inceptionv3':
        base_model = tf.keras.applications.InceptionV3(weights='imagenet', 
                                                       include_top=False,
                                                       input_shape=img_shape,
                                                       pooling='avg'
                                                       )
    elif network == 'resnet50':
        base_model = tf.keras.applications.ResNet50(weights="imagenet",
                                                    include_top=True,
                                                    input_shape=img_shape,
                                                    pooling='avg'
                                                    )

    elif network == 'efficientnetb5': 
        base_model = tf.keras.applications.EfficientNetB5(weights="imagenet",
                                                          include_top=False,
                                                          input_shape=img_shape,
                                                          pooling='avg'
                                                          )

    elif network == 'xception':
        base_model = tf.keras.applications.Xception(weights="imagenet",
                                                    include_top=True,
                                                    input_shape=img_shape,
                                                    pooling='avg'
                                                    )

    # Freeze the feature extractor
    for layer in base_model.layers:
        layer.trainable = False

    x = Dense(1024)(base_model.output)
    x = Dropout(dropout)(x)
    predictions = Dense(1, activation='sigmoid')(x)

    model = Model(inputs=base_model.input, outputs=predictions)

    optimizer= tf.keras.optimizers.RMSprop(learning_rate=warmup_learning_rate, decay=warmup_decay)

    model.compile(
        loss=tf.keras.losses.BinaryCrossentropy(),
        optimizer=optimizer,
        metrics=['accuracy', tf.keras.metrics.AUC()]
    )

    
    return model

    
    
def training(model_path, model_path_warmup, warmup_epochs, warmup_lr, warmup_decay, train_epochs, train_lr, train_decay, warmup, finding,  num, train_generator): 
    
    if warmup:
        steps_per_epoch = (num // batch_size)
        
        history = model.fit(
        train_generator,
        steps_per_epoch = steps_per_epoch,
        epochs = warmup_epochs,
        workers = 1
        #validation_data = val_generator,
        #validation_steps = validation_steps
        )
       
    
        plt.plot(history.history['loss'])
        #plt.plot(history.history['val_loss'])
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'val'], loc='upper left')
        plt.savefig('{}/{}_{}_{}.png'.format(model_path_warmup, finding, warmup_lr, warmup_epochs)) 
        
        
        # Serialize model to JSON
        model_json = model.to_json()
        warm_up_fullpath = '{}/{}'.format(model_path_warmup, "InceptionV3_{}_warmup_lr{}_epochs{}.json".format(finding, warmup_lr, warmup_epochs) )
        with open(warm_up_fullpath, "w") as json_file:
            json_file.write(model_json)

        # Serialize weights to HDF5
        model.save_weights('{}/{}'.format(model_path_warmup, "InceptionV3_{}_warmup_lr{}_epochs{}.h5".format(finding, warmup_lr, warmup_epochs) ))
        print("Saved model to disk")
    
    else:
        # Load json file structure and create model
        json_file = open('{}/{}'.format(model_path_warmup, "InceptionV3_{}_warmup_lr{}_epochs{}.json".format(finding, warmup_lr, warmup_epochs)), 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        model = model_from_json(loaded_model_json)

        # Load weights into new model
        model.load_weights('{}/{}'.format(model_path_warmup, "InceptionV3_{}_warmup_lr{}_epochs{}.h5".format(finding, warmup_lr, warmup_epochs) ))
        print("Model loaded ")

        # Freeze the feature extractor
        for layer in model.layers:
            layer.trainable = True

        file_name = "InceptionV3_{}_lr{}_epochs{}.hdf5".format(finding, train_lr, train_epochs)
        if not os.path.exists('{}/lr{}_epochs{}'.format(model_path, train_lr, train_epochs)):
            os.makedirs('{}/lr{}_epochs{}'.format(model_path, train_lr, train_epochs))
        filepath = '{}/lr{}_epochs{}/{}'.format(model_path, train_lr, train_epochs, file_name)
        
        steps_per_epoch = (num // batch_size)
        
        checkpoint = ModelCheckpoint(filepath, 
                                     monitor='loss', 
                                     verbose=1, 
                                     save_best_only = True, 
                                     mode='auto')

        early_stop = EarlyStopping(monitor='loss', 
                                   min_delta=0, 
                                   patience=200, 
                                   verbose=1, 
                                   mode='auto', 
                                   baseline=None, 
                                   restore_best_weights=True)
        
        optimizer= tf.keras.optimizers.RMSprop(learning_rate=train_lr, decay=train_decay)
        model.compile(
            loss=tf.keras.losses.BinaryCrossentropy(),
            optimizer=optimizer,
            metrics=['accuracy', tf.keras.metrics.AUC()]
        )
        
        history = model.fit(
            train_generator,
            steps_per_epoch = steps_per_epoch,
            epochs = train_epochs,
            callbacks = [checkpoint, early_stop]
            #validation_data = val_generator,
            #validation_steps = validation_steps
            )
    
                        
    return model, history



def save_results(network, model, model_path, path_test, finding, warmup_lr, warmup_epochs, warmup_decay, train_lr, train_epochs, train_decay, history):
    
    plt.plot(history.history['loss'])
    #plt.plot(history.history['val_loss'])
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.savefig('{}/lr{}_epochs{}/{}_lr{}_epochs{}.png'.format(model_path, train_lr, train_epochs, finding, train_lr, train_epochs)) 
        
    
    file_name = "/tf/data2/medel2/new_tesis_experiments/hdf5_with_images/findings/images_messidor2_goap.hdf5"
    finding = finding

    with h5py.File(file_name, 'r') as df:
        X_test = df['images'][:]
        y_test = df[finding][:]
        
    y_pred_test = model.predict(X_test)
    y_test_round = np.round(y_pred_test)
    
    f1_macro = f1_score(y_test, y_test_round,average="macro")
    f1_micro = f1_score(y_test, y_test_round,average="micro")
    #precision_macro = precision_score(y_test, y_test_round, average='macro')
    #precision_micro = precision_score(y_test, y_test_round, average='micro')
    #recall = recall_score(y_test, y_test_round,average="macro")
    #accuracy = accuracy_score(y_test, y_test_round)
    #kappa = cohen_kappa_score(y_test, y_test_round, weights='quadratic')
    
    
    con_mat=confusion_matrix(y_test,y_test_round)
    tp=con_mat[1][1]
    tn=con_mat[0][0]
    fp=con_mat[0][1]
    fn=con_mat[1][0]
    recall=tp/(tp+fn)
    specificity=tn/(tn+fp)
    precision=tp/(tp+fp)
    acc=(tp+tn)/(tp+tn+fp+fn)
    
    fpr, tpr, threshold=roc_curve(y_test, y_pred_test)
    auc=roc_auc_score(y_test, y_pred_test)
    
    #with open('{}/{}_results'.format(model_path, finding),'a') as fd:
    #    wr = csv.writer(fp, dialect='excel')
    #    wr.write([network, warmup_lr, warmup_epochs, warmup_decay, train_lr, train_epochs, train_decay, auc, recall, specificity, acc, f1_macro, f1_micro])
    df = pd.DataFrame([[network, warmup_lr, warmup_epochs, warmup_decay, train_lr, train_epochs, train_decay, auc, recall, specificity, acc, f1_macro, f1_micro]])
        
    with open('{}/{}_results.csv'.format(model_path, finding), 'a') as f:
        df.to_csv(f, header =['network', 'warmup_lr', 'warmup_epochs', 'warmup_decay', 'train_lr', 'train_epochs', 'train_decay', 'auc', 'recall', 'specificity', 'acc', 'f1_macro', 'f1_micro'])
    
 
    
    #plt.title('Receiver Operating Characteristic', fontsize = 22.0)
    plt.plot(fpr, tpr, 'b', label = 'AUC = %0.4f' % auc)
    plt.legend(loc='lower right')
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate', fontsize = 20.0)
    plt.xlabel('False Positive Rate', fontsize = 20.0)
    plt.xticks(fontsize=15, rotation=0)
    plt.yticks(fontsize=15, rotation=0)

    plt.savefig('{}/lr{}_epochs{}/AUC_{}_lr{}_epochs{}.png'.format(model_path, train_lr, train_epochs, finding, train_lr, train_epochs))  










if __name__=='__main__':
    file_name = "/tf/data2/medel2/new_tesis_experiments/hdf5_with_images/findings/images_messidor2_goap.hdf5"
    finding = 'aneurisms' 

    with h5py.File(file_name, 'r') as df:
        num = df[finding].shape[0]
    print(num)
    batch_size = 32
    sampling = 'rand'
    img_shape = (299, 299, 3)

    augmentation = {"preprocessing_function": lambda i:i, "apply":False,
                    "random_brightness": {"max_delta": 0.5},
                    "random_contrast": {"lower":0.6, "upper":1},
                    "random_hue": {"max_delta": 0.1},
                    "random_saturation": {"lower": 0.6, "upper":1},
                    "random_rotation": {"minval": 0, "maxval": 2*np.pi},
                    "horizontal_flip": True, "vertical_flip": True,
                    "rotation_range": {"minval": -0.3, "maxval": 0.3},
                    "width_shift_range": {"minval": -2, "maxval": 2},
                    "height_shift_range": {"minval": -2, "maxval": 2},}

    finding = 'aneurisms'                    
    network = 'inceptionv3'
    model_path_warmup = '/tf/home/repositories/Findings-Classification/models/warmup/{}'.format(finding)
    model_path = '/tf/home/repositories/Findings-Classification/models/trained_model/{}'.format(finding)
    train_epochs = 2
    train_decay = 1e-7
    train_lr = 1e-6
    warmup_epochs= 2
    warmup_decay = 1e-7
    warmup_lr = 1e-6
    train_decay = 1e-7
    dropout = 0.2
    warmup = False

    model = build_model(network, warmup_lr, warmup_epochs, dropout)
    train_generator = make_generator(file_name, batch_size, finding, sampling, augmentation, img_shape).repeat()
    
    trained_model, history = training(model_path, model_path_warmup, warmup_epochs, warmup_lr, warmup_decay, train_epochs, train_lr, train_decay, warmup, finding,  num, train_generator)
    
    path_test = "/tf/data2/medel2/new_tesis_experiments/hdf5_with_images/findings/images_messidor2_goap.hdf5"
    network = 'inceptionv3'
    save_results(network, model, model_path, path_test, finding, warmup_lr, warmup_epochs, warmup_decay, train_lr, train_epochs, train_decay, history)
    
    
    
    
