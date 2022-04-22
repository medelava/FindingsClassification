from inspect import getmembers, isfunction
from sklearn.model_selection import GridSearchCV
#import classifiers
from sklearn.gaussian_process.kernels import RBF as _RBF
from sklearn.gaussian_process.kernels import WhiteKernel as _WhiteKernel
from sklearn.gaussian_process.kernels import Matern as _Matern
from keras.preprocessing.image import ImageDataGenerator
from sklearn import preprocessing
from keras.models import Model
import os
import keras
import numpy as np
from keras.models import load_model

def features_extraction(model_name, test_dir, image_size, batch_size):
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"

    features = []
    y_true = []
    
    model=load_model('weights/'+model_name)
    
    if model_name == 'vgg16_CAB_DDR.h5' or 'densenet121_CAB_DDR.h5' or 'mobilenet1.0_CAB_DDR.h5':
        model = Model(inputs=model.inputs, output=model.layers[-24].output)
    elif model_name == 'vgg16_CAB_EyePACS.h5' or 'resnet50_CAB_EyePACS.h5' or 'xception_CAB_DDR.h5' or 'xception_CAB_EyePACS.h5' or 'densenet121_CAB_EyePACS.h5' or 'mobilenet1.0_CAB_EyePACS.h5':
        model = Model(inputs=model.layers[-2].inputs, outputs=model.layers[-2].layers[-24].output)
       
    x = keras.layers.GlobalAveragePooling2D()(model.output)
    model = keras.models.Model(inputs=model.inputs, outputs=x)
       
    for i in range(2):
        datadirs=test_dir+str(i)+'/'
        filenames=os.listdir(datadirs)
        num=len(filenames)
        generator = ImageDataGenerator()
        generator_data=generator.flow_from_directory(directory=test_dir,target_size=(image_size,image_size),
                                             batch_size=batch_size,class_mode=None,classes=str(i))
        predict=model.predict_generator(generator_data,steps=num/batch_size,verbose=1,workers=1)
        y_true.extend(np.tile(i, predict.shape[0]))
        features.extend(predict)
    return features, y_true


batch_size=3
image_size=512
finding='H_small'
train_dir='/home/mder/datasets/kaggle/goap_dataset/512/{}/'.format(finding)
test_dir='/home/mder/datasets/messidor2/dr_folders/grades/borrar/'  
#models_names = ['vgg16_CAB_DDR.h5', 'vgg16_CAB_EyePACS.h5', 'resnet50_CAB_EyePACS.h5']

model_name = 'vgg16_CAB_DDR.h5'
train_features, train_labels = features_extraction(model_name, train_dir, image_size, batch_size)

#%%
test_features, test_labels = features_extraction(model_name, test_dir, image_size, batch_size)

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
    
    
        
   