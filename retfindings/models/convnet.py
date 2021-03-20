import tensorflow as _tf


class CNNFindings(_tf.keras.Model):
    def __init__(self, cnn, n_outputs):
        super(CNNFindings, self).__init__()
        self.base_model = cnn
        self.dense1024 = _tf.keras.layers.Dense(1024)
        self.dropout = _tf.keras.layers.Dropout(0.2)
        self.predictions = _tf.keras.layers.Dense(n_outputs, activation='sigmoid')
        
    def call(self, input_tensor, training=True):
        #input_tensor = _tf.keras.Input(shape = img_shape)
        #model(input_tensor)
        features = self.base_model(input_tensor, training=training)
        x = self.dense1024(features, training=training)
        x = self.dropout(x, training=training)
        predictions = self.predictions(x, training=training)
        return predictions
    
    def features_extraction(self, input_tensor, training =True):
        return self.base_model(input_tensor, training=training)
        
        
    def freeze(self):
        for layer in self.base_model.layers:
            layer.trainable = False
            
    def unfreeze(self):
        for layer in self.base_model.layers:
            layer.trainable = True


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
        base_model = _tf.keras.applications.InceptionV3(weights = 'imagenet', 
                                                       include_top = False,
                                                       input_shape = img_shape,
                                                       pooling = 'avg'
                                                       )
    elif network == 'resnet50':
        base_model = _tf.keras.applications.ResNet50(weights = "imagenet",
                                                    include_top = False,
                                                    input_shape = img_shape,
                                                    pooling = 'avg'
                                                    )

    elif network == 'efficientnetb5': 
        base_model = _tf.keras.applications.EfficientNetB5(weights = "imagenet",
                                                          include_top = False,
                                                          input_shape = img_shape,
                                                          pooling = 'avg'
                                                          )

    elif network == 'xception':
        base_model = _tf.keras.applications.Xception(weights="imagenet",
                                                    include_top=False,
                                                    input_shape=img_shape,
                                                    pooling='avg'
                                                    )
        
    input_tensor = _tf.keras.Input(shape = img_shape)
    model = CNNFindings(base_model, kwargs["outputs"])
    #predictions = model(input_tensor) 
    features = model.features_extraction(input_tensor)

    return {'full_model': model, 
            'features_extractor': _tf.keras.Model(inputs = [input_tensor], outputs = [features])}
 


def compile_model(model, optimizer):
    model.compile(
            loss = _tf.keras.losses.BinaryCrossentropy(),
            optimizer = optimizer,
            metrics = ['accuracy', _tf.keras.metrics.AUC()]
        )
    return model