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




def compile_model(model, optimizer):
    model.compile(
            loss = _tf.keras.losses.BinaryCrossentropy(),
            optimizer = optimizer,
            metrics = ['accuracy', _tf.keras.metrics.AUC()]
        )
    return model