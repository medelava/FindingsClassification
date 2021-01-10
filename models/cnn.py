import tensorflow as tf
from tensorflow.keras.applications import InceptionV3 


class CNNFindings(tf.keras.Model):
    def __init__(self, cnn, n_outputs):
        super(CNNFindings, self).__init__()
        self.base_model = cnn
        self.dense1024 = tf.keras.layers.Dense(1024)
        self.dropout = tf.keras.layers.Dropout(0.2)
        self.predictions = tf.keras.layers.Dense(n_outputs, activation='sigmoid')
        
    def call(self, input_tensor, training=True):
        x = self.base_model(input_tensor, training=training)
        x = self.dense1024(x, training=training)
        x = self.dropout(x, training=training)
        predictions = self.predictions(x, training=training)
        return predictions
        
    def freeze(self):
        for layer in self.base_model.layers:
            layer.trainable = False
            
    def unfreeze(self):
        for layer in self.base_model.layers:
            layer.trainable = True
        
        