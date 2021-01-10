
import tensorflow as tf
import sys
from FindingsClassification import 


if __name__ == '__main__':
    cnn = tf.keras.applications.inception_v3.InceptionV3(
        include_top=False,
        weights="imagenet",
        input_shape=(299, 299, 3),
        pooling='avg',)

  
    inp = tf.keras.layers.Input((299, 299, 3))  
    output = models.CNNFindings(cnn, 2)(inp)
    model = tf.keras.Model(inputs=inp, outputs=output)
    model.compile(optimizer='adam', loss='mse')
    print(model.summary())

