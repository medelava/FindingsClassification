import h5py
import tensorflow as tf
import numpy as np
import tensorflow_addons as tfa
import matplotlib.pyplot as plt

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
                with h5py.File(self.file, 'r') as df:
                    img = df['images'][i:i+self.batch_size]
                    label = df[self.finding][i:i+self.batch_size]
                    yield img, label
                i = i + self.batch_size if i + self.batch_size < num_img else 0
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
    _gen = Generator('images_messidor2_goap.hdf5',batch_size, finding, sampling)
    
    findings_data = tf.data.Dataset.from_generator(
        _gen,
        output_types = ((tf.float32), (tf.float32)),
        output_shapes = ((batch_size, *img_shape), (batch_size,)) )

    findings_data = findings_data.map(lambda batch, label: apply_transform(batch, label, augmentation) )
    return findings_data
