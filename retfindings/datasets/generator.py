"""
Create a generator from hdf5 files, using tf.data.Datasets.
"""

import h5py as _h5py
import tensorflow as _tf
import numpy as _np
import tensorflow_addons as _tfa

class _Generator:
    """
    Implementation of three generators, from images stored in hdf5 files. 
    
    Parameters
    ----------
    path : str
        location of the hdf5 file with the information to build a generator.
    batch_size : int
        batch size in each batch in the tf dataset.
    label_name : str
        labels key in the hdf5 file.
    sampling: str
        kind of generator to use. It can be 'batch', 'rand' and 'oversampling'.
        
    Returns
    -------
    Generator batch.
    """
        
    def __init__(self, path, batch_size, label_name, sampling = 'batch'):
        self.__path = path
        self.__label_name = label_name
        self.__batch_size = batch_size
        self.__sampling = sampling
        self.init_generator()
        
    def _batch_sample(self):
        """
        Implementation of generator that retrieves images in batchs in the 
        same order as they are saved in the hdf5 file. 
        
        Parameters
        ----------
        self params
            
        Returns
        -------
        yields images and labels batch.
        """
        def generator():
            self.cont = 0
            with _h5py.File(self.__path, 'r') as df:
                n_samples = df[self.__label_name].shape[0]
            while True:
                if self.cont + self.__batch_size > n_samples:
                    self.cont = 0
                with _h5py.File(self.__path, 'r') as df:
                    X = df['images'][self.cont: self.cont + self.__batch_size]
                    label = df[self.__label_name][self.cont: self.cont + self.__batch_size]
                    yield X, label
                self.cont += self.__batch_size
        self.__generator = generator()
        
    def __random_sample(self):
        """
        Implementation of generator that reads batches of shape batch_size 
        with a random start point in the hdf5 file. 
        
        Parameters
        ----------
        self params
            
        Returns
        -------
        yields images and labels batch.
        """
        
        def generator(): 
            with _h5py.File(self.__path, 'r') as df:
                n_samples = df[self.__label_name].shape[0]
            while True:
                with _h5py.File(self.__path, 'r') as df:
                    rand = _np.random.randint(0, n_samples- self.__batch_size) 
                    X = df['images'][rand : rand + self.__batch_size]
                    label = df[self.__label_name][rand : rand + self.__batch_size]
                    yield X, label
        self.__generator = generator()
   
    def _oversampling(self):
        """
        Implementation of generator that retrieves same amount of images of 
        the class 0 and class 1 selected randomly.
        
        Parameters
        ----------
        self params
            
        Returns
        -------
        yields balanced image batch and labels batch.
        """   
        def generator(): 
            with _h5py.File(self.__path, 'r') as df:
                y = df[self.__label_name][:]
                n_samples = df[self.__label_name].shape[0]
            idx = _np.arange(n_samples)
            idx1 = idx[y==1]
            idx0 = idx[y==0]
            batch1 = self.__batch_size//2
            batch0 = self.__batch_size - batch1
            while True:
                with _h5py.File(self.__path, 'r') as df:
                    i1 = _np.random.randint(0, idx1.size-batch1) 
                    i0 = _np.random.randint(0, idx0.size-batch0)
                    img1 = df['images'][idx1[i1: i1 + batch1]]
                    img0 = df['images'][idx0[i0: i0 + batch0]]
                    label1 = df[self.__label_name][idx1[i1: i1 + batch1]]
                    label0 = df[self.__label_name][idx0[i0: i0 + batch0]]
                    yield _np.concatenate([img1, img0], axis=0), _np.concatenate([label1, label0], axis=0)
        self.__generator = generator()
        
    def init_generator(self):
        """
        Select the kind of generaor to be used. 
        
        Parameters
        ----------
        self params
            
        Returns
        -------
        
        """
        if self.__sampling == 'batch':
            self._batch_sample()
        elif self.__sampling == 'rand':
            self._random_sample()
        elif self.__sampling == 'oversampling':
            self._oversampling()
        
    def __call__(self):
        while True:
            yield next(self.__generator)

def _apply_transform(batch, label, augmentation, partition): 
    """
    Applyes data augmentation over a batch of images.
    
    Parameters
    ----------
    self params
        
    Returns
    -------
    yields images and labels batch.
    """
        

    if partition == 'train':
        batch = _tf.image.random_brightness(batch, **augmentation["random_brightness"])
        batch = _tf.image.random_contrast(batch, **augmentation["random_contrast"])
        batch = _tf.image.random_hue(batch, **augmentation["random_hue"])
        batch = _tf.image.random_saturation(batch, **augmentation["random_saturation"])
        
        #random_angles = _tf.random.uniform(shape = (batch.shape[0], ), **augmentation["rotation_range"])
        #batch = _tfa.image.transform(batch,
        #                            _tfa.image.transform_ops.angles_to_projective_transforms(
        #                            random_angles, _tf.cast(batch.shape[1], _tf.float32),
        #                           _tf.cast(batch.shape[2], _tf.float32)),
        #                            interpolation="BILINEAR")
    
        #random_x = _tf.random.uniform(shape = (batch.shape[0], 1), **augmentation["width_shift_range"])
        #random_y = _tf.random.uniform(shape = (batch.shape[0], 1), **augmentation["height_shift_range"])
        #translate = _tf.concat([random_x, random_y], axis=1)
        #batch = _tfa.image.translate(batch, translations = translate, interpolation="BILINEAR")
        
        if augmentation["horizontal_flip"]:
            batch = _tf.image.random_flip_left_right(batch)
        if augmentation["vertical_flip"]:
            batch = _tf.image.random_flip_up_down(batch)
    else:
        pass
    
    return augmentation["preprocessing_function"](batch), label


def make_generator(path, batch_size, label_name, sampling, augmentation, img_shape, partition):          
    _gen = _Generator(path ,batch_size, label_name, sampling)
    
    data = _tf.data.Dataset.from_generator(
        _gen,
        output_types = ((_tf.float32), (_tf.float32)),
        output_shapes = ((batch_size, *img_shape), (batch_size,)) 
        ) 

    return data.map(lambda batch, label: _apply_transform(batch, label, augmentation, partition) )
    