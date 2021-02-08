import h5py as _h5py
import tensorflow as _tf
import numpy as _np
import tensorflow_addons as _tfa

class _Generator:
    # TODOC
    def __init__(self, path, batch_size, label_name, sampling = 'batch'):
        self.path = path
        self.label_name = label_name
        self.batch_size = batch_size
        self.sampling = sampling
        self.init_generator()
        
    def batch_sample(self):
        def generator():
            i = 0
            with _h5py.File(self.file, 'r') as df:
                num_img = df[self.finding].shape[0]
            while True:
                if i + self.batch_size > num_img:
                    i=0
                with _h5py.File(self.file, 'r') as df:
                    img = df['images'][i:i+self.batch_size]
                    label = df[self.finding][i:i+self.batch_size]
                    yield img, label
                i = i + self.batch_size
                
        self.generator = generator()
        
    def random_sample(self):
        def generator(): 
            with _h5py.File(self.path, 'r') as df:
                num_img = df[self.label_name].shape[0]
            while True:
                with _h5py.File(self.path, 'r') as df:
                    i = _np.random.randint(0, num_img- self.batch_size) 
                    img = df['images'][i:i + self.batch_size]
                    label = df[self.label_name][i:i + self.batch_size]
                    yield img, label
        self.generator = generator()
   
    def oversampling(self):
        def generator(): 
            with _h5py.File(self.path, 'r') as df:
                y = df[self.label_name][:]
                num_img = df[self.label_name].shape[0]
            idx = _np.arange(num_img)
            idx1 = idx[y==1]
            idx0 = idx[y==0]
            batch1 = self.batch_size//2
            batch0 = self.batch_size - batch1
            while True:
                with _h5py.File(self.path, 'r') as df:
                    i1 = _np.random.randint(0, idx1.size-batch1) 
                    i0 = _np.random.randint(0, idx0.size-batch0)
                    img1 = df['images'][idx1[i1:i1+batch1]]
                    img0 = df['images'][idx0[i0:i0+batch0]]
                    label1 = df[self.label_name][idx1[i1:i1+batch1]]
                    label0 = df[self.label_name][idx0[i0:i0+batch0]]
                    yield _np.concatenate([img1, img0], axis=0), _np.concatenate([label1, label0], axis=0)
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

def _apply_transform(batch, label, augmentation): 
    batch = _tf.image.random_brightness(batch, **augmentation["random_brightness"])
    batch = _tf.image.random_contrast(batch, **augmentation["random_contrast"])
    batch = _tf.image.random_hue(batch, **augmentation["random_hue"])
    batch = _tf.image.random_saturation(batch, **augmentation["random_saturation"])
    
    random_angles = _tf.random.uniform(shape = (batch.shape[0], ), **augmentation["rotation_range"])
    batch = _tfa.image.transform(batch,
                                _tfa.image.transform_ops.angles_to_projective_transforms(
                                random_angles, _tf.cast(batch.shape[1], _tf.float32),
                                _tf.cast(batch.shape[2], _tf.float32)),
                                interpolation="BILINEAR")
    
    if augmentation["horizontal_flip"]:
        batch = _tf.image.random_flip_left_right(batch)
    if augmentation["vertical_flip"]:
        batch = _tf.image.random_flip_up_down(batch)
    return augmentation["preprocessing_function"](batch), label

    random_x = _tf.random.uniform(shape = (batch.shape[0], 1), **augmentation["width_shift_range"])
    random_y = _tf.random.uniform(shape = (batch.shape[0], 1), **augmentation["height_shift_range"])
    translate = _tf.concat([random_x, random_y], axis=1)
    batch = _tfa.image.translate(batch, translations = translate, interpolation="BILINEAR")
    return batch, label


def make_generator(path, batch_size, label_name, sampling, augmentation, img_shape, partition):          
    _gen = _Generator(path ,batch_size, label_name, sampling)
    
    data = _tf.data.Dataset.from_generator(
        _gen,
        output_types = ((_tf.float32), (_tf.float32)),
        output_shapes = ((batch_size, *img_shape), (batch_size,)) )

    if partition == 'train':
        return data.map(lambda batch, label: _apply_transform(batch, label, augmentation) )
    else:
        return data