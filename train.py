import numpy as np
import argparse
import cv2
from tensorflow.python.framework.tensor_conversion_registry import get

from i3d_inception import Inception_Inflated3d
from read_dataset import load_data_tfrecord
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout
import tensorflow as tf

physical_devices = tf.config.list_physical_devices('GPU')
try:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
except:
    pass

NUM_FRAMES = 16
FRAME_HEIGHT = 240
FRAME_WIDTH = 240
NUM_RGB_CHANNELS = 3
NUM_CLASSES = 226

def get_model():
    rgb_model = Inception_Inflated3d(
        include_top=False,
        weights='rgb_imagenet_and_kinetics',
        input_shape=(NUM_FRAMES, FRAME_HEIGHT, FRAME_WIDTH, NUM_RGB_CHANNELS),
        classes=NUM_CLASSES)

    x = rgb_model.output
    x = Dense(128, activation='elu', name='fc1')(x)
    x = Dropout(0.4)(x)
    x = Dense(128, activation='elu', name='fc2')(x)
    x = Dropout(0.3)(x)
    predictions = Dense(NUM_FRAMES, activation='softmax', name='predictions')(x)

    model = Model(rgb_model.input, predictions)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        steps_per_execution=50,
        loss='sparse_categorical_crossentropy',  
        metrics=['accuracy'])

    return model 

model = get_model()

train_fns = tf.io.gfile.glob('/home/alvaro/Documentos/video2tfrecord/example/output/*.tfrecords')
validation_fns = tf.io.gfile.glob('/home/alvaro/Documentos/video2tfrecord/example/test/*.tfrecords')


data_generator = {}
data_generator['train'] = load_data_tfrecord(train_fns)
data_generator['test'] = load_data_tfrecord(validation_fns)

batch_size = 2
train_steps = 28142 // batch_size
validation_steps = 3742 // batch_size

history = model.fit(
    data_generator['train'],
    steps_per_epoch=train_steps,
    epochs=1,
    validation_data=data_generator['test'],
    validation_steps=validation_steps)

print(history)