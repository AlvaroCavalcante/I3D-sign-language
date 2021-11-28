import argparse

from tensorflow.python.keras.utils.generic_utils import default

from i3d_inception import Inception_Inflated3d
from read_dataset import load_data_tfrecord
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras import layers
import tensorflow as tf
import time 
from tensorflow.keras.callbacks import LearningRateScheduler, ModelCheckpoint

physical_devices = tf.config.list_physical_devices('GPU')
try:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
except:
    pass

NUM_FRAMES = 16
FRAME_HEIGHT = 224
FRAME_WIDTH = 224
NUM_RGB_CHANNELS = 3
NUM_CLASSES = 226

def lr_time_based_decay(epoch, lr, nb_epoch=1):
    decay = lr / nb_epoch
    return lr * 1 / (1 + decay * epoch)

def get_model(freeze, lr, checkpoint_path):
    if not checkpoint_path:
        rgb_model = Inception_Inflated3d(
            include_top=False,
            weights='rgb_imagenet_and_kinetics',
            input_shape=(NUM_FRAMES, FRAME_HEIGHT, FRAME_WIDTH, NUM_RGB_CHANNELS),
            classes=NUM_CLASSES)

        if freeze:
            rgb_model.trainable = False
        else:
            for layer in rgb_model.layers[50:]:
                if not isinstance(layer, layers.BatchNormalization):
                    layer.trainable = True

        x = rgb_model.output
        x = Dense(512, activation='elu', name='fc1')(x)
        x = Dropout(0.4)(x)
        x = Dense(256, activation='elu', name='fc2')(x)
        x = Dropout(0.3)(x)
        x = Dense(128, activation='elu', name='fc3')(x)
        x = Dropout(0.3)(x)
        predictions = Dense(NUM_CLASSES, activation='softmax', name='predictions')(x)

        model = Model(rgb_model.input, predictions)
    else:
        model = load_model(checkpoint_path)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        steps_per_execution=50,
        loss='sparse_categorical_crossentropy',  
        metrics=['accuracy'])

    return model 

def main(args):
    model = get_model(args.freeze, args.learning_rate, args.checkpoint_path)

    train_fns = tf.io.gfile.glob(args.train_path)
    validation_fns = tf.io.gfile.glob(args.val_path)

    batch_size = args.batch_size
    train_steps = 28142 // batch_size
    validation_steps = 3742 // batch_size

    data_generator = {}
    data_generator['train'] = load_data_tfrecord(train_fns, batch_size)
    data_generator['test'] = load_data_tfrecord(validation_fns, batch_size)

    # tf.keras.utils.plot_model(model, "model.png", show_shapes=True)

    callbacks_list = [
        ModelCheckpoint(args.output_path, monitor='val_accuracy', verbose=1, save_best_only=True),
        LearningRateScheduler(lr_time_based_decay, verbose=1),
    ]

    start = time.time()
    history = model.fit(
        data_generator['train'],
        steps_per_epoch=train_steps,
        epochs=args.epochs,
        validation_data=data_generator['test'],
        validation_steps=validation_steps,
        callbacks=callbacks_list)

    print('Fitting time: ', time.time() - start)
if __name__ == '__main__':
    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', type=int, default='64')
    parser.add_argument('--epochs', type=int, default='1000')
    parser.add_argument('--train-path', type=str, default="/home/alvaro/Documentos/video2tfrecord/example/output/*.tfrecords")
    parser.add_argument('--val-path', type=str, default="/home/alvaro/Documentos/video2tfrecord/example/test/*.tfrecords")
    parser.add_argument('--freeze', type=bool, default=True)
    parser.add_argument('--output-path', type=str, default='./')
    parser.add_argument('--learning-rate', type=float, default=0.1)
    parser.add_argument('--checkpoint-path', type=str, default=None)

    args = parser.parse_args()
    main(args)

# python3 train.py --batch-size 2 --epochs 10 --train-path "/home/alvaro/Documentos/video2tfrecord/example/output/*.tfrecords" --val-path "/home/alvaro/Documentos/video2tfrecord/example/test/*.tfrecords"python3 train.py --batch-size 2 --epochs 10 --train-path "/home/alvaro/Documentos/video2tfrecord/example/output/*.tfrecords" --val-path "/home/alvaro/Documentos/video2tfrecord/example/test/*.tfrecords"