import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np 
from augmentation import transform_batch

AUTOTUNE = tf.data.experimental.AUTOTUNE

def center_crop(img, dims):
	width, height = img.shape[1], img.shape[0]

	crop_width = dims[0] if dims[0] < img.shape[1] else img.shape[1]
	crop_height = dims[1] if dims[1] < img.shape[0] else img.shape[0] 
	mid_x, mid_y = int(width/2), int(height/2)
	cw2, ch2 = int(crop_width/2), int(crop_height/2) 
	crop_img = img[mid_y-ch2:mid_y+ch2, mid_x-cw2:mid_x+cw2]
	return crop_img

def read_tfrecord(example_proto):   
    image_seq = []

    for image_count in range(24-8):
        path = 'blob' + '/' + str(image_count)

        feature_dict = {path: tf.io.FixedLenFeature([], tf.string),
        'height': tf.io.FixedLenFeature([], tf.int64),
        'width': tf.io.FixedLenFeature([], tf.int64),
        'depth': tf.io.FixedLenFeature([], tf.int64),
        'label': tf.io.FixedLenFeature([], tf.int64)}

        features = tf.io.parse_single_example(example_proto, features=feature_dict)

        width = 224
        height = 224

        image = tf.image.decode_jpeg(features[path], channels=3)
        image = tf.image.resize(image, [256, 256])
        image = center_crop(image, (width, height))

        # image = tf.cast(image, tf.float32) / 255.
        image = (tf.cast(image, tf.float32) / 127.5) - 1.0

        image = tf.reshape(image, tf.stack([height, width, 3]))
        image = tf.reshape(image, [1, height, width, 3])
        image_seq.append(image)

        label = tf.cast(features['label'], tf.int32)

    image_seq = tf.concat(image_seq, 0)

    return image_seq, label

def load_dataset(tf_record_path):
    raw_dataset = tf.data.TFRecordDataset(tf_record_path, num_parallel_reads=AUTOTUNE)
    parsed_dataset = raw_dataset.map(read_tfrecord, num_parallel_calls=AUTOTUNE)
    return parsed_dataset 

def prepare_for_training(ds, batch_size, shuffle_buffer_size=250, augmentation=False):
    ds.cache() # I can remove this to don't use cache or use cocodata.tfcache
    ds = ds.repeat()
    ds = ds.batch(batch_size)

    if augmentation:
        ds = ds.map(transform_batch, num_parallel_calls=AUTOTUNE)

    ds = ds.unbatch()
    ds = ds.shuffle(buffer_size=shuffle_buffer_size)
    ds = ds.batch(batch_size)
    ds = ds.prefetch(AUTOTUNE)
    return ds

def load_data_tfrecord(tfrecord_path, batch_size):
    dataset = load_dataset(tfrecord_path)
    dataset = prepare_for_training(dataset, batch_size)
    return dataset

def visualize_imgs():
    row = 4; col = 4
    train_fns = tf.io.gfile.glob('/home/alvaro/Documentos/video2tfrecord/example/output/*.tfrecords')

    all_elements = load_data_tfrecord(train_fns, 1).unbatch()
    dataset = all_elements.repeat().batch(1)

    for (seq, label) in dataset:
        plt.figure(figsize=(15,int(15*row/col)))
        for j in range(row*col):
            plt.subplot(row,col,j+1)
            plt.axis('off')
            plt.imshow(np.array(seq[0, j,]))
        plt.show()

if __name__ == '__main__':
    visualize_imgs()