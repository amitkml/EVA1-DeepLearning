
# train data prep

import os
from os import listdir
from os.path import join
 
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.python.keras.preprocessing.image import img_to_array, load_img
 
# IMAGE_HEIGHT = 40
# IMAGE_WIDTH = 40
# IMAGE_DEPTH = 3
# NUM_CLASSES = 10
 
 
def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))
 
 
def save_tf_records(x_train, y_train, out_path):
    writer = tf.python_io.TFRecordWriter(out_path)
 
    for i in range(y_train.shape[0]):
   
        example = tf.train.Example(features=tf.train.Features(
            feature={'image': _bytes_feature(x_train[i].tostring()),
                     'labels': _bytes_feature(
                         y_train[i].tostring())
                     }))
 
        writer.write(example.SerializeToString())
 
    writer.close()

def load_tf_records(path):
    dataset = tf.data.TFRecordDataset(path)
 
    def parser(record):
        featdef = {
            'image': tf.FixedLenFeature(shape=[], dtype=tf.string),
            'labels': tf.FixedLenFeature(shape=[], dtype=tf.string),
        }
 
        example = tf.parse_single_example(record, featdef)
        im = tf.decode_raw(example['image'], tf.float32)
        im = tf.reshape(im, (-1, 40, 40, 3))
        lbl = tf.decode_raw(example['labels'], tf.int64)
        return im, lbl
 
    dataset = dataset.map(parser)
    #dataset = dataset.shuffle(buffer_size=50000)
    dataset = dataset.batch(50000)
    dataset = dataset.repeat(1)
    iterator = dataset.make_one_shot_iterator()
    return iterator.get_next()


# test data prep

# Type convertion functions
def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

# convert images to tfrecords

def _convert_to_tfrecord(data, labels, tfrecords_filename):
  """Converts a file to TFRecords."""
  print('Generating %s' % tfrecords_filename)
  with tf.python_io.TFRecordWriter(tfrecords_filename) as record_writer:
    num_entries_in_batch = len(labels)
    for i in range(num_entries_in_batch):
      example = tf.train.Example(features=tf.train.Features(
        feature={
          'image': _bytes_feature(data[i].tobytes()),
          'label': _int64_feature(labels[i])
        }))
      record_writer.write(example.SerializeToString())

# parsing the tf-record stored
def parse_record(serialized_example, isTraining = True):
  features = tf.parse_single_example(
    serialized_example,
    features={
      'image': tf.FixedLenFeature([], tf.string),
      'label': tf.FixedLenFeature([], tf.int64),
    })

  image = features['image']
  # decoding image data in bytes format to array
  image = tf.decode_raw(image, tf.float32)
  # reshape the image from linear list to image shape
  if(isTraining):
    image.set_shape([3 * 40 * 40])
    image = tf.reshape(image, [40, 40, 3])
  else:
    image.set_shape([3 * 32 * 32])
    image = tf.reshape(image, [32, 32, 3])
  
  #casting label data to integer format
  label = tf.cast(features['label'], tf.int64)

  return image, label

# IMAGE_HEIGHT = 40
# IMAGE_WIDTH = 40
# IMAGE_DEPTH = 3
# NUM_CLASSES = 10

def get_decoded_records(file_name, isTraining = True):
  
  # returns list of tuples each containing image and label
  dataset = tf.data.TFRecordDataset(filenames=file_name)
  dataset = dataset.map(lambda x: parse_record(x, isTraining))
  return dataset