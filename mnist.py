#!/usr/bin/env python

import os
from argparse import ArgumentParser
import tensorflow as tf
import tensorflow.keras.layers as l
import json
    
def _tuple_to_feat_label(x, y):
  x = tf.expand_dims(x, axis=-1)
  x = tf.image.convert_image_dtype(x, dtype=tf.float32)
  y = tf.cast(y, dtype=tf.int32)
  return { 'image': x }, y
  
def make_dataset(xs, ys, batch_size=1024, shuffle=False, num_epochs=None):
  
  dataset = tf.data.Dataset.from_tensor_slices((xs, ys))
  if shuffle:
    dataset = dataset.apply(tf.contrib.data.shuffle_and_repeat(
      buffer_size=4096, count=num_epochs))
  else:
    dataset = dataset.repeat(num_epochs)
      
  dataset = dataset.apply(tf.contrib.data.map_and_batch(
    _tuple_to_feat_label, batch_size=batch_size))
  
  return dataset
  
def mnist_cnn_model_fn(features, labels, mode, params):

  is_training = mode == tf.estimator.ModeKeys.TRAIN
  
  if is_training:
    dropout_rate = params['dropout_rate'] if 'dropout_rate' in params else 0.4
  else:
    dropout_rate = 0
  
  learning_rate = params['learning_rate'] if 'learning_rate' in params else 1e-3
  decay_rate = params['decay_rate'] if 'decay_rate' in params else 1e-6
  decay_steps = params['decay_steps'] if 'decay_steps' in params else 500
  
  conv1 = l.Conv2D(filters=32, kernel_size=5, padding='same', activation=tf.nn.relu)(features['image'])
  pool1 = l.MaxPooling2D(pool_size=2, strides=2, padding='same')(conv1)
  
  conv2 = l.Conv2D(filters=64, kernel_size=5, padding='same', activation=tf.nn.relu)(pool1)
  pool2 = l.MaxPooling2D(pool_size=2, strides=2, padding='same')(conv2)
  
  pool2_flat = l.Flatten()(pool2)
  dense = l.Dense(1024, activation=tf.nn.relu)(pool2_flat)
  dropout = l.Dropout(rate=dropout_rate)(dense)
  
  logits = l.Dense(10)(dropout)
  
  head = tf.contrib.estimator.multi_class_head(
    n_classes=10)
  
  decayed_learning_rate = tf.train.exponential_decay(
    learning_rate, tf.train.get_global_step(), 
    decay_steps, decay_rate, staircase=True)

  optimizer = tf.train.AdamOptimizer(decayed_learning_rate)
  
  return head.create_estimator_spec(
    features, mode, logits, labels, optimizer=optimizer,
  )

def remap_tfconfig(is_master):
  tf_config = json.loads(os.environ['TF_CONFIG'])
  master_worker = tf_config['cluster']['worker'][0]
  tf_config['cluster']['worker'] = tf_config['cluster']['worker'][1:]
  tf_config['cluster']['chief'] = [master_worker]
  if is_master:
    tf_config['task']['type'] = 'chief'
    tf_config['task']['index'] = 0
  elif tf_config['task']['type'] == 'worker':
    tf_config['task']['index'] -= 1
  
  os.environ['TF_CONFIG'] = json.dumps(tf_config)

def main():

  parser = ArgumentParser()
  parser.add_argument('--master', action='store_true')
  parser.add_argument('--model-directory')
  args, other_args = parser.parse_known_args()
  remap_tfconfig(args.master)

  tf.logging.set_verbosity(tf.logging.INFO)

  (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
  train_ds = lambda: make_dataset(x_train, y_train, shuffle=True)
  test_ds = lambda: make_dataset(x_test, y_test)
  
  estimator = tf.estimator.Estimator(
    mnist_cnn_model_fn, 
    model_dir=args.model_directory, 
    params={
      'learning_rate': 1e-3,
      'dropout_rate': 0.4,
  })

  train_spec = tf.estimator.TrainSpec(train_ds, max_steps=1000)
  eval_spec = tf.estimator.EvalSpec(test_ds)
  tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)

if __name__ == '__main__':
  main()