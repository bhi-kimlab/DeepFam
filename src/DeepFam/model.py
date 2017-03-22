import tensorflow as tf
import numpy as np
import tensorflow.contrib.slim as slim

import re



def inference(data, FLAGS, for_training=False, scope=None):
  pred, layers = network( data, FLAGS, 
                          is_training=for_training,
                          scope=scope)

  return pred, layers


def get_placeholders(FLAGS):
  placeholders = {}
  placeholders['data'] = tf.placeholder(tf.float32, [None, FLAGS.seq_len * FLAGS.charset_size])
  placeholders['labels'] = tf.placeholder(tf.float32, [None, FLAGS.num_classes])
  return placeholders




def network(data, FLAGS, is_training=True, scope=''):
  batch_norm_params = {
    'decay': 0.9, # might problem if too small updates
    'is_training': FLAGS.is_training,
    'updates_collections': None
  }

  layers = {}

  x_data = tf.reshape( data, [-1, 1, FLAGS.seq_len, FLAGS.charset_size] )


  ### 
  # define layers
  ###

  with tf.name_scope(scope, 'v1', [x_data]):
    with slim.arg_scope([slim.conv2d, slim.fully_connected],
                          activation_fn=tf.nn.relu,
                          biases_initializer=tf.constant_initializer(0.1),
                          normalizer_fn=slim.batch_norm,
                          normalizer_params=batch_norm_params,
                          weights_initializer=tf.contrib.layers.xavier_initializer() ):
      
      layers['conv'] = []
      layers['hidden1'] = []
      for i, wlen in enumerate(FLAGS.window_lengths):
        layers['conv'].append ( slim.conv2d( x_data,
                                      FLAGS.num_windows[i],
                                      [1, wlen],
                                      # padding='SAME',
                                      padding='VALID',
                                      weights_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                      scope="conv%d" % i ) )
        # max pooling
        max_pooled = slim.max_pool2d( layers['conv'][i],
                                      # [1, FLAGS.seq_len],
                                      [1, FLAGS.seq_len - wlen + 1],
                                      stride=[1, FLAGS.seq_len],
                                      padding='VALID',
                                      scope="pool%d" % i )
        # reshape
        layers['hidden1'].append( slim.flatten( max_pooled, scope="flatten%d" % i) )

      # concat
      layers['concat'] = tf.concat( layers['hidden1'], 1 )

      # dropout
      dropped = slim.dropout( layers['concat'], 
                              keep_prob=FLAGS.keep_prob,
                              is_training=FLAGS.is_training,
                              scope="dropout" )

      # fc layers
      layers['hidden2'] = slim.fully_connected( dropped,
                                                FLAGS.num_hidden,
                                                weights_regularizer=None,
                                                scope="fc1" )

      layers['pred'] = slim.fully_connected( layers['hidden2'],
                                            FLAGS.num_classes,
                                            activation_fn=None,
                                            normalizer_fn=None,
                                            normalizer_params=None,
                                            weights_regularizer=slim.l2_regularizer(FLAGS.regularizer) if FLAGS.regularizer > 0 else None,
                                            scope="fc2" )

  return layers['pred'], layers



