import tensorflow as tf
import numpy as np
import tensorflow.contrib.slim as slim

import re



def inference(data, FLAGS, for_training=False, scope=None):
  pred = network( data, FLAGS, 
                  is_training=for_training,
                  scope=scope)

  return pred


def get_placeholders(FLAGS):
  placeholders = {}
  placeholders['data'] = tf.placeholder(tf.float32, [None, FLAGS.word_size])
  placeholders['labels'] = tf.placeholder(tf.float32, [None, FLAGS.num_classes])
  return placeholders





def network(data, FLAGS, is_training=True, scope=''):
  ### 
  # define layers
  ###
  net = slim.fully_connected( data,
                              FLAGS.num_classes,
                              activation_fn=None,
                              biases_initializer=tf.constant_initializer(0.1),
                              normalizer_fn=None,
                              normalizer_params=None,
                              weights_initializer=tf.contrib.layers.xavier_initializer(),
                              weights_regularizer=slim.l2_regularizer(FLAGS.regularizer) if FLAGS.regularizer > 0 else None,
                              scope="fc" )

  return net

