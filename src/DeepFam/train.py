from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import re
import time
from datetime import datetime
import os


import tensorflow as tf
import numpy as np
import tensorflow.contrib.slim as slim

from utils import argparser, logging
from dataset import DataSet
from model import get_placeholders, inference



# ########## #
#
# Single-gpu version training.
#
# ########## #
def train( FLAGS ):
  # read data
  dataset = DataSet( fpath = FLAGS.train_file, 
                      seqlen = FLAGS.seq_len,
                      n_classes = FLAGS.num_classes,
                      need_shuffle = True )
  # set character set size
  FLAGS.charset_size = dataset.charset_size

  with tf.Graph().as_default():
    # get placeholders
    global_step = tf.placeholder( tf.int32 )
    placeholders = get_placeholders(FLAGS)

    # prediction
    pred, layers = inference( placeholders['data'], FLAGS,
                              for_training=True )
    # loss
    # slim.losses.softmax_cross_entropy(pred, placeholders['labels'])
    # class_weight = tf.constant([[1.0, 5.0]])
    # weight_per_label = tf.transpose( tf.matmul(placeholders['labels']
    #                        , tf.transpose(class_weight)) )
    # loss = tf.multiply(weight_per_label, 
    #         tf.nn.softmax_cross_entropy_with_logits(labels=placeholders['labels'], logits=pred))
    # loss = tf.losses.compute_weighted_loss(loss)

    tf.losses.softmax_cross_entropy(placeholders['labels'], pred)
    loss = tf.losses.get_total_loss()

    # accuracy
    _acc_op = tf.equal( tf.argmax(pred, 1), tf.argmax(placeholders['labels'], 1))
    acc_op = tf.reduce_mean( tf.cast( _acc_op ,tf.float32 ) )
    
    # optimization
    train_op = tf.train.AdamOptimizer( FLAGS.learning_rate ).minimize( loss )
    # train_op = tf.train.RMSPropOptimizer( FLAGS.learning_rate ).minimize( loss )

    # Create a saver.
    saver = tf.train.Saver(max_to_keep=None)

    with tf.Session() as sess:
      sess.run( tf.global_variables_initializer() )

      if tf.train.checkpoint_exists( FLAGS.prev_checkpoint_path ):
        if FLAGS.fine_tuning:
          logging('%s: Fine Tuning Experiment!' %
              (datetime.now()), FLAGS)
          restore_variables = slim.get_variables_to_restore(exclude=FLAGS.fine_tuning_layers)
          restorer = tf.train.Saver( restore_variables )
        else:
          restorer = tf.train.Saver()
        restorer.restore(sess, FLAGS.prev_checkpoint_path)
        logging('%s: Pre-trained model restored from %s' %
            (datetime.now(), FLAGS.prev_checkpoint_path), FLAGS)
        step = int(FLAGS.prev_checkpoint_path.split('/')[-1].split('-')[-1]) + 1
      else:
        step = 0

      # iter epoch
      # for data, labels in dataset.iter_batch( FLAGS.batch_size, 5 ):
      for data, labels in dataset.iter_once( FLAGS.batch_size ):
        start_time = time.time()
        _, loss_val, acc_val = sess.run([train_op, loss, acc_op], feed_dict={
          placeholders['data']: data,
          placeholders['labels']: labels,
          global_step: step
        })
        duration = time.time() - start_time

        assert not np.isnan(loss_val), 'Model diverge'

        # logging
        if step > 0 and step % FLAGS.log_interval == 0:
          examples_per_sec = FLAGS.batch_size / float(duration)
          format_str = ('%s: step %d, loss = %.2f, acc = %.2f (%.1f examples/sec; %.3f '
                      'sec/batch)')
          logging(format_str % (datetime.now(), step, loss_val, acc_val,
                            examples_per_sec, duration), FLAGS)

        # save model
        if step > 0 and step % FLAGS.save_interval == 0:
          saver.save(sess, FLAGS.checkpoint_path, global_step=step)

        # counter
        step += 1

      # save for last
      saver.save(sess, FLAGS.checkpoint_path, global_step=step-1)




if __name__ == '__main__':
  FLAGS = argparser()
  FLAGS.is_training = True
  logging(str(FLAGS), FLAGS)

  train( FLAGS )




