from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import re
import time
from datetime import datetime
import os
import math

import tensorflow as tf
import numpy as np
import tensorflow.contrib.slim as slim


from utils import argparser, logging
from dataset import DataSet
from model import get_placeholders, inference



def test( FLAGS ):
  # read data
  dataset = DataSet( fpath = FLAGS.test_file, 
                      seqlen = FLAGS.seq_len,
                      n_classes = FLAGS.num_classes,
                      need_shuffle = False )
  # set character set size
  FLAGS.charset_size = dataset.charset_size

  with tf.Graph().as_default():
    # placeholder
    placeholders = get_placeholders(FLAGS)
    
    # get inference
    pred, layers = inference( placeholders['data'], FLAGS, 
                      for_training=False )

    # calculate prediction
    _hit_op = tf.equal( tf.argmax(pred, 1), tf.argmax(placeholders['labels'], 1))
    hit_op = tf.reduce_sum( tf.cast( _hit_op ,tf.float32 ) )

    # create saver
    saver = tf.train.Saver()

    # summary 
    summary_op = tf.summary.merge_all()

    with tf.Session() as sess:
      # load model
      ckpt = tf.train.latest_checkpoint( os.path.dirname( FLAGS.checkpoint_path ) )
      if tf.train.checkpoint_exists( ckpt ):
        saver.restore( sess, ckpt )
        global_step = ckpt.split('/')[-1].split('-')[-1]
        logging('Succesfully loaded model from %s at step=%s.' %
              (ckpt, global_step), FLAGS)
      else:
        logging("[ERROR] Checkpoint not exist", FLAGS)
        return

      # iter batch
      hit_count = 0.0
      total_count = 0

      logging("%s: starting test." % (datetime.now()), FLAGS)
      start_time = time.time()
      total_batch_size = math.ceil( dataset._num_data / FLAGS.batch_size )

      for step, (data, labels) in enumerate(dataset.iter_once( FLAGS.batch_size )):
        hits = sess.run( [hit_op], feed_dict={
          placeholders['data']: data,
          placeholders['labels']: labels
        })

        hit_count += np.sum( hits )
        total_count += len( data )

        if step % FLAGS.log_interval == 0:
          duration = time.time() - start_time
          sec_per_batch = duration / FLAGS.log_interval
          examples_per_sec = FLAGS.batch_size / sec_per_batch
          logging('%s: [%d batches out of %d] (%.1f examples/sec; %.3f'
                'sec/batch)' % (datetime.now(), step, total_batch_size,
                                examples_per_sec, sec_per_batch), FLAGS)
          start_time = time.time()


      # micro precision
      logging("%s: micro-precision = %.5f" % 
            (datetime.now(), (hit_count/total_count)), FLAGS)





if __name__ == '__main__':
  FLAGS = argparser()
  FLAGS.is_training = False
  logging(str(FLAGS), FLAGS)
  test( FLAGS )
