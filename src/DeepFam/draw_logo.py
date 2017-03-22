from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import re
import time
from datetime import datetime
import os
import math
from heapq import *
from collections import defaultdict

import tensorflow as tf
import numpy as np
import tensorflow.contrib.slim as slim


from utils import argparser
from dataset import DataSet
from model import get_placeholders, inference




def test( FLAGS ):
  # read data
  dataset = DataSet( fpath = FLAGS.test_file, 
                      seqlen = FLAGS.seq_len,
                      n_classes = FLAGS.num_classes,
                      need_shuffle = False )

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

    # argmax of hidden1
    h1_argmax_ops = []
    for op in layers['conv']:
      h1_argmax_ops.append(tf.argmax(op, axis=2))


    with tf.Session() as sess:
      # load model
      ckpt = tf.train.latest_checkpoint( os.path.dirname( FLAGS.checkpoint_path ) )
      if tf.train.checkpoint_exists( ckpt ):
        saver.restore( sess, ckpt )
        global_step = ckpt.split('/')[-1].split('-')[-1]
        print('Succesfully loaded model from %s at step=%s.' %
              (ckpt, global_step))
      else:
        print("[ERROR] Checkpoint not exist")
        return


      # iter batch
      hit_count = 0.0
      total_count = 0
      # top_matches = [ ([], []) ] * FLAGS.hidden1 # top 100 matching proteins
      wlens = [4, 8, 12, 16, 20]
      hsize = int(FLAGS.hidden1 / 5)
      motif_matches = (defaultdict(list), defaultdict(list))

      print("%s: starting test." % (datetime.now()))
      start_time = time.time()
      total_batch_size = math.ceil( dataset._num_data / FLAGS.batch_size )

      for step, (data, labels, raws) in enumerate(dataset.iter_once( FLAGS.batch_size, with_raw=True )):
        res_run = sess.run( [hit_op, h1_argmax_ops] + layers['conv'], feed_dict={
          placeholders['data']: data,
          placeholders['labels']: labels
        })

        hits = res_run[0]
        max_idxs = res_run[1] # shape = (wlens, N, 1, # of filters)
        motif_filters = res_run[2:]


        # mf.shape = (N, 1, l-w+1, # of filters)
        for i in range(len(motif_filters)):
          s = motif_filters[i].shape
          motif_filters[i] = np.transpose( motif_filters[i], (0, 1, 3, 2) ).reshape( (s[0], s[3], s[2]) )

        # mf.shape = (N, # of filters, l-w+1)
        for gidx, mf in enumerate(motif_filters):
          wlen = wlens[gidx]
          for ridx, row in enumerate(mf):
            for fidx, vals in enumerate(row):
              # for each filter, get max value and it's index
              max_idx = max_idxs[gidx][ridx][0][fidx]
              # max_idx = np.argmax(vals)
              max_val = vals[ max_idx ]

              hidx = gidx * hsize + fidx

              if max_val > 0:
                # get sequence
                rawseq = raws[ridx][1]
                subseq = rawseq[ max_idx : max_idx+wlen ]
                # heappush( top_matches[hidx], (max_val, subseq) )
                motif_matches[0][hidx].append( max_val )
                motif_matches[1][hidx].append( subseq )
                # motif_matches[gidx][fidx][0].append( max_val )
                # motif_matches[gidx][fidx][1].append( subseq )


        hit_count += np.sum( hits )
        total_count += len( data )
        # print("total:%d" % total_count)

        if step % FLAGS.log_interval == 0:
          duration = time.time() - start_time
          sec_per_batch = duration / FLAGS.log_interval
          examples_per_sec = FLAGS.batch_size / sec_per_batch
          print('%s: [%d batches out of %d] (%.1f examples/sec; %.3f'
                'sec/batch)' % (datetime.now(), step, total_batch_size,
                                examples_per_sec, sec_per_batch))
          start_time = time.time()

        # if step > 10:
        #   break


      # # micro precision
      # print("%s: micro-precision = %.5f" % 
      #       (datetime.now(), (hit_count/total_count)))
  
        
      ### sort top lists
      print('%s: write result to file' % (datetime.now()) )
      for fidx in motif_matches[0]:
        val_lst = motif_matches[0][fidx]
        seq_lst = motif_matches[1][fidx]
        # top k
        k = wlens[ int(fidx / hsize) ] * 25
        l = min(k, len(val_lst)) * -1
        tidxs = np.argpartition(val_lst, l)[l:]
        with open("/home/kimlab/project/CCC/tmp/logos/test/p%d.txt"%fidx, 'w') as fw:
          for idx in tidxs:
            fw.write("%f\t%s\n" % (val_lst[idx], seq_lst[idx]) )

        if fidx % 50 == 0:
          print('%s: [%d filters out of %d]' % (datetime.now(), fidx, FLAGS.hidden1))
          # print(len(val_lst))





if __name__ == '__main__':
  FLAGS = argparser()
  FLAGS.is_training = False

  test( FLAGS )
