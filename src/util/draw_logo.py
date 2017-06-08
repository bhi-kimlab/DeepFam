from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import re
import time
from datetime import datetime
import os
import math
import sys
from collections import defaultdict

import tensorflow as tf
import numpy as np
import tensorflow.contrib.slim as slim


sys.path.append( os.path.join( os.path.dirname(__file__), "..", "DeepFam" ) )
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
    label_op = tf.argmax(pred, 1)
    prob_op = tf.nn.softmax(pred)
    # _hit_op = tf.equal( tf.argmax(pred, 1), tf.argmax(placeholders['labels'], 1))
    # hit_op = tf.reduce_sum( tf.cast( _hit_op ,tf.float32 ) )

    # create saver
    saver = tf.train.Saver()

    # argmax of hidden1
    h1_argmax_ops = []
    for op in layers['conv']:
      h1_argmax_ops.append(tf.argmax(op, axis=2))


    with tf.Session() as sess:
      # load model
      # ckpt = tf.train.latest_checkpoint( os.path.dirname( FLAGS.checkpoint_path ) )
      ckpt = FLAGS.checkpoint_path
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

      wlens = FLAGS.window_lengths
      hsizes = FLAGS.num_windows
      motif_matches = (defaultdict(list), defaultdict(list))
      pred_labels = []
      pred_prob = []

      print("%s: starting test." % (datetime.now()))
      start_time = time.time()
      total_batch_size = math.ceil( dataset._num_data / FLAGS.batch_size )

      for step, (data, labels, raws) in enumerate(dataset.iter_once( FLAGS.batch_size, with_raw=True )):
        res_run = sess.run( [label_op, prob_op, h1_argmax_ops] + layers['conv'], feed_dict={
          placeholders['data']: data,
          placeholders['labels']: labels
        })

        pred_label = res_run[0]
        pred_arr = res_run[1]
        max_idxs = res_run[2] # shape = (wlens, N, 1, # of filters)
        motif_filters = res_run[3:]

        for i, l in enumerate(pred_label):
          pred_labels.append(l)
          pred_prob.append( pred_arr[i][ 388 ] )

        # mf.shape = (N, 1, l-w+1, # of filters)
        for i in range(len(motif_filters)):
          s = motif_filters[i].shape
          motif_filters[i] = np.transpose( motif_filters[i], (0, 1, 3, 2) ).reshape( (s[0], s[3], s[2]) )

        # mf.shape = (N, # of filters, l-w+1)
        for gidx, mf in enumerate(motif_filters):
          wlen = wlens[gidx]
          hsize = hsizes[gidx]
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


        # hit_count += np.sum( hits )
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
      
      print(pred_labels)
        
      ### sort top lists
      # report whose activation was higher
      mean_acts = {}
      on_acts = {}
      print('%s: write result to file' % (datetime.now()) )
      for fidx in motif_matches[0]:
        val_lst = motif_matches[0][fidx]
        seq_lst = motif_matches[1][fidx]
        # top k
        # k = wlens[ int(fidx / hsize) ] * 25
        k = 30
        # l = min(k, len(val_lst)) * -1
        l = len(val_lst) * -1
        tidxs = np.argpartition(val_lst, l)[l:]
        # tracking acts
        acts = 0.0

        opath = os.path.join(FLAGS.motif_outpath, "p%d.txt"%fidx)
        with open(opath, 'w') as fw:
          for idx in tidxs:
            fw.write("%f\t%s\n" % (val_lst[idx], seq_lst[idx]) )
            acts += val_lst[idx]

        mean_acts[fidx] = acts / l * -1
        on_acts[fidx] = len(val_lst)

        if fidx % 50 == 0:
          print('%s: [%d filters out of %d]' % (datetime.now(), fidx, sum(FLAGS.num_windows)))
          # print(len(val_lst))

      # report mean acts
      with open(os.path.join(FLAGS.motif_outpath, "report.txt"), 'w') as fw:
        for i in sorted(on_acts, key=on_acts.get, reverse=True):
          fw.write("%f\t%f\t%d\n" % (on_acts[i] / total_count, mean_acts[i], i))

      with open(os.path.join(FLAGS.motif_outpath, "predictions.txt"), 'w') as fw:
        for i, p in enumerate(pred_labels):
          fw.write("%s\t%f\n" % (str(p), pred_prob[i]))







if __name__ == '__main__':
  FLAGS = argparser()
  FLAGS.is_training = False
  FLAGS.motif_outpath = FLAGS.log_dir
  if not os.path.exists(FLAGS.motif_outpath):
    os.mkdir(FLAGS.motif_outpath)

  test( FLAGS )
