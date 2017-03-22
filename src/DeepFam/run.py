import re
import time
from datetime import datetime
import os

import tensorflow as tf

from utils import argparser, logging
from train import train
from test import test


if __name__ == '__main__':
  FLAGS = argparser()
  logging(str(FLAGS), FLAGS)

  logdir = FLAGS.log_dir

  ### iterate over max_epoch
  for i in range( FLAGS.max_epoch ):
    logging("%s: Run epoch %d" % (datetime.now(), i), FLAGS)
    # for training
    FLAGS.is_training = True
    FLAGS.keep_prob = 0.7
    FLAGS.log_dir = os.path.join( logdir, "train" )
    train( FLAGS )

    # for test
    FLAGS.is_training = False
    FLAGS.keep_prob = 1.0
    FLAGS.log_dir = os.path.join( logdir, "test" )
    test( FLAGS )

    FLAGS.prev_checkpoint_path = tf.train.latest_checkpoint( os.path.dirname( FLAGS.checkpoint_path ) )
    FLAGS.fine_tuning = False