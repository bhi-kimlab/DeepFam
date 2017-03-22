import argparse
import os



def argparser():
  parser = argparse.ArgumentParser()
  # for model
  parser.add_argument(
      '--embedding_file',
      type=str,
      help='ProtVec embedding file.'
  )
  parser.add_argument(
      '--regularizer',
      type=float,
      default=0.001,
      help='(Lambda value / 2) of L2 regularizer on weights connected to last layer (0 to exclude).'
  )
  parser.add_argument(
      '--num_classes',
      type=int,
      default=0,
      help='Number of classes.'
  )
  # for learning
  parser.add_argument(
      '--learning_rate',
      type=float,
      default=0.001,
      help='Initial learning rate.'
  )
  parser.add_argument(
      '--max_epoch',
      type=int,
      default=1,
      help='Number of epochs to train.'
  )
  parser.add_argument(
      '--batch_size',
      type=int,
      default=64,
      help='Batch size.'
  )
  parser.add_argument(
      '--train_file',
      type=str,
      default='/tmp/tensorflow/mnist/input_data',
      help='Directory for input data.'
  )
  parser.add_argument(
      '--test_file',
      type=str,
      default='/tmp/tensorflow/mnist/input_data',
      help='Directory for input data.'
  )
  parser.add_argument(
      '--prev_checkpoint_path',
      type=str,
      default='',
      help='Restore from pre-trained model if specified.'
  )
  parser.add_argument(
      '--checkpoint_path',
      type=str,
      default='',
      help='Path to write checkpoint file.'
  )
  parser.add_argument(
      '--log_dir',
      type=str,
      default='/tmp/tensorflow/mnist/logs/fully_connected_feed',
      help='Directory for log data.'
  )
  parser.add_argument(
      '--log_interval',
      type=int,
      default=100,
      help='Number of gpus to use'
  )
  parser.add_argument(
      '--save_interval',
      type=int,
      default=4000,
      help='Number of gpus to use'
  )

  FLAGS, unparsed = parser.parse_known_args()

  return FLAGS






def logging(msg, FLAGS):
  fpath = os.path.join( FLAGS.log_dir, "log.txt" )
  with open( fpath, "a" ) as fw:
    fw.write("%s\n" % msg)
  print(msg)