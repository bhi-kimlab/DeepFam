from Queue import Queue
from threading import Thread
from subprocess import call


import sys
import os

## get argv
num_classes = 1074
seqlen = 1000
# seqlen = 993



## globals
num_threads = 4
enclosure_queue = Queue()


## helpers
def mkdir(d):
  if not os.path.exists(d):
    os.makedirs(d)

def run(i, queue):
  while True:
    # print("%d: Looking for next command" % i)
    cmd = queue.get()
    cmd = "export TF_CPP_MIN_LOG_LEVEL=2; export CUDA_VISIBLE_DEVICES=%d; %s" % (i, cmd)
    # cmd = "export CUDA_VISIBLE_DEVICES=%d; %s" % (7, cmd)
    call(cmd, shell=True)
    queue.task_done()



####### hyperpara search #########
# hidden1 = [ 800, 1400, 2000 ]
# hidden2 = [ 800, 1400, 2000 ]
# indir = os.path.join( "/home/kimlab/project/DeepFam/src/DeepFam", "test1" )
# outdir = "/home/kimlab/project/CCC/HyperRES"


# for h1 in hidden1:
#   for h2 in hidden2:
#     expname = os.path.join( outdir, "%d-%d" % (h1, h2) )
#     logdir = os.path.join( expname, "logs" )
#     mkdir( os.path.join( logdir, "train" ) )
#     mkdir( os.path.join( logdir, "test" ) )

#     ckptdir = os.path.join( expname, "save" )
#     ckptfile = os.path.join( ckptdir, "model.ckpt" )
#     mkdir( ckptdir )


#     # cmd
#     s="python %s" % os.path.join( os.path.dirname(os.path.realpath(__file__)), "run_gpu.py" )
#     s+= " --num_classes=%s --seq_len=%s" % ( num_classes, seqlen )
#     s+= " --window_lengths=%s" % ( " ".join([8, 12, 16, 20, 24, 28, 32, 36]) ) 
#     s+= " --num_windows=%s" % ( " ".join([256, 256, 256, 256, 256, 256, 256, 256]) ) 
#     s+= " --num_hidden=%d --max_epoch=%d" % ( 2000, 20 )
#     s+= " --train_file=%s" % os.path.join( indir, "train.txt" )
#     s+= " --test_file=%s" % os.path.join( indir, "test.txt" )
#     s+= " --checkpoint_path=%s --log_dir=%s" % (ckptfile, logdir)
#     s+= " --batch_size=%d" % (100)
#     # s+= " --prev_checkpoint_path=/home/kimlab/project/CCC/RES/otest-1665-1536-0.001000/save-158019"
#     # s+= " --fine_tuning=True --fine_tuning_layers=fc2"
#     enclosure_queue.put( s )











####### general task #########
# # tests = ["test1", "test2", "test3"]
# tests = ["90percent"]
# indirs = [ os.path.join( "/home/kimlab/project/CCC/TASK_NEW", p ) for p in tests ]
# outdirs = [ os.path.join( "/home/kimlab/project/CCC/RES_WIDE", p ) for p in tests ]

# for idx, expname in enumerate(outdirs):
tests = ["90percent", "test1", "test2", "test3"]

for exp in tests:
  indir = os.path.join( "/home/kimlab/project/DeepFam", "data", exp )
  outdir = os.path.join( "/home/kimlab/project/DeepFam", "result" )
  expname = os.path.join( outdir, exp )

  logdir = os.path.join( expname, "logs" )
  mkdir( os.path.join( logdir, "train" ) )
  mkdir( os.path.join( logdir, "test" ) )

  ckptdir = os.path.join( expname, "save" )
  ckptfile = os.path.join( ckptdir, "model.ckpt" )
  mkdir( ckptdir )


  # cmd
  s="python %s" % os.path.join( os.path.dirname(os.path.realpath(__file__)), "run.py" )
  s+= " --num_classes %s --seq_len %s" % ( num_classes, seqlen )
  s+= " --window_lengths %s" % ( " ".join(['8', '12', '16', '20', '24', '28', '32', '36']) ) 
  s+= " --num_windows %s" % ( " ".join(['256', '256', '256', '256', '256', '256', '256', '256']) ) 
  s+= " --num_hidden %d --max_epoch %d" % ( 2000, 25 )
  s+= " --train_file %s" % os.path.join( indir, "train.txt" )
  s+= " --test_file %s" % os.path.join( indir, "test.txt" )
  s+= " --checkpoint_path %s --log_dir %s" % (ckptfile, logdir)
  s+= " --batch_size %d" % (100)
  # s+= " --prev_checkpoint_path=/home/kimlab/project/CCC/RES/otest-1665-1536-0.001000/save-158019"
  # s+= " --fine_tuning=True --fine_tuning_layers fc2"
  enclosure_queue.put( s )







# ####### fine-tuning task #########
# tests = ["cv1", "cv2", "cv3", "cv4", "cv5"]
# indirs = [ os.path.join( "/home/kimlab/project/CCC/tmp/tbpred/data", p ) for p in tests ]
# outdirs = [ os.path.join( "/home/kimlab/project/DeepFam/result/tbpred", p ) for p in tests ]

# for idx, expname in enumerate(outdirs):
#   indir = indirs[idx]

#   logdir = os.path.join( expname, "logs" )
#   mkdir( os.path.join( logdir, "train" ) )
#   mkdir( os.path.join( logdir, "test" ) )

#   ckptdir = os.path.join( expname, "save" )
#   ckptfile = os.path.join( ckptdir, "save" )
#   mkdir( ckptdir )


#   # cmd
#   s="python %s" % os.path.join( os.path.dirname(os.path.realpath(__file__)), "run.py" )
#   s+= " --num_classes %s --seq_len %s" % ( num_classes, seqlen )
#   s+= " --window_lengths %s" % ( " ".join(['8', '12', '16', '20', '24', '28', '32', '36']) ) 
#   s+= " --num_windows %s" % ( " ".join(['256', '256', '256', '256', '256', '256', '256', '256']) ) 
#   s+= " --num_hidden %d --max_epoch %d" % ( 2000, 20 )
#   s+= " --train_file %s" % os.path.join( indir, "train.txt" )
#   s+= " --test_file %s" % os.path.join( indir, "test.txt" )
#   s+= " --checkpoint_path %s --log_dir %s" % (ckptfile, logdir)
#   s+= " --batch_size %d" % (100)
#   s+= " --prev_checkpoint_path=/home/kimlab/project/DeepFam/result/90percent/save/save-203219"
#   s+= " --fine_tuning True --fine_tuning_layers %s" % ( " ".join(['fc2']) )
#   # s+= " --prev_checkpoint_path=/home/kimlab/project/CCC/RES/otest-1665-1536-0.001000/save-158019"
#   # s+= " --fine_tuning=True --fine_tuning_layers fc2"
#   enclosure_queue.put( s )







## main
for i in range( num_threads ):
  worker = Thread( target=run, args=(i, enclosure_queue,) )
  worker.setDaemon(True)
  worker.start()


## run
enclosure_queue.join()