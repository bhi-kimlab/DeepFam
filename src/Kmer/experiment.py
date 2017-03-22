from Queue import Queue
from threading import Thread
from subprocess import call


import sys
import os

## get argv
# outdir = sys.argv[1]
# indir = sys.argv[2]
num_classes = 1074



## globals
num_threads = 3
enclosure_queue = Queue()


## helpers
def mkdir(d):
  if not os.path.exists(d):
    os.makedirs(d)

def run(i, queue):
  while True:
    # print("%d: Looking for next command" % i)
    cmd = queue.get()
    cmd = "export CUDA_VISIBLE_DEVICES=%d; %s" % (i, cmd)
    # cmd = "export CUDA_VISIBLE_DEVICES=%d; %s" % (0, cmd)
    call(cmd, shell=True)
    queue.task_done()



k = 3
tests = ["test1", "test2", "test3"]
indirs = [ os.path.join( "/home/kimlab/project/DeepFam/data", p ) for p in tests ]
outdirs = [ os.path.join( "/home/kimlab/project/DeepFam/Kmer", p ) for p in tests ]


for idx, expname in enumerate(outdirs):
  indir = indirs[idx]
  logdir = os.path.join( expname, "logs" )
  mkdir( os.path.join( logdir, "train" ) )
  mkdir( os.path.join( logdir, "test" ) )

  ckptdir = os.path.join( expname, "save" )
  ckptfile = os.path.join( ckptdir, "model.ckpt" )
  mkdir( ckptdir )

  # cmd
  s="python %s" % os.path.join( os.path.dirname(os.path.realpath(__file__)), "run.py" )
  s+= " --num_classes=%s " % ( num_classes )
  s+= " --k=%d " % ( k )
  s+= " --max_epoch=30"
  s+= " --train_file=%s" % os.path.join( indir, "train.txt" )
  s+= " --test_file=%s" % os.path.join( indir, "test.txt" )
  s+= " --checkpoint_path=%s --log_dir=%s" % (ckptdir, logdir)
  s+= " --batch_size=100"

  enclosure_queue.put( s )





## main
# for i in range( num_threads ):
for i in [4, 5, 6]:
  worker = Thread( target=run, args=(i, enclosure_queue,) )
  worker.setDaemon(True)
  worker.start()


## run
enclosure_queue.join()