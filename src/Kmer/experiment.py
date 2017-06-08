from Queue import Queue
from threading import Thread
from subprocess import call


import sys
import os

## get argv
# outdir = sys.argv[1]
# indir = sys.argv[2]
num_classes = 1074
# num_classes = 107



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
    cmd = "export TF_CPP_MIN_LOG_LEVEL=2; export CUDA_VISIBLE_DEVICES=%d; %s" % (i, cmd)
    # cmd = "export CUDA_VISIBLE_DEVICES=%d; %s" % (0, cmd)
    call(cmd, shell=True)
    queue.task_done()




# ## general
# k = 3
# tests = ["90percent", "dataset1", "dataset2", "dataset0"]
# tests = ["dataset0"]

# for exp in tests:
#   indir = os.path.join( "/home/kimlab/project/DeepFam", "data", "cog_cv", exp )
#   outdir = os.path.join( "/home/kimlab/project/DeepFam", "result", "Kmer" )
#   expname = os.path.join( outdir, exp )

#   logdir = os.path.join( expname, "logs" )
#   mkdir( os.path.join( logdir, "train" ) )
#   mkdir( os.path.join( logdir, "test" ) )

#   ckptdir = os.path.join( expname, "save" )
#   ckptfile = os.path.join( ckptdir, "model.ckpt" )
#   mkdir( ckptdir )


#   s="python %s" % os.path.join( os.path.dirname(os.path.realpath(__file__)), "run.py" )
#   s+= " --num_classes=%s " % ( num_classes )
#   s+= " --k=%d " % ( k )
#   s+= " --max_epoch %d" % (25)
#   s+= " --train_file=%s" % os.path.join( indir, "train.txt" )
#   s+= " --test_file=%s" % os.path.join( indir, "test.txt" )
#   s+= " --checkpoint_path=%s --log_dir=%s" % (ckptfile, logdir)
#   s+= " --batch_size=100"
#   s+= " --log_interval %d --save_interval %d" % (100, 10000)

#   enclosure_queue.put( s )




# num_classes = 1796
# seqlen = 1000

num_classes = 2892
seqlen = 1000

k = 3
tests = ["dataset0", "dataset1", "dataset2"]

for exp in tests:
  indir = os.path.join( "/home/kimlab/project/DeepFam", "data", "l_1000_s_100", exp )
  outdir = os.path.join( "/home/kimlab/project/DeepFam", "result", "l_1000_s_100_kmer" )
  expname = os.path.join( outdir, exp )

  logdir = os.path.join( expname, "logs" )
  mkdir( os.path.join( logdir, "train" ) )
  mkdir( os.path.join( logdir, "test" ) )

  ckptdir = os.path.join( expname, "save" )
  ckptfile = os.path.join( ckptdir, "model.ckpt" )
  mkdir( ckptdir )


  s="python %s" % os.path.join( os.path.dirname(os.path.realpath(__file__)), "run.py" )
  s+= " --num_classes=%s " % ( num_classes )
  s+= " --k=%d " % ( k )
  s+= " --max_epoch %d" % (20)
  s+= " --train_file=%s" % os.path.join( indir, "train.txt" )
  s+= " --test_file=%s" % os.path.join( indir, "test.txt" )
  s+= " --checkpoint_path=%s --log_dir=%s" % (ckptfile, logdir)
  s+= " --batch_size=100"
  s+= " --log_interval %d --save_interval %d" % (100, 10000)

  enclosure_queue.put( s )









# ## GPCR
# k = 3
# tests = [ "cv_%d" % i for i in range(10) ]
# indirs = [ os.path.join( "/home/kimlab/project/DeepFam/tmp/gpcr/data/subfamily_level", p ) for p in tests ]
# outdirs = [ os.path.join( "/home/kimlab/project/DeepFam/tmp/gpcr/result_kmer", p ) for p in tests ]

# for idx, expname in enumerate(outdirs):
#   indir = indirs[idx]

#   logdir = os.path.join( expname, "logs" )
#   mkdir( os.path.join( logdir, "train" ) )
#   mkdir( os.path.join( logdir, "test" ) )

#   ckptdir = os.path.join( expname, "save" )
#   ckptfile = os.path.join( ckptdir, "model.ckpt" )
#   mkdir( ckptdir )


#   s="python %s" % os.path.join( os.path.dirname(os.path.realpath(__file__)), "run.py" )
#   s+= " --num_classes=%s " % ( num_classes )
#   s+= " --k=%d " % ( k )
#   s+= " --max_epoch %d" % (20)
#   s+= " --train_file=%s" % os.path.join( indir, "train.txt" )
#   s+= " --test_file=%s" % os.path.join( indir, "test.txt" )
#   s+= " --checkpoint_path=%s --log_dir=%s" % (ckptfile, logdir)
#   s+= " --batch_size=100"

#   enclosure_queue.put( s )








## main
# for i in range( num_threads ):
for i in range( 4, 4+num_threads ):
  worker = Thread( target=run, args=(i, enclosure_queue,) )
  worker.setDaemon(True)
  worker.start()


## run
enclosure_queue.join()