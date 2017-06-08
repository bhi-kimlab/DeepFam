from Queue import Queue
from threading import Thread
from subprocess import call


import sys
import os

## get argv
num_classes = 2892
seqlen = 1000

# num_classes = 1796
# seqlen = 1000

# num_classes = 5
# num_classes = 85
# seqlen = 999
# num_classes = 2
# seqlen = 1000



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
    # cmd = "export CUDA_VISIBLE_DEVICES=%d; %s" % (7, cmd)
    call(cmd, shell=True)
    queue.task_done()



# ####### hyperpara search #########

# # hidden1 = [ '100', '150', '200', '250', '300' ]
# hidden1 = [ '150', '200', '250', '300' ]
# hidden2 = [ 1000, 1500, 2000 ]
# indir = os.path.join( "/home/kimlab/project/DeepFam", "data", "cog_cv", "dataset0" )
# outdir = os.path.join( "/home/kimlab/project/DeepFam", "result", "hyper_search" )

# for h1 in hidden1:
#   for h2 in hidden2:
#     expname = os.path.join( outdir, "%s-%d" % (h1, h2) )
#     logdir = os.path.join( expname, "logs" )
#     mkdir( os.path.join( logdir, "train" ) )
#     mkdir( os.path.join( logdir, "test" ) )

#     ckptdir = os.path.join( expname, "save" )
#     ckptfile = os.path.join( ckptdir, "model.ckpt" )
#     mkdir( ckptdir )


#     # cmd
#     s="python %s" % os.path.join( os.path.dirname(os.path.realpath(__file__)), "run.py" )
#     s+= " --num_classes %s --seq_len %s" % ( num_classes, seqlen )
#     s+= " --window_lengths %s" % ( " ".join(['8', '12', '16', '20', '24', '28', '32', '36']) ) 
#     s+= " --num_windows %s" % ( " ".join([h1, h1, h1, h1, h1, h1, h1, h1]) ) 
#     s+= " --num_hidden %d --max_epoch %d" % ( h2, 25 )
#     s+= " --train_file %s" % os.path.join( indir, "train.txt" )
#     s+= " --test_file %s" % os.path.join( indir, "test.txt" )
#     s+= " --checkpoint_path %s --log_dir %s" % (ckptfile, logdir)
#     s+= " --save_interval %d" % (10000)
#     s+= " --batch_size %d" % (100)

#     enclosure_queue.put( s )











###### general task #########
tests = ["dataset1", "dataset2", "dataset0"]
# tests = ["dataset0"]

for exp in tests:
  indir = os.path.join( "/home/kimlab/project/DeepFam", "data", "l_1000_s_100", exp )
  outdir = os.path.join( "/home/kimlab/project/DeepFam", "result", "l_1000_s_100_try2" )
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
  s+= " --num_windows %s" % ( " ".join(['250', '250', '250', '250', '250', '250', '250', '250']) ) 
  s+= " --num_hidden %d --max_epoch %d" % ( 2000, 20 )
  s+= " --train_file %s" % os.path.join( indir, "train.txt" )
  s+= " --test_file %s" % os.path.join( indir, "test.txt" )
  s+= " --checkpoint_path %s --log_dir %s" % (ckptfile, logdir)
  s+= " --batch_size %d" % (100)
  s+= " --log_interval %d --save_interval %d" % (1000, 10000)
  # s+= " --prev_checkpoint_path=/home/kimlab/project/CCC/RES/otest-1665-1536-0.001000/save-158019"
  # s+= " --fine_tuning=True --fine_tuning_layers fc2"
  enclosure_queue.put( s )




# ###### re- task #########
# tests = ["dataset0", "dataset1", "dataset2"]
# paths = [ "/home/kimlab/project/DeepFam/result/l_1000_s_250/dataset0/save/model.ckpt-138869",
# "/home/kimlab/project/DeepFam/result/l_1000_s_250/dataset1/save/model.ckpt-138959",
# "/home/kimlab/project/DeepFam/result/l_1000_s_250/dataset2/save/model.ckpt-139064" ]

# for idx, exp in enumerate(tests):
#   indir = os.path.join( "/home/kimlab/project/DeepFam", "data", "l_1000_s_250", exp )
#   outdir = os.path.join( "/home/kimlab/project/DeepFam", "result", "l_1000_s_250_re" )
#   expname = os.path.join( outdir, exp )

#   logdir = os.path.join( expname, "logs" )
#   mkdir( os.path.join( logdir, "train" ) )
#   mkdir( os.path.join( logdir, "test" ) )

#   ckptdir = os.path.join( expname, "save" )
#   ckptfile = os.path.join( ckptdir, "model.ckpt" )
#   mkdir( ckptdir )


#   # cmd
#   s="python %s" % os.path.join( os.path.dirname(os.path.realpath(__file__)), "run.py" )
#   s+= " --num_classes %s --seq_len %s" % ( num_classes, seqlen )
#   s+= " --window_lengths %s" % ( " ".join(['8', '12', '16', '20', '24', '28', '32', '36']) ) 
#   s+= " --num_windows %s" % ( " ".join(['250', '250', '250', '250', '250', '250', '250', '250']) ) 
#   s+= " --num_hidden %d --max_epoch %d" % ( 2000, 7 )
#   s+= " --train_file %s" % os.path.join( indir, "train.txt" )
#   s+= " --test_file %s" % os.path.join( indir, "test.txt" )
#   s+= " --checkpoint_path %s --log_dir %s" % (ckptfile, logdir)
#   s+= " --batch_size %d" % (100)
#   s+= " --log_interval %d --save_interval %d" % (100, 10000)
#   s+= " --prev_checkpoint_path %s" % (paths[idx])
#   # s+= " --fine_tuning=True --fine_tuning_layers fc2"
#   enclosure_queue.put( s )









# ####### fine-tuning task #########
# # tests = ["class_level"]
# tests = ["family_level"]
# indirs = [ os.path.join( "/home/kimlab/project/DeepFam/tmp/gpcr/data", p ) for p in tests ]
# outdirs = [ os.path.join( "/home/kimlab/project/DeepFam/tmp/gpcr/result", p ) for p in tests ]

# for idx, expname in enumerate(outdirs):
#   indir = indirs[idx]

#   logdir = os.path.join( expname, "logs" )
#   mkdir( os.path.join( logdir, "train" ) )
#   mkdir( os.path.join( logdir, "test" ) )

#   ckptdir = os.path.join( expname, "save" )
#   ckptfile = os.path.join( ckptdir, "model.ckpt" )
#   mkdir( ckptdir )


#   # for cv in [1, 2]:
#     # cmd
#   s="python %s" % os.path.join( os.path.dirname(os.path.realpath(__file__)), "run.py" )
#   s+= " --num_classes %s --seq_len %s" % ( num_classes, seqlen )
#   s+= " --window_lengths %s" % ( " ".join(['8', '12', '16', '20', '24', '28', '32', '36']) ) 
#   s+= " --num_windows %s" % ( " ".join(['250', '250', '250', '250', '250', '250', '250', '250']) ) 
#   s+= " --num_hidden %s --max_epoch %d" % ( 2000, 10 )
#   s+= " --train_file %s" % os.path.join( indir, "train.txt" )
#   s+= " --test_file %s" % os.path.join( indir, "test.txt" )
#   s+= " --checkpoint_path %s --log_dir %s" % (ckptfile, logdir)
#   s+= " --batch_size %d" % (100)
#   s+= " --prev_checkpoint_path /home/kimlab/project/DeepFam/result/cog_cv/90percent/save/model.ckpt-25402"
#   s+= " --fine_tuning True --fine_tuning_layers %s" % ( " ".join(['fc2']) )
#   s+= " --log_interval %d" % (10)
#   s+= " --save_interval %d" % (10000)
#   s+= " --learning_rate %f" % (0.001)

#   enclosure_queue.put( s )



# ####### csss task #########
# # tests = ['1.27.1.1', '1.27.1.2', '1.36.1.2', '1.36.1.5', '1.4.1.1', '1.41.1.2', '1.41.1.5', '1.4.1.2', '1.4.1.3', '1.45.1.2', '2.1.1.1', '2.1.1.2', '2.1.1.3', '2.1.1.4', '2.1.1.5', '2.28.1.1', '2.28.1.3', '2.38.4.1', '2.38.4.3', '2.38.4.5', '2.44.1.2', '2.5.1.1', '2.5.1.3', '2.52.1.2', '2.56.1.2', '2.9.1.2', '2.9.1.3', '2.9.1.4', '3.1.8.1', '3.1.8.3', '3.2.1.2', '3.2.1.3', '3.2.1.4', '3.2.1.5', '3.2.1.6', '3.2.1.7', '3.3.1.2', '3.3.1.5', '3.32.1.1', '3.32.1.11', '3.32.1.13', '3.32.1.8', '3.42.1.1', '3.42.1.5', '3.42.1.8', '7.3.10.1', '7.3.5.2', '7.3.6.1', '7.3.6.2', '7.3.6.4', '7.39.1.2', '7.39.1.3', '7.41.5.1', '7.41.5.2']
# tests = ['2.44.1.2','2.28.1.1','3.3.1.5','3.42.1.5','2.28.1.3','2.38.4.5','2.38.4.3','3.2.1.3','3.42.1.8','2.1.1.5','2.52.1.2','2.38.4.1','3.32.1.8','3.32.1.1','3.2.1.4','3.3.1.2','3.2.1.5','2.5.1.3','1.41.1.5','2.5.1.1','3.2.1.2','7.39.1.3','3.2.1.6','1.36.1.5','2.1.1.3','7.39.1.2','2.9.1.2','3.42.1.1','2.1.1.1','3.32.1.13','2.1.1.4','1.36.1.2','3.32.1.11','7.41.5.1','1.27.1.2','1.4.1.2','3.2.1.7','1.27.1.1','7.3.10.1','3.1.8.3','1.45.1.2','2.9.1.4','7.41.5.2','2.56.1.2','2.1.1.2','7.3.6.2','1.4.1.3','3.1.8.1','1.41.1.2','1.4.1.1','7.3.5.2','2.9.1.3','7.3.6.1','7.3.6.4']
# # tests = ['2.44.1.2','2.28.1.1','3.3.1.5']
# indirs = [ os.path.join( "/home/kimlab/project/DeepFam/tmp/csss/data2", p ) for p in tests ]
# outdirs = [ os.path.join( "/home/kimlab/project/DeepFam/tmp/csss/result", p ) for p in tests ]

# for idx, expname in enumerate(outdirs):
#   indir = indirs[idx]

#   logdir = os.path.join( expname, "logs" )
#   mkdir( os.path.join( logdir, "train" ) )
#   mkdir( os.path.join( logdir, "test" ) )

#   ckptdir = os.path.join( expname, "save" )
#   ckptfile = os.path.join( ckptdir, "model.ckpt" )
#   mkdir( ckptdir )


#   s="python %s" % os.path.join( os.path.dirname(os.path.realpath(__file__)), "run.py" )
#   s+= " --num_classes %s --seq_len %s" % ( num_classes, seqlen )
#   s+= " --window_lengths %s" % ( " ".join(['8', '12', '16', '20', '24', '28', '32', '36']) ) 
#   s+= " --num_windows %s" % ( " ".join(['250', '250', '250', '250', '250', '250', '250', '250']) ) 
#   s+= " --num_hidden %s --max_epoch %d" % ( 2000, 10 )
#   s+= " --train_file %s" % os.path.join( indir, "train.txt" )
#   s+= " --test_file %s" % os.path.join( indir, "test.txt" )
#   s+= " --checkpoint_path %s --log_dir %s" % (ckptfile, logdir)
#   s+= " --batch_size %d" % (100)
#   # s+= " --prev_checkpoint_path /home/kimlab/project/DeepFam/result/cog_cv/90percent/save/model.ckpt-254024"
#   # s+= " --fine_tuning True --fine_tuning_layers %s" % ( " ".join(['fc1', 'fc2']) )
#   s+= " --log_interval %d" % (10)
#   s+= " --save_interval %d" % (10000)
#   s+= " --learning_rate %f" % (0.001)

#   enclosure_queue.put( s )








## main
# for i in range( num_threads ):
for i in range( 1, 1+num_threads ):
  worker = Thread( target=run, args=(i, enclosure_queue,) )
  worker.setDaemon(True)
  worker.start()


## run
enclosure_queue.join()