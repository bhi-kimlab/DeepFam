"""
" Dataset module
" Handling protein sequence data.
"   Batch control, encoding, etc.
"""
import sys
import os

import tensorflow as tf
import numpy as np




class WordDict(object):
  def __init__(self, files, k, logpath):
    self._kmers = set()
    self.k = k
    self.w2i = dict()
    self.size = 0

    # check if file exist
    w2i_path = os.path.join(logpath, "w2i.txt")

    if os.path.exists(w2i_path):
      # load
      print("Load W2I")
      with open(w2i_path, 'r') as fr:
        for line in fr.readlines():
          self.size += 1
          word, i = line.strip().split("\t")
          i = int(i)
          self.w2i[word] = i
    else:
      print("Create new W2I")
      # parse files
      for f in files:
        self.parse_file(f)

      # set to sorted list -> dict
      for i, word in enumerate(sorted(self._kmers)):
        self.w2i[ word ] = i
        self.size += 1

      del self._kmers
      # save w2i
      with open(w2i_path, 'w') as fw:
        for word, i in self.w2i.iteritems():
          fw.write("%s\t%d\n" % (word, i))
 

  def parse_file(self, fpath):
    with open( fpath, 'r') as tr:
      for row in tr.readlines():
        (label, seq) = row.strip().split("\t")
        seq = seq.strip("_")
        seqlen = len(seq)

        for i in range( seqlen - self.k + 1 ):
          self._kmers.add( seq[i:i+self.k] )









## ######################## ##
#
#  DATASET Class
#
## ######################## ## 
# works for large dataset
class DataSet(object):
  def __init__(self, fpath, n_classes, wd, need_shuffle = True):
    self.NCLASSES = n_classes
    self.worddict = wd

    # read raw file
    self._raw = self.read_raw( fpath, wd.k )

    # iteration flags
    self._num_data = len(self._raw)
    self._epochs_completed = 0
    self._index_in_epoch = 0

    self._perm = np.arange(self._num_data)
    if need_shuffle:
      # shuffle data
      print("Needs shuffle")
      np.random.shuffle(self._perm)
    print("Reading data done")


  def next_batch(self, batch_size):
    start = self._index_in_epoch
    self._index_in_epoch += batch_size

    if self._index_in_epoch > self._num_data:
      print("%d epoch finish!" % self._epochs_completed)
      # finished epoch
      self._epochs_completed += 1
      # shuffle the data
      np.random.shuffle(self._perm)

      # start next epoch
      start = 0
      self._index_in_epoch = batch_size
      assert batch_size <= self._num_data
    
    end = self._index_in_epoch
    idxs = self._perm[start:end]
    return self.parse_data( idxs )


  def iter_batch(self, batch_size, max_iter):
    while True:
      batch = self.next_batch( batch_size )
      
      if self._epochs_completed >= max_iter:
        break
      else:
        yield batch


  def iter_once(self, batch_size):
    while True:
      start = self._index_in_epoch
      self._index_in_epoch += batch_size

      if self._index_in_epoch > self._num_data:
        end = self._num_data
        idxs = self._perm[start:end]
        yield self.parse_data( idxs )
        break
      
      end = self._index_in_epoch
      idxs = self._perm[start:end]
      yield self.parse_data( idxs )


  def full_batch(self):
    return self.parse_data( self._perm )


  def read_raw(self, fpath, k):
    # read raw files
    print("Read %s start" % fpath)
    res = []

    with open( fpath, 'r') as tr:
      for row in tr.readlines():
        (label, seq) = row.strip().split("\t")
        seq = seq.strip("_")

        res.append( (label, seq) )
    return res


  def parse_data(self, idxs):
    isize = len(idxs)

    data = np.zeros( (isize, self.worddict.size), dtype=np.float32 )
    labels = np.zeros( (isize, self.NCLASSES), dtype=np.uint8 )    

    for i, idx in enumerate(idxs):
      label, seq = self._raw[ idx ]

      ### encoding label
      labels[i][ int(label) ] = 1
      ### encoding seq
      seqlen = len(seq)
      # count frequency
      for j in range( seqlen - self.worddict.k + 1 ):
        word = seq[j:j+self.worddict.k]
        widx = self.worddict.w2i[ word ]
        data[i][ widx ] += (1.0 / seqlen - self.worddict.k + 1)


    return ( data, labels )