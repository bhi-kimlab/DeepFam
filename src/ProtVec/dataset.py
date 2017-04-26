"""
" Dataset module
" Handling protein sequence data.
"   Batch control, encoding, etc.
"""
import sys
import os

import tensorflow as tf
import numpy as np

from preprocess import getProtVec


class WordDict(object):
  def __init__(self, embedding_file):
    self.k = 3
    self.protVec = getProtVec(embedding_file) # 3mer -> vector
    self.size = 100







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
      elif len(batch) == 0:
        continue
      else:
        yield batch


  def iter_once(self, batch_size):
    while True:
      start = self._index_in_epoch
      self._index_in_epoch += batch_size

      if self._index_in_epoch > self._num_data:
        end = self._num_data
        idxs = self._perm[start:end]
        if len(idxs) > 0:
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
        word = seq[j:j+self.worddict.k] # 3mer
        v = self.worddict.protVec[ word ]
        data[i] += v # sum of all 3mer
      # normalize by length
      # data[i] /= (1.0 / seqlen - self.worddict.k + 1)


    return ( data, labels )