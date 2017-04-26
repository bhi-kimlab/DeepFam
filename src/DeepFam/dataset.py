"""
" Dataset module
" Handling protein sequence data.
"   Batch control, encoding, etc.
"""
import sys
import os

import tensorflow as tf
import numpy as np


## ######################## ##
#
#  Define CHARSET, CHARLEN
#
## ######################## ## 
CHARSET = { 'A': 0, 'C': 1, 'D': 2, 'E': 3, 'F': 4, 'G': 5, 'H': 6, \
            'I': 7, 'K': 8, 'L': 9, 'M': 10, 'N': 11, 'P': 12, 'Q': 13, \
            'R': 14, 'S': 15, 'T': 16, 'V': 17, 'W': 18, 'Y': 19, 'X': 20, \
            'O': 20, 'U': 20,
            'B': (2, 11),
            'Z': (3, 13),
            'J': (7, 9) }
CHARLEN = 21



## ######################## ##
#
#  Encoding Helpers
#
## ######################## ## 
def encoding_seq_np( seq, arr):
  for i, c in enumerate(seq):
    if c == "_":
      # let them zero
      continue
    elif isinstance(CHARSET[ c ], int):
      idx = CHARLEN * i + CHARSET[ c ]
      arr[ idx ] = 1
    else:
      idx1 = CHARLEN * i + CHARSET[ c ][0]
      idx2 = CHARLEN * i + CHARSET[ c ][1]
      arr[ idx1 ] = 0.5
      arr[ idx2 ] = 0.5
      # raise Exception("notreachhere")


def encoding_label_np( l, arr):
  arr[ int(l) ] = 1




## ######################## ##
#
#  DATASET Class
#
## ######################## ## 
# works for large dataset
class DataSet(object):
  def __init__(self, fpath, seqlen, n_classes, need_shuffle = True):
    self.SEQLEN = seqlen
    self.NCLASSES = n_classes
    self.charset = CHARSET
    self.charset_size = CHARLEN

    # read raw file
    self._raw = self.read_raw( fpath )

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
      # print("%d epoch finish!" % self._epochs_completed)
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


  def iter_once(self, batch_size, with_raw=False):
    while True:
      start = self._index_in_epoch
      self._index_in_epoch += batch_size

      if self._index_in_epoch > self._num_data:
        end = self._num_data
        idxs = self._perm[start:end]
        if len(idxs) > 0:
          yield self.parse_data( idxs, with_raw )
        break
      
      end = self._index_in_epoch
      idxs = self._perm[start:end]
      yield self.parse_data( idxs, with_raw )


  def full_batch(self):
    return self.parse_data( self._perm )


  def read_raw(self, fpath):
    print("Read %s start" % fpath)
    res = []

    with open( fpath, 'r') as tr:
      for row in tr.readlines():
        (label, seq) = row.strip().split("\t")
        seqlen = len(seq)

        if (seqlen != self.SEQLEN):
          raise Exception("SEQLEN is different from input data (%d / %d)"
            % (seqlen, self.SEQLEN))

        res.append( (label, seq) )
    return res


  def parse_data(self, idxs, with_raw=False):
    isize = len(idxs)

    data = np.zeros( (isize, CHARLEN * self.SEQLEN), dtype=np.float32 )
    labels = np.zeros( (isize, self.NCLASSES), dtype=np.uint8 )    
    raw = []

    for i, idx in enumerate(idxs):
      label, seq = self._raw[ idx ]

      encoding_label_np(label, labels[i] )
      encoding_seq_np(seq, data[i] )
      raw.append( (label, seq) )

    if with_raw:
      return ( data, labels, raw )
    else:
      return ( data, labels )

