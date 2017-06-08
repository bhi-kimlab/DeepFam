import glob
import os
import sys
import numpy as np
import csv
from collections import defaultdict
import re


def read_fasta(fasta_file):
  input = open(fasta_file, 'r')

  chrom_seq = ''
  chrom_id = None

  for line in input:
    if line[0] == '>':
      if chrom_id is not None:
        yield chrom_id, chrom_seq
      
      chrom_seq = ''
      chrom_id = line.split()[0][1:].replace("|", "_")
    else:
      chrom_seq += line.strip().upper()

  yield chrom_id, chrom_seq

  input.close()




def read_data(sample_file):
  maxlen = 0
  data = []
  for sid, seq in read_fasta(sample_file):
    data.append( (sid, seq) )
    maxlen = max(maxlen, len(seq))

  return data, maxlen




if __name__ == '__main__':
  target_path = sys.argv[1]
  sample_file = os.path.join(target_path, "prot.fasta")
  

  data, maxlen = read_data( sample_file )

  # parsing data
  trans_path = os.path.join(target_path, "trans.txt")
  data_path = os.path.join(target_path, "data.txt")
  
  maxlen = 1000
  with open(trans_path, 'w') as transw, open(data_path, 'w') as dataw:
    # first write maxlen
    transw.write("MAXLEN\t%d\n" % (maxlen))
    for idx, (pid, seq) in enumerate(data):
      if len(seq) <= maxlen:
        transw.write("%s\t%d\n" % (pid, idx))

        padded_seq = seq + "_"*(maxlen - len(seq))
        dataw.write("0\t%s\n" % (padded_seq) )
    
  