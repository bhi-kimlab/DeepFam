import glob
import os
import sys
import numpy as np
import csv
from collections import defaultdict
import re


def read_data(sample_file):
  data = []

  with open(sample_file, 'r') as fr:
    for line in fr.readlines():
      fam, seq = line.strip().split("\t")
      data.append( (fam, seq.replace("_", "")) )

  return data




if __name__ == '__main__':
  f1 = sys.argv[1] # f1 is reference
  f2 = sys.argv[2]
  f3 = sys.argv[3] # f3 is output
  f4 = sys.argv[4] # f4 is filtered seqs

  d1 = read_data(f1)
  d2 = read_data(f2)

  intersect = []

  for _, s1 in d1:
    for _, s2 in d2:
      if s1 == s2:
        intersect.append(s1)
        

  print("Intersection : %d" % len(intersect))

  with open(f3, 'w') as fw, open(f4, 'w') as flfw:
    with open(f2, 'r') as fr:
      for line in fr.readlines():
        fam, seq = line.strip().split("\t")
        rawseq = seq.replace("_", "")

        if rawseq not in intersect:
          fw.write("%s\t%s\n" % (fam, seq))
        else:
          flfw.write("%s\t%s\n" % (fam, seq))
  