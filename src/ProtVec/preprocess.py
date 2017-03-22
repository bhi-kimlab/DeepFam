from collections import defaultdict
import numpy as np


def getProtVec(embedding_file):
  zeroVec = np.zeros(100, dtype=np.float16)
  protVec = defaultdict(lambda: zeroVec)

  with open(embedding_file, 'r') as fr:
    for row in fr.readlines():
      row = row.replace("\"", "").rstrip()
      elems = row.split("\t")
      assert len(elems) == 101

      protVec[ elems[0] ] = np.asarray( [ float(x) for x in elems[1:] ], dtype=np.float32 )

  return protVec





# ### #
# parse raw seq into protVec
# ### #
def parse(seq, protVec=None):
  # read 3gram
  if protVec == None:
    protVec = getProtVec()

  # parse seqs
  seq = seq.rstrip("_")
  vec = np.zeros( 100, dtype=np.float16 )

  for sidx in range(1):
    tmpseq = seq[sidx:]
    l = len(tmpseq) - 3 + 1
    
    for i in range(l):
      vec = vec + protVec[ tmpseq[i:i+3] ]

  return vec


def save_to_file( seqfile, outfile ):
  protVec = getProtVec()

  data = []
  labels = []
  familyset = set()
  with open(seqfile, 'r') as fr:
    for row in fr.readlines():
      (family, seq) = row.rstrip().split("\t")
      embedded = parse( seq, protVec )
      labels.append( family )
      data.append( embedded )
      familyset.add( family )

  with open(outfile, 'w') as fw:
    ### write header
    # # of data, # of features, classes
    header = [ str(len(data)), str(len(data[0])) ] + [ str(x) for x in range(len(familyset)) ]
    fw.write("%s\n" % ",".join( header ))

    ### write row
    for i, d in enumerate(data):
      l = labels[i]
      row = [ "%.3f" % x for x in list(d) ]
      row.append( l )
      fw.write("%s\n" % ",".join( row ))

  





### main
if __name__ == '__main__':
  train_file = "/home/kimlab/project/CCC/tmp/swissprot/data/example/train.txt"
  test_file = "/home/kimlab/project/CCC/tmp/swissprot/data/example/test.txt"
  save_to_file(train_file, "train.txt")
  save_to_file(test_file, "test.txt")