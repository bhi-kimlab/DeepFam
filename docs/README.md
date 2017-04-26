# About DeepFam

DeepFam is a deep learning based alignment-free protein function prediction method. DeepFam first extracts motif-like features from a raw sequence in convolution layer and make a prediction based on the features.

![Figure](/docs/images/Figure1.png)


# Features

* Do not need multiple or pairwise alignment of training sequences.
* Instead capture locally similar fragment (motif) within a family by convolutional neural network.
* Combine existence of motifs in dense layer for more accurate modelling.


# Installation

DeepFam is implemented in with [Tensorflow](https://www.tensorflow.org/) library. Both CPU and GPU machines are supported. For detail instruction of installing Tensorflow, see the [guide on official website](https://www.tensorflow.org/install).

## Requirements

* Python: 2.7
* Tensorflow: over 1.0

# Usage

First, clone the repository or download compressed source code files.
```
$ git clone https://github.com/bhi-kimlab/DeepFam.git
$ cd DeepFam
```
You can see the valid paramenters for DeepFam by help option:
```
$ python src/DeepFam/run.py --help
```
One example of parameter setting is like:
```
$ python src/DeepFam/run.py \
  --num_windows [256, 256, 256, 256, 256, 256, 256, 256] \ 
  --window_lengths [8, 12, 16, 20, 24, 28, 32, 36] \
  --num_hidden 2000 \
  --batch_size 100 \
  --keep_prob 0.7 \ 
  --learning_rate 0.001 \
  --regularizer 0.001 \ 
  --max_epoch 25 \
  --seq_len 1000 \ 
  --num_classes 1074 \ 
  --log_interval 100 \ 
  --save_interval 100 \ 
  --log_dir '/tmp/logs' \  
  --test_file '/data/test.txt' \ 
  --train_file '/data/train.txt' 
```



# Contact
If you have any question or problem, please send a email to [dane2522@gmail.com](mailto:dane2522@gmail.com)
