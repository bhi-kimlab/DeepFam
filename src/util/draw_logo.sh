#!/bin/bash

# $1: ps name

basepath="/home/kimlab/project/DeepFam/data/prosite"
utilpath=$(dirname "$0")

# iFile="${basepath}/$1/data.txt"
iFile="${basepath}/$1/pure_data.txt"
oPath="${basepath}/$1/res"
lPath="${basepath}/$1/logo"

mkdir -p $lPath

seqlen=$(head -n 1 "${basepath}/$1/trans.txt" | cut -f 2)


export CUDA_VISIBLE_DEVICES=0; \
  python "${utilpath}/draw_logo.py" \
  --num_classes 1074 \
  --seq_len $seqlen \
  --window_lengths 8 12 16 20 24 28 32 36 \
  --num_windows 250 250 250 250 250 250 250 250 \
  --num_hidden 2000 \
  --max_epoch=25 \
  --train_file=$iFile \
  --test_file=$iFile \
  --batch_size=10 \
  --checkpoint_pat /home/kimlab/project/DeepFam/result/cog_cv/90percent/save/model.ckpt-203219 \
  --log_dir=$oPath  
  
  

# draw logo
cat "$oPath/report.txt" | \
  awk -v OFS="\t" '{if($1>=0.7) print $2, $3}' | \
  sort -rn | cut -f 2 | head -n 30 | \
  xargs -P 10 -I {} sh -c "cut -f 2 $oPath/p{}.txt | grep -v _ | weblogo -F pdf -A protein -s large -t '[DeepFam] {}' > $lPath/l{}.pdf"
#head -n 20 "$oPath/report.txt" | cut -f 2 | xargs -P 10 -I {} sh -c "echo {}.txt"

