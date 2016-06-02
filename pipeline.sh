#!/bin/bash

CORPUS=$1
NAME=$2
TRAIN=`pwd`/splits/train_${NAME}.lst
TEST=`pwd`/splits/test_${NAME}.lst
OUT=`pwd`/output/$NAME
RESULT_FILE=results_${NAME}.txt

SPLIT_SH=`pwd`/split_tt.sh
EVAL_SH=`pwd`/eval.sh

function split {
    FILES=$1
    TOT=`ls ${FILES}/*.npy | wc -l`
    SPLIT=$((TOT / 10))
    bash $SPLIT_SH $1 $SPLIT npy $2
}


mkdir splits output 2> /dev/null

## Auto split
#cd splits
#split $CORPUS/mfcc_npy $NAME
#cd ..


## Run network (training + test)
`pwd`/train_keras.sh $TRAIN $TEST $OUT 39 20 rmsprop mse 7

## Post processing
`pwd`/post_process.sh $OUT 3 xxx 0.03 100

## Copy to corpus location for evaluation
cp -r ${OUT}_syldet $CORPUS

cd $CORPUS

rm $RESULT_FILE 2> /dev/null

## Evaluate over any kind of segmentation
$EVAL_SH phn ${OUT}_syldet 0.025 >> $RESULT_FILE 2>/dev/null 
$EVAL_SH syl ${OUT}_syldet 0.025 >> $RESULT_FILE 2>/dev/null  
$EVAL_SH wrd ${OUT}_syldet 0.025 >> $RESULT_FILE 2>/dev/null 
