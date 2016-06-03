#!/bin/bash


CFG_FILE=$1

source $CFG_FILE
TRAIN=`pwd`/splits/train_${NAME}.lst
TEST=`pwd`/splits/test_${NAME}.lst
OUT=`pwd`/output/$NAME
RESULT_FILE=results_${NAME}.txt

SPLIT_SH=`pwd`/split_tt.sh
EVAL_SH=`pwd`/eval.sh
EVAL_PY=`pwd`/seg_eval/run_eval.py
function split {
    FILES=$1
    TOT=`ls ${FILES}/*.npy | wc -l`
    SPLIT=$((TOT / 10))
    bash $SPLIT_SH $1 $SPLIT npy $2
}


mkdir splits output 2> /dev/null

## Auto split
cd splits
split $CORPUS/mfcc_npy_$INPUT $NAME
cd ..


## Run network (training + test)
`pwd`/train_keras.sh $TRAIN $TEST $OUT $INPUT $HIDDEN $OPTIM $LOSS $SPAN $MODEL

## Post processing
`pwd`/post_process.sh $OUT 3 xxx 0.03 100 $POST_PROC_METHOD

## Copy to corpus location for evaluation
cp -r ${OUT}_syldet $CORPUS

cd $CORPUS

rm $RESULT_FILE 2> /dev/null

printf $TEXT > $RESULT_FILE

## Evaluate over any kind of segmentation
printf "Evaluation on phone boundaries (if applicable)\n" >> $RESULT_FILE
$EVAL_SH phn ${OUT}_syldet 0.025 $EVAL_PY >> $RESULT_FILE 2>/dev/null 
printf "Evaluation on syllable boundaries (if applicable)\n" >> $RESULT_FILE
$EVAL_SH syl ${OUT}_syldet 0.025 $EVAL_PY >> $RESULT_FILE 2>/dev/null  
printf "Evaluation on word boundaries (if applicable)\n" >> $RESULT_FILE
$EVAL_SH wrd ${OUT}_syldet 0.025 $EVAL_PY >> $RESULT_FILE 2>/dev/null 
