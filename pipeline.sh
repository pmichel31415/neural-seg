#!/bin/bash


CFG_FILE=$1

source $CFG_FILE
TRAIN=`pwd`/splits/train_${SPLIT}.lst
TEST=`pwd`/splits/test_${SPLIT}.lst
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

function summarize {
    printf "Automatic segmentation using rnn with : \n\t- model %s\n\t- input size %d\n\t- hidden layer size %d\n\t- optimization method : %s\n\t- loss function : %s\n\t- BPTT nb of frames : %d\n\t- batch size : %d\n\t- post processing method : %s,%s\n" $MODEL $INPUT $HIDDEN $OPTIM $LOSS $SPAN $BATCH_SIZE $POST_PROC_METHOD $POST_PROC_OPTION
}


mkdir splits output 2> /dev/null

## Auto split
#cd splits
#split $CORPUS/mfcc_npy_$INPUT $NAME
#cd ..


## Run network (training + test)
`pwd`/train_keras.sh $TRAIN $TEST $OUT $INPUT $HIDDEN $OPTIM $LOSS $SPAN $MODEL $BATCH_SIZE

## Post processing
`pwd`/post_process.sh $OUT 3 xxx 0.02 100 $POST_PROC_METHOD $POST_PROC_OPTION

## Copy to corpus location for evaluation
cp -r ${OUT}_syldet $CORPUS

cd $CORPUS

#rm $RESULT_FILE 2> /dev/null

summarize >> $RESULT_FILE

## Evaluate over any kind of segmentation
printf "Evaluation on phone boundaries (if applicable)\n" >> $RESULT_FILE
$EVAL_SH phn ${OUT}_syldet 0.020001 $EVAL_PY >> $RESULT_FILE 2>/dev/null 
printf "Evaluation on syllable boundaries (if applicable)\n" >> $RESULT_FILE
$EVAL_SH syl ${OUT}_syldet 0.020001 $EVAL_PY >> $RESULT_FILE 2>/dev/null  
printf "Evaluation on word boundaries (if applicable)\n" >> $RESULT_FILE
$EVAL_SH wrd ${OUT}_syldet 0.020001 $EVAL_PY >> $RESULT_FILE 2>/dev/null 
