#!/bin/bash

CORPUS=$1
NAME=$2
TRAIN=`pwd`/splits/train_${NAME}.lst
TRAIN2=`pwd`/splits/train_${NAME}_2.lst
TEST=`pwd`/splits/test_${NAME}.lst
TEST2=`pwd`/splits/test_${NAME}_2.lst
OUT=`pwd`/output/$NAME
RESULT_FILE=results_${NAME}.txt

SPLIT_SH=`pwd`/split_tt.sh
EVAL_SH=`pwd`/eval.sh

function split {
    FILES=$1
    TOT=`ls ${FILES}/*.npy | wc -l`
    SPLIT=$((TOT / 2))
    bash $SPLIT_SH $1 $SPLIT npy $2
}

mkdir splits output 2> /dev/null


cd splits
split $CORPUS/mfcc_npy $NAME
cd ..

`pwd`/train_keras.sh $TRAIN $TEST $OUT 39 64 rmsprop mse 7
`pwd`/post_process.sh $OUT 3 xxx 0.03 100
cp -r ${OUT}_syldet $CORPUS

mkdir $CORPUS/phn_emb $CORPUS/phn_times 2>/dev/null
 
./phn2syl.sh ${OUT}_syldet $CORPUS/mfcc_npy $CORPUS/phn_emb $CORPUS/phn_times

cd splits
split $CORPUS/phn_emb ${NAME}_2
cd ..

`pwd`/train_keras.sh $TRAIN2 $TEST2 ${OUT}_2 117 128 rmsprop mse 7
`pwd`/post_process.sh ${OUT}_2 1 xxx 0.03 100 ${OUT}_syldet
cp -r ${OUT}_2_syldet $CORPUS

cd $CORPUS

rm $RESULT_FILE 2> /dev/null

$EVAL_SH phn ${OUT}_syldet 0.03 >> $RESULT_FILE 2>/dev/null 
$EVAL_SH syl ${OUT}_syldet 0.03 >> $RESULT_FILE 2>/dev/null  
$EVAL_SH wrd ${OUT}_syldet 0.03 >> $RESULT_FILE 2>/dev/null 
$EVAL_SH phn ${OUT}_2_syldet 0.03 >> $RESULT_FILE 2>/dev/null 
$EVAL_SH syl ${OUT}_2_syldet 0.03 >> $RESULT_FILE 2>/dev/null 
$EVAL_SH wrd ${OUT}_2_syldet 0.03 >> $RESULT_FILE 2>/dev/null 

