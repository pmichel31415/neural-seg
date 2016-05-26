#!/bin/bash

SEGDIR=$1
MFCC_DIR=$2
OUTDIR=$3
TIMEDIR=$4

for f in $SEGDIR/*.syldet
do
    bf=`basename $f`
    name=${bf%.syldet}
    outf=$OUTDIR/${name}.npy
    mfcc=$MFCC_DIR/${name}.npy
    time=$TIMEDIR/${name}.npy
    python seg2emb.py -s $f -i $mfcc -d 3 -r 100 -o $outf -t $time
done

