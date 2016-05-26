DIR=$1
KER=$2
VADDIR=$3
CLIP=$4
RATE=$5
TIMEDIR=$6
SYLDIR=${DIR}_syldet


rm -r $SYLDIR
mkdir $SYLDIR

for f in $DIR/*.loss; do 
    base_f=`basename $f`

    if [ -z $TIMEDIR ]
    then
        python post_process_rnn_error.py -i $f -o $SYLDIR/${base_f%.loss}.syldet -k $KER -c $CLIP -r $RATE -v # -vad $VADDIR/${base_f%.loss}.sil
    else
        time=$TIMEDIR/${base_f%.loss}.syldet
        python post_process_rnn_error.py -i $f -o $SYLDIR/${base_f%.loss}.syldet -k $KER -c $CLIP -r $RATE -t $time -v # -vad $VADDIR/${base_f%.loss}.sil
    fi
done
