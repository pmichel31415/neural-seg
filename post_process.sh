DIR=$1
KER=$2
VADDIR=$3
CLIP=$4
RATE=$5
POST_PROC_METHOD=$6
NUM=$7

SYLDIR=${DIR}_syldet

rm -r $SYLDIR
mkdir $SYLDIR

for f in $DIR/*_loss.npy; do 
    base_f=`basename $f`
    python post_process_rnn_error.py -i $f -o $SYLDIR/${base_f%_loss.npy}.syldet -k $KER -c $CLIP -r $RATE -m $POST_PROC_METHOD -v -n $NUM
done
