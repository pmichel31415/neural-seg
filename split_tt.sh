DIR=$1
NUM=$2
EXT=$3
NAME=$4
CURDIR=`pwd`

cd $DIR

ls `pwd`/*.$EXT -rt | shuf  > $CURDIR/full_${NAME}.lst

cd $CURDIR

TOT=`cat full_${NAME}.lst | wc -l`
((NUM2=TOT - NUM))
cat full_${NAME}.lst | head -n $NUM > train_${NAME}.lst
cat full_${NAME}.lst | tail -n $NUM2 > test_${NAME}.lst
rm full_${NAME}.lst

