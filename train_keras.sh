TRAIN=$1
TEST=$2
OUT=$3
DI=$4
DH=$5
OPTIM=$6
LOSS=$7
C=$8

if [[ -z $DI ]]
then
    DI=39
fi

if [[ -z $DH ]]
then
    DH=50
fi

if [[ -z $OPTIM ]]
then
    OPTIM="sgd"
fi

if [[ -z $LOSS ]]
then
    LOSS="mse"
fi

if [[ -z $C ]]
then
    C=7
fi

mkdir $OUT
python kmodel.py -train $TRAIN -test $TEST -de $DI -dh $DH -optim $OPTIM -loss $LOSS -c $C -o $OUT -v

