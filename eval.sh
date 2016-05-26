
TYPE=$1
DIR=$2
GAP=$3

GOLD_DIR=${TYPE}_gold
# rm -r syldet
# mkdir syldet
# for f in wav/*.wav
# do
#     base_f=`basename $f`
#     python ../../thetaOscillator/thetaOscillator_py/SylSegDemo.py $f -o syldet/${base_f%.wav}.syldet >> log.txt
# done

rm -r $GOLD_DIR
mkdir $GOLD_DIR

for f in $TYPE/*.$TYPE 
do
    base_f=`basename $f`
    cat $f | awk '{print $1;}' > $GOLD_DIR/${base_f%.$TYPE}.syldet
done

for f in $TYPE/*.$TYPE 
do
    base_f=`basename $f`
    tail -n 1 $f | awk '{print $2;}' >> $GOLD_DIR/${base_f%.$TYPE}.syldet
done

# for f in syldet/*.syldet 
# do 
#     cat $f | awk '{print $1;}' > ${f}_tmp;mv ${f}_tmp $f
# done

rm stats
for f in $GOLD_DIR/*.syldet
do
    base_f=`basename $f`
    python ../../eval/seg_eval/run_eval.py -g $f -b $DIR/$base_f -t $GAP -o matches/${base_f%.syldet}_res.csv 2>>log.txt >> "stats"
done

cat stats | awk '{ P += $1;R+=$2;F+=$3;T+=1} END { print P/T;print R/T;print F/T }'

rm stats log.txt
