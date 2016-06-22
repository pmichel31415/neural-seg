
TYPE=$1
DIR=$2
GAP=$3
EVAL_PY=$4

GOLD_DIR=${TYPE}_gold
# rm -r syldet
# mkdir syldet
# for f in wav/*.wav
# do
#     base_f=`basename $f`
#     python ../../thetaOscillator/thetaOscillator_py/SylSegDemo.py $f -o syldet/${base_f%.wav}.syldet >> log.txt
# done

#rm -r $GOLD_DIR
#mkdir $GOLD_DIR

#for f in $TYPE/*.$TYPE 
#do
#    base_f=`basename $f`
#    cat $f | awk '{printf("%.2f\n",$1);}' > $GOLD_DIR/${base_f%.$TYPE}.syldet
#done

#for f in $TYPE/*.$TYPE 
#do
#    base_f=`basename $f`
#    tail -n 1 $f | awk '{printf("%.2f\n",$2);}' >> $GOLD_DIR/${base_f%.$TYPE}.syldet
#done

# for f in syldet/*.syldet 
# do 
#     cat $f | awk '{print $1;}' > ${f}_tmp;mv ${f}_tmp $f
# done

ID=`uuidgen`
rm stats_$ID
for f in $DIR/*.syldet
do
    base_f=`basename $f`
    python $EVAL_PY -g $GOLD_DIR/$base_f -b $f -t $GAP -o matches/${base_f%.syldet}_res.csv 2>>log.txt >> "stats_$ID"
done

cat stats_$ID | awk '{ M += $1;B+=$2;G+=$3} END { print M/B;print M/G;print 2*M/(B+G) }'

rm stats_$ID log.txt
