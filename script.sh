for t in 0.01 0.02 0.04 0.06 0.08 0.1
do
    for i in 1 2 3
    do
        python run.py --model $1 --mode crit -tn $t \
            -eid run$i --gpu $2 -clf rbf\_htn\_$t\_crit.log
    done
done
