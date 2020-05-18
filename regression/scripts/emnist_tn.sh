for tn in 0.02 0.03 0.04 0.05 0.06 0.07 0.08 0.09
do
    for i in {1..5}
    do
        python emnist.py \
            --model $1 \
            --mode eval \
            --expid run$i \
            --t_noise $tn \
            --gpu $2
    done
done
