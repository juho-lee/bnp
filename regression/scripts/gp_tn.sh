for tn in 0.02 0.04 0.06 0.08 0.1 0.12 0.14 0.16
do
    for i in {1..5}
    do
        python gp.py \
            --model $1 \
            --mode eval \
            --expid run$i \
            --t_noise $tn \
            --gpu $2
    done
done
