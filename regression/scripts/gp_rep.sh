for i in {1..5}
do
    python gp.py \
        --model $1 \
        --expid run$i \
        --gpu $2

    python gp.py \
        --model $1 \
        --expid run$i \
        --mode eval \
        --t_noise 0.1 \
        --gpu $2

    python gp.py \
        --model $1 \
        --expid run$i \
        --mode eval \
        --pp 0.5 \
        --gpu $2
done
