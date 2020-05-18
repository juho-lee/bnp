for i in {1..5}
do
    python emnist.py \
        --model $1 \
        --expid run$i \
        --gpu $2

    python emnist.py \
        --model $1 \
        --expid run$i \
        --mode eval \
        --class_range 10 47 \
        --gpu $2

    python emnist.py \
        --model $1 \
        --expid run$i \
        --mode eval \
        --t_noise 0.05 \
        --gpu $2
done
