python emnist.py \
    --model cnp \
    --expid run1 \
    --resume \
    --gpu $1

python emnist.py \
    --model cnp \
    --expid run1 \
    --mode eval \
    --class_range 10 47 \
    --gpu $1

python emnist.py \
    --model cnp \
    --expid run1 \
    --mode eval \
    --t_noise 0.05 \
    --gpu $1

for i in {2..5}
do
    python emnist.py \
        --model cnp \
        --expid run$i \
        --gpu $1

    python emnist.py \
        --model cnp \
        --expid run$i \
        --mode eval \
        --class_range 10 47 \
        --gpu $1

    python emnist.py \
        --model cnp \
        --expid run$i \
        --mode eval \
        --t_noise 0.05 \
        --gpu $1
done
