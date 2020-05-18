python celeba.py \
    --model banp \
    --expid run1 \
    --gpu $1 \
    --resume

for i in {2..5}
do
    python celeba.py \
        --model banp \
        --expid run$i \
        --gpu $1

    python celeba.py \
        --model banp \
        --expid run$i \
        --mode eval \
        --t_noise 0.05 \
        --gpu $1
done
