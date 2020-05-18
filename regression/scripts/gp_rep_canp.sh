python gp.py \
    --model canp \
    --expid run5 \
    --gpu $1

python gp.py \
    --model canp \
    --expid run5 \
    --mode eval \
    --t_noise 0.1 \
    --gpu $1

python gp.py \
    --model canp \
    --expid run5 \
    --mode eval \
    --pp 0.5 \
    --gpu $1
