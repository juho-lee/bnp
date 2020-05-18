python gp.py \
    --model cnp \
    --expid run2 \
    --resume \
    --gpu $1

python gp.py \
    --model cnp \
    --expid run2 \
    --mode eval \
    --t_noise 0.1 \
    --gpu $1

python gp.py \
    --model cnp \
    --expid run2 \
    --mode eval \
    --pp 0.5 \
    --gpu $1

for i in {3..5}
do
    python gp.py \
        --model cnp \
        --expid run$i \
        --gpu $1

    python gp.py \
        --model cnp \
        --expid run$i \
        --mode eval \
        --t_noise 0.1 \
        --gpu $1

    python gp.py \
        --model cnp \
        --expid run$i \
        --mode eval \
        --pp 0.5 \
        --gpu $1
done
