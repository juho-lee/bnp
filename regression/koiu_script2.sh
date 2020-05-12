python run_celeba.py --resume \
    --expid run2 \
    --model rbnp \
    --base np

for i in {3..5}
do
    python run_celeba.py \
        --expid run\$i \
        --model rbnp \
        --base np
done
