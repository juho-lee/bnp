for r in 0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8
do
    python run.py \
        --mode eval \
        --model $1 \
        --expid $2 \
        --gpu $3 \
        --r_bs $r \
        -el rbf/$r\_eval.log

    python run.py \
        --mode eval \
        --model $1 \
        --expid $2 \
        --gpu $3 \
        --r_bs $r \
        -ed periodic \
        -el periodic/$r\_eval.log

    python run.py \
        --mode eval \
        --model $1 \
        --expid $2 \
        --gpu $3 \
        --r_bs $r \
        -htn \
        -el htn/$r\_eval.log
done
