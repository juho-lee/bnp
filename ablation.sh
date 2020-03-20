# python run.py --gpu 0 --num_steps 500000 --model cnp --train_batch_size 512 &
# python run.py --gpu 1 --num_steps 500000 --model cnp --train_batch_size 1024 &
# python run.py --gpu 2 --num_steps 500000 --model cnp --train_batch_size 2048 &
# python run.py --gpu 3 --num_steps 500000 --model cnp --train_batch_size 256 &
# python run.py --gpu 4 --num_steps 500000 --model bnp --train_batch_size 512 &
# python run.py --gpu 5 --num_steps 500000 --model bnp --train_batch_size 1024 &
# python run.py --gpu 6 --num_steps 500000 --model bnp --train_batch_size 2048 &
# python run.py --gpu 7 --num_steps 500000 --model bnp --train_batch_size 256 &
# wait

# python run.py --gpu 0 --num_steps 100000 --model np --train_batch_size 100 &
# python run.py --gpu 1 --num_steps 100000 --model cnp --train_batch_size 100 &
# python run.py --gpu 2 --num_steps 100000 --model anp --train_batch_size 100 &
# python run.py --gpu 3 --num_steps 100000 --model bnp --train_batch_size 100 &
# python run.py --gpu 4 --num_steps 100000 --model banp --train_batch_size 100 &
# python run.py --gpu 5 --num_steps 100000 --model canp --train_batch_size 100 &
# wait

# python run.py --gpu 0 --num_steps 500000 --model anp --train_batch_size 512 &
# python run.py --gpu 1 --num_steps 500000 --model anp --train_batch_size 1024 &
# python run.py --gpu 2 --num_steps 500000 --model anp --train_batch_size 2048 &
# python run.py --gpu 3 --num_steps 500000 --model anp --train_batch_size 256 &
# python run.py --gpu 4 --num_steps 500000 --model banp --train_batch_size 512 &
# python run.py --gpu 5 --num_steps 500000 --model banp --train_batch_size 1024 &
# python run.py --gpu 6 --num_steps 500000 --model banp --train_batch_size 2048 &
# python run.py --gpu 7 --num_steps 500000 --model banp --train_batch_size 256 &
# wait

# python run.py --gpu 2 --num_steps 100000 --model cnp --train_batch_size 100 & python run.py --gpu 3 --num_steps 100000 --model cnp --train_batch_size 512 &
# python run.py --gpu 6 --num_steps 100000 --model bnp --train_batch_size 100 &
# python run.py --gpu 7 --num_steps 100000 --model bnp --train_batch_size 512 &
# wait

python run.py --gpu 0 --num_steps 100000 --model bnp_single --dim 32 &
python run.py --gpu 1 --num_steps 100000 --model bnp_single --dim 64 &
python run.py --gpu 2 --num_steps 100000 --model bnp_single --dim 128 &
python run.py --gpu 3 --num_steps 100000 --model bnp_single --dim 256 &
python run.py --gpu 4 --num_steps 100000 --model bnp_single --dim 512 &
python run.py --gpu 5 --num_steps 100000 --model bnp_single --dim 1024 &
python run.py --gpu 6 --num_steps 100000 --model bnp_single --dim 2048 &
python run.py --gpu 7 --num_steps 100000 --model bnp_single --dim 16 &
wait