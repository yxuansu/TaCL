CUDA_VISIBLE_DEVICES=0,1 python train.py\
    --language english\
    --model_name bert-base-uncased\
    --train_data ../pretraining_data/example_data/english_data_500_lines.txt\
    --number_of_gpu 2\
    --max_len 256\
    --batch_size_per_gpu 16\
    --gradient_accumulation_steps 1\
    --effective_batch_size 32\
    --learning_rate 1e-4\
    --total_steps 18\
    --print_every 2\
    --save_every 5\
    --ckpt_save_path ./debug_ckpt/tacl_english/
