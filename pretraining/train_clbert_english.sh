CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python train.py\
    --language english\
    --model_name bert-base-uncased\
    --train_data ../pretraining_data/english/english_wiki.txt\
    --number_of_gpu 8\
    --max_len 256\
    --batch_size_per_gpu 16\
    --gradient_accumulation_steps 2\
    --effective_batch_size 256\
    --learning_rate 1e-4\
    --total_steps 150010\
    --print_every 500\
    --save_every 10000\
    --ckpt_save_path ./ckpt/clbert_english/
