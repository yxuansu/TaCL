CUDA_VISIBLE_DEVICES=6,7 python  ../../train.py\
    --model_name cambridgeltl/tacl-bert-base-chinese\
    --train_path ../../../benchmark_data/NER/MSRANER/MSRA.train.char.txt\
    --dev_path ../../../benchmark_data/NER/MSRANER/MSRA.dev.char.txt\
    --test_path ../../../benchmark_data/NER/MSRANER/MSRA.test.char.txt\
    --label_path ../../../benchmark_data/NER/MSRANER/MSRA_NER_Label.txt\
    --learning_rate 2e-5\
    --batch_size 128\
    --batch_size_per_gpu 64\
    --number_of_gpu 2\
    --gradient_accumulation_steps 1\
    --total_epochs 40\
    --number_of_runs 5\
    --save_path_prefix ../../ckpt/msra/


