CUDA_VISIBLE_DEVICES=0,1 python  ../../train.py\
    --model_name cambridgeltl/tacl-bert-base-chinese\
    --train_path ../../../benchmark_data/CWS_data/CITYU/cityu_train.txt\
    --dev_path ../../../benchmark_data/CWS_data/CITYU/cityu_test.txt\
    --test_path ../../../benchmark_data/CWS_data/CITYU/cityu_test.txt\
    --label_path ../../../benchmark_data/CWS_data/CITYU/cityu_label.txt\
    --learning_rate 2e-5\
    --batch_size 128\
    --batch_size_per_gpu 64\
    --number_of_gpu 2\
    --gradient_accumulation_steps 1\
    --total_epochs 30\
    --number_of_runs 5\
    --save_path_prefix ../../ckpt/cityu_cws/


