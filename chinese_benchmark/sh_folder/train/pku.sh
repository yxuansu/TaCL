CUDA_VISIBLE_DEVICES=4,5 python  ../../train.py\
    --model_name cambridgeltl/tacl-bert-base-chinese\
    --train_path ../../../benchmark_data/CWS_data/PKU/pku_train_all_processed.txt\
    --dev_path ../../../benchmark_data/CWS_data/PKU/pku_dev_processed.txt\
    --test_path ../../../benchmark_data/CWS_data/PKU/pku_test_processed.txt\
    --label_path ../../../benchmark_data/CWS_data/PKU/PKU_CWS_label.txt\
    --learning_rate 2e-5\
    --batch_size 128\
    --batch_size_per_gpu 64\
    --number_of_gpu 2\
    --gradient_accumulation_steps 1\
    --total_epochs 100\
    --number_of_runs 5\
    --save_path_prefix ../../ckpt/pku_cws/


