CUDA_VISIBLE_DEVICES=0 python  ../../inference.py\
    --model_name cambridgeltl/tacl-bert-base-chinese\
    --saved_ckpt_path ../../pretrained_ckpt/pku_cws/pku_ckpt\
    --train_path ../../../benchmark_data/CWS_data/PKU/pku_train_all_processed.txt\
    --dev_path ../../../benchmark_data/CWS_data/PKU/pku_dev_processed.txt\
    --test_path ../../../benchmark_data/CWS_data/PKU/pku_test_processed.txt\
    --label_path ../../../benchmark_data/CWS_data/PKU/PKU_CWS_label.txt\
    --batch_size 64
