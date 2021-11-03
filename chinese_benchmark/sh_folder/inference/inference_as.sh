CUDA_VISIBLE_DEVICES=3 python  ../../inference.py\
    --model_name cambridgeltl/tacl-bert-base-chinese\
    --saved_ckpt_path ../../pretrained_ckpt/as_cws/as_ckpt\
    --train_path ../../../benchmark_data/CWS_data/AS/as_train.txt\
    --dev_path ../../../benchmark_data/CWS_data/AS/as_test.txt\
    --test_path ../../../benchmark_data/CWS_data/AS/as_test.txt\
    --label_path ../../../benchmark_data/CWS_data/AS/as_label.txt\
    --batch_size 64
