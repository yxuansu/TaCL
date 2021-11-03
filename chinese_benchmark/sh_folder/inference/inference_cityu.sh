CUDA_VISIBLE_DEVICES=3 python  ../../inference.py\
    --model_name cambridgeltl/tacl-bert-base-chinese\
    --saved_ckpt_path ../../pretrained_ckpt/cityu_cws/cityu_ckpt\
    --train_path ../../../benchmark_data/CWS_data/CITYU/cityu_train.txt\
    --dev_path ../../../benchmark_data/CWS_data/CITYU/cityu_test.txt\
    --test_path ../../../benchmark_data/CWS_data/CITYU/cityu_test.txt\
    --label_path ../../../benchmark_data/CWS_data/CITYU/cityu_label.txt\
    --batch_size 64
