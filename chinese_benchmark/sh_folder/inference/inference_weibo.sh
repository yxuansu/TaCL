CUDA_VISIBLE_DEVICES=0 python  ../../inference.py\
    --model_name cambridgeltl/tacl-bert-base-chinese\
    --saved_ckpt_path ../../pretrained_ckpt/weibo/weibo_ckpt\
    --train_path ../../../benchmark_data/NER/WeiboNER/Weibo.train.all.char.txt\
    --dev_path ../../../benchmark_data/NER/WeiboNER/Weibo.dev.all.char.txt\
    --test_path ../../../benchmark_data/NER/WeiboNER/Weibo.test.all.char.txt\
    --label_path ../../../benchmark_data/NER/WeiboNER/Weibo_NER_Label.txt\
    --batch_size 64


