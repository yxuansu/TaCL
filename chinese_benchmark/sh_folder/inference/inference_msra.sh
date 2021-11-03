CUDA_VISIBLE_DEVICES=0 python  ../../inference.py\
    --model_name cambridgeltl/tacl-bert-base-chinese\
    --saved_ckpt_path ../../pretrained_ckpt/msra/msra_ckpt\
    --train_path ../../../benchmark_data/NER/MSRANER/MSRA.train.char.txt\
    --dev_path ../../../benchmark_data/NER/MSRANER/MSRA.dev.char.txt\
    --test_path ../../../benchmark_data/NER/MSRANER/MSRA.test.char.txt\
    --label_path ../../../benchmark_data/NER/MSRANER/MSRA_NER_Label.txt\
    --batch_size 64
