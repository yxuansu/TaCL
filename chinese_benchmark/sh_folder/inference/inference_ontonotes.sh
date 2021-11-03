CUDA_VISIBLE_DEVICES=0 python  ../../inference.py\
    --model_name cambridgeltl/tacl-bert-base-chinese\
    --saved_ckpt_path ../../pretrained_ckpt/ontonotes/ontonotes_ckpt\
    --train_path ../../../benchmark_data/NER/OntoNote4NER/OntoNote4NER.train.char.txt\
    --dev_path ../../../benchmark_data/NER/OntoNote4NER/OntoNote4NER.dev.char.txt\
    --test_path ../../../benchmark_data/NER/OntoNote4NER/OntoNote4NER.test.char.txt\
    --label_path ../../../benchmark_data/NER/OntoNote4NER/OntoNote4NER_NER_Label.txt\
    --batch_size 64
