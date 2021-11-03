CUDA_VISIBLE_DEVICES=3 python  ../../train.py\
    --model_name cambridgeltl/tacl-bert-base-chinese\
    --train_path ../../../benchmark_data/NER/OntoNote4NER/OntoNote4NER.train.char.txt\
    --dev_path ../../../benchmark_data/NER/OntoNote4NER/OntoNote4NER.dev.char.txt\
    --test_path ../../../benchmark_data/NER/OntoNote4NER/OntoNote4NER.test.char.txt\
    --label_path ../../../benchmark_data/NER/OntoNote4NER/OntoNote4NER_NER_Label.txt\
    --learning_rate 2e-5\
    --batch_size 64\
    --batch_size_per_gpu 64\
    --number_of_gpu 1\
    --gradient_accumulation_steps 2\
    --total_epochs 40\
    --number_of_runs 5\
    --save_path_prefix ../../ckpt/ontonotes/


