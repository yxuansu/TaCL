CUDA_VISIBLE_DEVICES=0 python  ../../inference.py\
    --model_name cambridgeltl/tacl-bert-base-chinese\
    --saved_ckpt_path ../../pretrained_ckpt/resume/resume_ckpt\
    --train_path ../../../benchmark_data/NER/ResumeNER/ResumeNER.train.char.txt\
    --dev_path ../../../benchmark_data/NER/ResumeNER/ResumeNER.dev.char.txt\
    --test_path ../../../benchmark_data/NER/ResumeNER/ResumeNER.test.char.txt\
    --label_path ../../../benchmark_data/NER/ResumeNER/ResumeNER_NER_label.txt\
    --batch_size 64
