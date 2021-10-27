## Fine-tuning on English Benchmarks
All the experiments on English benchmarks are conducted using [huggingface transformers](https://github.com/huggingface/transformers) library. It is super easy to use and can be finished with a few commands.

### In the following, we give examples on how to run experiments on SQuAD 1.1 and 2.0 with our released model:
#### (1) First download huggingface:
```yaml
git clone https://github.com/huggingface/transformers
cd transformers
pip install .
```

#### (2) Go to SQuAD directory and prepare environment:
```yaml
cd ./examples/pytorch/question-answering/
pip install -r requirements.txt
```

#### (3) Run experiments on SQuAD 1.1:
```yaml
python run_qa.py \
  --model_name_or_path cambridgeltl/clbert-base-uncased \
  --dataset_name squad \
  --do_train \
  --do_eval \
  --per_device_train_batch_size 12 \
  --learning_rate 3e-5 \
  --num_train_epochs 2 \
  --max_seq_length 384 \
  --doc_stride 128 \
  --output_dir your_path_to_save_model
```

#### (4) Run experiments on SQuAD 2.0:
```yaml
python run_qa.py \
  --model_name_or_path cambridgeltl/clbert-base-uncased \
  --dataset_name squad_v2 \
  --do_train \
  --do_eval \
    --version_2_with_negative \
  --per_device_train_batch_size 12 \
  --learning_rate 3e-5 \
  --num_train_epochs 2 \
  --max_seq_length 384 \
  --doc_stride 128 \
  --output_dir your_path_to_save_model
```

The detailed instructions of running experiments on other benchmarks can be found [here](https://github.com/huggingface/transformers/tree/master/examples).
