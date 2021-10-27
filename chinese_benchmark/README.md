# Fine-tuning on Chinese Benchmarks

## 1. Conduct inference from our released checkpoints:

### (1) Downloading checkpoints for all evaluated tasks:
```yaml
cd ckpt
chmod +x ./download_checkpoints.sh
./download_checkpoints.sh
```

### (2) Perform inference on different benchmarks:
```yaml
cd ./sh_folder/inference/
chmod +x ./inference_{}.sh
./inference_{}.sh
```

Here, {} is in ['msra', 'ontonotes', 'weibo', 'resume', 'pku'] and the parameters are described below:

```yaml
--saved_ckpt_path: The trained model checkpoint path. Remember to modify it when you train your own model.
--train_path: Training data path.
--dev_path: Validation data path.
--test_path: Test data path.
--label_path: Data label path.
--batch_size: Inference batch size.
```

### (3) Results from trained checkpoints:
After running the scripts, you should get the following test set results for different datasets.

|     Dataset | Precision       |Recall|F1|
| :-------------: |:-------------:|:-----:|:-----:|
|MSRA||||
|OntoNotes|81.88|82.98|82.42|
|Resume|96.48|96.42|96.45|
|Weibo|68.40|70.73|69.54|
|PKU||||

## 2. Train a new model:
```yaml
cd ./sh_folder/train/
chmod +x ./{}.sh
./{}.sh
```
Here, {} is in ['msra', 'onto', 'weibo', 'resume', 'pku'] and the parameters are described below:

```yaml
--model_name: The name of our released Chinese model, cambridgeltl/clbert-base-chinese.
--train_path: Training data path.
--dev_path: Validation data path.
--test_path: Test data path.
--label_path: Data label path.
--learning_rate: The learning rate. 
--number_of_gpu: Number of available GPUs.
--number_of_runs: Number of different runs you want to experiment on the benchmark.
--save_path_prefix: The path to save the trained model.
```

**[Note 1]** We do not do any learning selection, the 2e-5 is just a default value. Better results are possible if you choose to use a different learning rate.

**[Note 2]** The actual batch size equals to gradient_accumulation_steps x number_of_gpu x batch_size_per_gpu. We recommend
you to set the actual batch size value as 128.





