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
|OntoNotes|0.819|0.830|0.824|
|Resume||||
|Weibo||||
|PKU||||


