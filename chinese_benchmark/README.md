# Fine-tuning on Chinese Benchmarks

## 1. Conduct inference from our released checkpoints:

### (1). Downloading checkpoints for all evaluated tasks:
```yaml
cd ckpt
chmod +x ./download_checkpoints.sh
./download_checkpoints.sh
```

### (2). Perform inference on different benchmarks:
```yaml
cd ./sh_folder/inference/
chmod +x ./inference_{}.sh
./inference_{}.sh
```

Here, {} is in ['msra', 'ontonotes', 'weibo', 'resume', 'pku'] and some key parameters are described below:

```yaml
--use_db_as_input: Whether use DB result as input. It should be set as the same value as the 
                   --use_db_as_input argument in the training script.
                   
--pretrained_path: The path that stores the model from training. Should be the same value as the 
                   --ckpt_save_path argument in the training script.
                   
--output_save_path: The directory to save the predicted result.
```

