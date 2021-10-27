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
```

