# CLBERT: Improving BERT with Contrastive Token Regularization

### 1. Environment Setup:
```yaml
pip3 install -r requirements.txt
```
### 2. Training CLBERT:

### 3. Experiments on English Benchmarks:
Please refer to details provided in ./english_benchmark directory.

### 4. Experiments on Chinese Benchmarks:
#### (1) Chinese Benchmark Data Preparation:
```yaml
chmod +x ./download_benchmark_data.sh
./download_benchmark_data.sh
```
#### (2) Fine-tuning and Inference:
Please refer to details provided in ./chinese_benchmark directory.

### 5. Heatmap Visualization:
We also provide code to replicate our heatmap analysis. The two jupyter notebooks are provided in ./heatmap_analysis directory. You can choose different sentences that are sampled from Wikipedia corpus to compare the results of vanilla BERT model and our CLBERT. Have fun!


