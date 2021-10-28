# CLBERT: Improving BERT with Contrastive Token Regularization

## Huggingface Models:

|Model Name|Model Address|
|:-------------:|:-------------:|
|English CLBERT (cambridgeltl/clbert-base-uncased)|[link](https://huggingface.co/cambridgeltl/clbert-base-uncased)|
|Chinese CLBERT (cambridgeltl/clbert-base-chinese)|[link](https://huggingface.co/cambridgeltl/clbert-base-chinese)|

## Main Results:
|Model Name|Model Address|
|<td colspan=4>triple|



### 1. Environment Setup:
```yaml
pip3 install -r requirements.txt
```
### 2. Train CLBERT:
#### (1) Prepare pre-training data:
Please refer to details provided in ./pretraining_data directory.
#### (2) Train the model:
Please refer to details provided in ./pretraining directory.

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
We provide code to replicate our heatmap analysis. The jupyter notebooks are provided in ./heatmap_analysis directory. You can choose different sentences that are sampled from Wikipedia corpus to compare the results of vanilla BERT model and our CLBERT. **Have fun!**


