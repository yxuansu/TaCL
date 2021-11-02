# TaCL: Improving BERT Pre-training with Token-aware Contrastive Learning

## Main Results:

We show the comparison between our TaCL (base version) and the original BERT (base version). 

(1) English benchmark results on **[SQuAD (Rajpurkar et al., 2018)](https://rajpurkar.github.io/SQuAD-explorer/)** (dev set) and **[GLUE (Wang et al., 2019)](https://gluebenchmark.com/)** average score.
|**Model**|SQuAD 1.1 EM/F1|SQuAD 2.0 EM/F1|GLUE Average|
|:-------------:|:-------------:|:-------------:|:-------------:|
|BERT|80.8/88.5|73.4/76.8|79.6|
|TaCL|**81.4/88.9**|**74.0/77.2**|**81.2**|

(2) Chinese benchmark results (test set F1) on four NER tasks (MSRA, OntoNotes, Resume, and Weibo) and three Chinese word segmentation (CWS) tasks (PKU, CityU, and AS).
|**Model**|MSRA|OntoNotes|Resume|Weibo|PKU|CityU|AS|
|:-------------:|:-------------:|:-------------:|:-------------:|:-------------:|:-------------:|:-------------:|:-------------:|
|BERT|94.95|80.14|95.53|68.20|96.50|97.60||
|TaCL|**95.44**|**82.42**|**96.45**|**69.54**|**96.75**|**98.16**||
## Huggingface Models:

|Model Name|Model Address|
|:-------------:|:-------------:|
|English (**cambridgeltl/tacl-bert-base-uncased**)|[link](https://huggingface.co/cambridgeltl/tacl-bert-base-uncased)|
|Chinese (**cambridgeltl/tacl-bert-base-chinese**)|[link](https://huggingface.co/cambridgeltl/tacl-bert-base-chinese)|

## Example Usage:
```python
import torch
# initialize model
from transformers import AutoModel, AutoTokenizer
model_name = 'cambridgeltl/tacl-bert-base-uncased'
model = AutoModel.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
# create input ids
text = '[CLS] clbert is awesome. [SEP]'
tokenized_token_list = tokenizer.tokenize(text)
input_ids = torch.LongTensor(tokenizer.convert_tokens_to_ids(tokenized_token_list)).view(1, -1)
# compute hidden states
representation = model(input_ids).last_hidden_state # [1, seqlen, embed_dim]
```

### 1. Environment Setup:
```yaml
python version 3.8
pip3 install -r requirements.txt
```
### 2. Train TaCL:
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


