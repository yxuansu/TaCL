# TaCL: Improving BERT Pre-training with Token-aware Contrastive Learning
**Authors**: Yixuan Su, Fangyu Liu, Zaiqiao Meng, Tian Lan, Lei Shu, Ehsan Shareghi, and Nigel Collier

Code of our paper: [TaCL: Improving BERT Pre-training with Token-aware Contrastive Learning](https://arxiv.org/abs/2111.04198)

[[使用中文TaCL-BERT进行中文命名实体识别及中文分词教程]](https://github.com/yxuansu/Chinese-TaCL-BERT-NER-CWS)

### News:
* [2022/04/08] TaCL is accepted to NAACL 2022!
* [2021/11/09]  The first version of TaCL is released.


## Introduction:
Masked language models (MLMs) such as BERT and RoBERTa have revolutionized the field of Natural Language Understanding in the past few years. However, existing pre-trained MLMs often output an anisotropic distribution of token representations that occupies a narrow subset of the entire representation space. Such token representations are not ideal, especially for tasks that demand discriminative semantic meanings of distinct tokens. In this work, we propose **TaCL** (**T**oken-**a**ware **C**ontrastive **L**earning), a novel continual pre-training approach that encourages BERT to learn an isotropic and discriminative distribution of token representations. TaCL is fully unsupervised and requires no additional data. We extensively test our approach on a wide range of English and Chinese benchmarks. The results show that TaCL brings consistent and notable improvements over the original BERT model. Furthermore, we conduct detailed analysis to reveal the merits and inner-workings of our approach.

<img src="https://github.com/yxuansu/TaCL/blob/main/overview.png" width="400" height="280">



## Main Results:

We show the comparison between TaCL (base version) and the original BERT (base version). 

(1) English benchmark results on **[SQuAD (Rajpurkar et al., 2018)](https://rajpurkar.github.io/SQuAD-explorer/)** (dev set) and **[GLUE (Wang et al., 2019)](https://gluebenchmark.com/)** average score.
|**Model**|SQuAD 1.1 (EM/F1)|SQuAD 2.0 (EM/F1)|GLUE Average|
|:-------------:|:-------------:|:-------------:|:-------------:|
|BERT|80.8/88.5|73.4/76.8|79.6|
|TaCL|**81.6/89.0**|**74.4/77.5**|**81.2**|

(2) Chinese benchmark results (test set F1) on four NER tasks (MSRA, OntoNotes, Resume, and Weibo) and three Chinese word segmentation (CWS) tasks (PKU, CityU, and AS).
|**Model**|MSRA|OntoNotes|Resume|Weibo|PKU|CityU|AS|
|:-------------:|:-------------:|:-------------:|:-------------:|:-------------:|:-------------:|:-------------:|:-------------:|
|BERT|94.95|80.14|95.53|68.20|96.50|97.60|96.50|
|TaCL|**95.44**|**82.42**|**96.45**|**69.54**|**96.75**|**98.16**|**96.75**|
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

## Tutorial on how to reproduce the results in our paper:
### 1. Environment Setup:
```yaml
python version: 3.8
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

### 5. Replicate Our Analysis Results:
We provide all essential code to replicate the results (the images below) provided in our analysis section. The related codes and instructions are located in ./analysis directory. **Have fun!** 

<img src="https://github.com/yxuansu/TaCL/blob/main/analysis/self-similarity.png" width="380" height="250">
<img src="https://github.com/yxuansu/TaCL/blob/main/analysis/bert_heatmap.png" width="380" height="280">
<img src="https://github.com/yxuansu/TaCL/blob/main/analysis/tacl_heatmap.png" width="380" height="280">

### Citation:
If you find our paper and resources useful, please kindly cite our paper:

```bibtex
@article{DBLP:journals/corr/abs-2111-04198,
  author    = {Yixuan Su and
               Fangyu Liu and
               Zaiqiao Meng and
               Tian Lan and
               Lei Shu and
               Ehsan Shareghi and
               Nigel Collier},
  title     = {TaCL: Improving {BERT} Pre-training with Token-aware Contrastive Learning},
  journal   = {CoRR},
  volume    = {abs/2111.04198},
  year      = {2021},
  url       = {https://arxiv.org/abs/2111.04198},
  eprinttype = {arXiv},
  eprint    = {2111.04198},
  timestamp = {Wed, 10 Nov 2021 16:07:30 +0100},
  biburl    = {https://dblp.org/rec/journals/corr/abs-2111-04198.bib},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}
```

### Contact
If you have any questions, feel free to contact me via (ys484@cam.ac.uk).
