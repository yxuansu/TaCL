### All the experiments on English benchmarks are conducted using [huggingface transformers](https://github.com/huggingface/transformers) library. The detailed instructions can be found [here](https://github.com/huggingface/transformers/tree/master/examples). 

### In the following, we give examples on how to run experiments on SQuAD 1.1 and 2.0
#### (1) First download huggingface:
```yaml
git clone https://github.com/huggingface/transformers
cd transformers
pip install .
```

#### (2) Go to SQuAD directory and prepare environment:
```yaml
cd ./pytorch/question-answering/
pip install -r requirements.txt
```
