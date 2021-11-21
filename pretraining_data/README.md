### Prepare English Wikipedia
(1) Install environment
```yaml
pip install datasets 
```

(2) Download raw Wikipedia text:
```yaml
cd english
chmod +x ./download_raw_data.sh
./download_raw_data.sh
```
This process takes around 2 hours.

(3) Create pre-training data for BERT:
```yaml
cd english
chmod +x ./tokenize_bert_uncased_data_example.sh
./tokenize_bert_uncased_data_example.sh
chmod +x ./tokenize_bert_uncased_data.sh
./tokenize_bert_uncased_data.sh
```
This process takes around 10 hours on a local laptop.

### Prepare Chinese Wikipedia
Due to copy right issue, we will not publicly redistribute our processed Chinese Wikipedia data. If you need the data, please contact me via (ys484 at cam.ac.uk).
