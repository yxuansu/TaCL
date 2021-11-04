# Instructions on recreating our analysis results:

### 1. Recreating the layer-wise cross-similarity plot:
#### (i) Get token representations of sentences sampled from Wikipedia:
```yaml
chmod +x ./analyze_bert.sh
./analyze_bert.sh

chmod +x ./analyze_tacl.sh
./analyze_tacl.sh

chmod +x ./analyze_zh_tacl.sh
./analyze_zh_tacl.sh

chmod +x ./analyze_zh_bert.sh
./analyze_zh_bert.sh
```
This process would take around half hour to complete on a single GPU. Alternatively, you can download our computed results using the command below.
```yaml
chmod +x ./download_json.sh
./download_json.sh
```

#### (ii)Then run the following command and you will get the Figure below.
```yaml
python3 plot_result.py
```
<img src="https://github.com/yxuansu/TaCL/blob/main/analysis/cross-similarity.png" width="400" height="280">

### 2. Plot visualization of cross-similarity matrix:
#### (i) Install required libraries:
```yaml
pip install seaborn
pip install pandas
```

### (ii) Run the following command:
```yaml
python3 plot_cross_similarity_matrix.py
```

Then you will get the visualization of BERT as

<img src="https://github.com/yxuansu/TaCL/blob/main/analysis/bert_heatmap.png" width="400" height="280">

and the visualization of TaCL as

<img src="https://github.com/yxuansu/TaCL/blob/main/analysis/tacl_heatmap.png" width="400" height="280">

**[Note]** To visualize the results of a different sentence, you can freely modify the default text in plot_cross_similarity_matrix.py.
