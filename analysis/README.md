# Instructions on recreating our analysis results:

### 1. Recreating the layer-wise cross-similarity plot:
#### (1) Get token representations of sentences sampled from Wikipedia:
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
Then run the following command and you will get the Figure below.
```yaml
python3 plot_result.py
```
<img src="https://github.com/yxuansu/TaCL/blob/main/analysis/cross-similarity.png" width="400" height="280">
