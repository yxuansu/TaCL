# Instructions on recreating our analysis results:

### 1. Recreating the layer-wise cross-similarity plot:
<img src="https://github.com/yxuansu/TaCL/blob/main/analysis/cross-similarity.png" width="500" height="350">
#### (1) Get token representations of sentences sampled from Wikipedia:
```yaml
# 
chmod +x ./analyze_bert.sh
./analyze_bert.sh

chmod +x ./analyze_tacl.sh
./analyze_tacl.sh

chmod +x ./analyze_zh_tacl.sh
./analyze_zh_tacl.sh

chmod +x ./analyze_zh_bert.sh
./analyze_zh_bert.sh
```

