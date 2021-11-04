gdown "https://drive.google.com/u/1/uc?export=download&confirm=j53u&id=1awNji9IMjwe6kPSiYspZhJYotShWzbaj"
unzip json_zip.zip
rm json_zip.zip
cd json_zip
mv bert_result.json ../
mv tacl_result.json ../
mv zh_bert_result.json ../
mv zh_tacl_result.json ../
cd ..
rm -r json_zip