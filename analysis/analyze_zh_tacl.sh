CUDA_VISIBLE_DEVICES=3 python  ./layerwise_intra_sentence_similarity.py\
    --model_name cambridgeltl/tacl-bert-base-chinese\
    --file_path ./zh_wiki_randomly_select_50k.txt\
    --output_path ./zh_tacl_result.json

