CUDA_VISIBLE_DEVICES=3 python  ./layerwise_intra_sentence_similarity.py\
    --model_name cambridgeltl/tacl-bert-base-uncased\
    --file_path ./en_wiki_randomly_select_50k.txt\
    --output_path ./tacl_result.json

