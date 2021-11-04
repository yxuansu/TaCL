import torch
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import torch.nn.functional as F

def compute_correlation_matrix(model, tokenizer, text):
    text = '[CLS] ' + text.strip('\n') + ' [SEP]'
    token_list = tokenizer.tokenize(text)
    input_ids = torch.LongTensor(tokenizer.convert_tokens_to_ids(token_list)).view(1,-1)
    _, seq_len = input_ids.size()
    hidden = model(input_ids).last_hidden_state
    norm_hidden = hidden / hidden.norm(dim=2, keepdim=True)
    correlation_matrix = torch.matmul(norm_hidden, norm_hidden.transpose(1,2)).view(seq_len, seq_len)
    return correlation_matrix.detach().numpy(), token_list

from transformers import AutoModel, AutoTokenizer
def load_model(model_name):
    model = AutoModel.from_pretrained(model_name)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return model, tokenizer

import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

if __name__ == '__main__':
    # load BERT
    model_name = 'bert-base-uncased'
    bert, bert_tokenizer = load_model(model_name)

    # load TaCL
    model_name = 'cambridgeltl/tacl-bert-base-uncased'
    tacl, tacl_tokenizer = load_model(model_name)

    text = "His best friend is Count Alessandro Sturani, extremely awkward but a good-hearted man."
    bert_res, token_list = compute_correlation_matrix(bert, bert_tokenizer, text)

    # Create a dataset
    df = pd.DataFrame(bert_res, 
                      index=token_list,
                      columns=token_list)


    sns.heatmap(df, cmap="Blues")
    plt.savefig('bert_heatmap.png', format='png', dpi=500, bbox_inches = 'tight')
    plt.show()

    tacl_res, token_list = compute_correlation_matrix(tacl, tacl_tokenizer, text)

    df = pd.DataFrame(tacl_res, 
                      index=token_list,
                      columns=token_list)

    sns.heatmap(df, cmap="Blues")
    plt.savefig('tacl_heatmap.png', format='png', dpi=500, bbox_inches = 'tight')
    plt.show()
