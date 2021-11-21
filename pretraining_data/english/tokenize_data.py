import argparse, os
import progressbar

def tokenize_line(text, tokenizer):
    token_list = tokenizer.tokenize(text.strip('\n'), max_length=512, truncation=True)
    return ' '.join(token_list).strip()

def process_file(in_f, out_f, tokenizer):
    with open(in_f, 'r', encoding = 'utf8') as i:
        lines = i.readlines()
    line_num = len(lines)
    
    with open(out_f, 'w', encoding = 'utf8') as o:
        p = progressbar.ProgressBar(line_num)
        p.start()
        for idx in range(line_num):
            p.update(idx)
            line = lines[idx]
            tokenized_text = tokenize_line(line.strip('\n'), tokenizer)
            if len(tokenized_text) == 0:
                o.writelines('\n')
            else:
                o.writelines(tokenized_text + '\n')
        p.finish()

def parse_config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, help="roberta-* or bert-*")
    parser.add_argument("--raw_data_path", type=str)
    parser.add_argument("--output_dir", type=str)
    parser.add_argument("--output_name", type=str)
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_config()

    import os
    if os.path.exists(args.output_dir):
        pass
    else: # recursively construct directory
        os.makedirs(args.output_dir, exist_ok=True)

    model_name = args.model_name
    print ('Loading tokenizer...')
    if model_name.startswith('bert'):
        from transformers import BertTokenizerFast
        tokenizer = BertTokenizerFast.from_pretrained(model_name)
    elif model_name.startswith('roberta'):
        from transformers import RobertaTokenizerFast
        tokenizer = RobertaTokenizerFast.from_pretrained(model_name)
    else:
        raise Exception('Wrong tokenizer configuration')

    out_f = args.output_dir + '/' + args.output_name
    process_file(args.raw_data_path, out_f, tokenizer)
