from funcs import *
from datasets import load_dataset

if __name__ == '__main__':
    dataset = load_dataset('wikipedia', "20200501.en", split='train')
    stop_prefix_list = ['References', 'External links', 'Category:', 'See also']
    all_doc_list = process_corpus(dataset, stop_prefix_list)
    out_f = './english_wiki.txt'
    with open(out_f, 'w', encoding = 'utf8') as o:
        for doc in all_doc_list:
            for sen in doc:
                o.writelines(sen + '\n')
            o.writelines('\n')

    # write example data
    out_f = r'./english_wiki_20k_lines.txt'
    with open('./english_wiki.txt', 'r', encoding = 'utf8') as i:
        with open(out_f, 'w', encoding = 'utf8') as o:
            lines = i.readlines()[:20000]
            for l in lines:
                o.writelines(l)