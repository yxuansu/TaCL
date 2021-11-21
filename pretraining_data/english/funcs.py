import re
import progressbar

alphabets= "([A-Za-z])"
prefixes = "(Mr|St|Mrs|Ms|Dr)[.]"
suffixes = "(Inc|Ltd|Jr|Sr|Co)"
starters = "(Mr|Mrs|Ms|Dr|He\s|She\s|It\s|They\s|Their\s|Our\s|We\s|But\s|However\s|That\s|This\s|Wherever)"
acronyms = "([A-Z][.][A-Z][.](?:[A-Z][.])?)"
websites = "[.](com|net|org|io|gov)"

def split_into_sentences(text):
    text = " " + text + "  "
    text = text.replace("\n"," ")
    text = re.sub(prefixes,"\\1<prd>",text)
    text = re.sub(websites,"<prd>\\1",text)
    if "Ph.D" in text: text = text.replace("Ph.D.","Ph<prd>D<prd>")
    text = re.sub("\s" + alphabets + "[.] "," \\1<prd> ",text)
    text = re.sub(acronyms+" "+starters,"\\1<stop> \\2",text)
    text = re.sub(alphabets + "[.]" + alphabets + "[.]" + alphabets + "[.]","\\1<prd>\\2<prd>\\3<prd>",text)
    text = re.sub(alphabets + "[.]" + alphabets + "[.]","\\1<prd>\\2<prd>",text)
    text = re.sub(" "+suffixes+"[.] "+starters," \\1<stop> \\2",text)
    text = re.sub(" "+suffixes+"[.]"," \\1<prd>",text)
    text = re.sub(" " + alphabets + "[.]"," \\1<prd>",text)
    if "”" in text: text = text.replace(".”","”.")
    if "\"" in text: text = text.replace(".\"","\".")
    if "!" in text: text = text.replace("!\"","\"!")
    if "?" in text: text = text.replace("?\"","\"?")
    text = text.replace(".",".<stop>")
    text = text.replace("?","?<stop>")
    text = text.replace("!","!<stop>")
    text = text.replace("<prd>",".")
    sentences = text.split("<stop>")
    sentences = sentences[:-1]
    sentences = [s.strip() for s in sentences]
    return sentences

def process_chunk_data(item, stop_prefix_list):
    article = item['text']
    article_list = article.split('\n')
    res_list = []
    break_flag = False
    for text in article_list:
        for prefix in stop_prefix_list:
            if text.startswith(prefix):
                break_flag = True
                
        if len(text.split()) < 3:
            pass
        else:
            res_list.append(text)
        if break_flag:
            break
            
    sentence_list = []
    for text in res_list:
        one_sen_list = split_into_sentences(text)
        for sen in one_sen_list:
            if len(sen) < 3:
                pass
            else:
                sentence_list.append(sen)
    return sentence_list

def process_corpus(dataset, stop_prefix_list):
    doc_num = len(dataset)
    p = progressbar.ProgressBar(doc_num)
    print ('Start processing data...')
    p.start()
    all_doc_list = []
    for idx in range(doc_num):
        p.update(idx)
        sentence_list = process_chunk_data(dataset[idx], stop_prefix_list)
        if len(sentence_list) < 2:
            continue
        all_doc_list += [sentence_list]
    p.finish()
    return all_doc_list

if __name__ == '__main__':
    from datasets import load_dataset
    dataset = load_dataset('wikipedia', "20200501.en", split='train')
    stop_prefix_list = ['References', 'External links', 'Category:', 'See also']
    all_doc_list = process_corpus(dataset, stop_prefix_list)

    out_f = './eng_wiki.txt'
    with open(out_f, 'w', encoding = 'utf8') as o:
        for doc in all_doc_list:
            for sen in doc:
                o.writelines(sen + '\n')
            o.writelines('\n')









