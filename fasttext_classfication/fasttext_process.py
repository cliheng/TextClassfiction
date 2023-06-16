import os
import numpy as np
import pandas as pd
import jieba
import fasttext
import csv

def convert_corpus(corpus_dir):
    """
    加载语料目录中文件
    """
    files = []
    for item in os.listdir(corpus_dir):
        item_path = os.path.join(corpus_dir, item)
        if os.path.isfile(item_path):
            df = pd.read_csv(item_path, sep=r'\t', engine='python')
            df['text_a'] = df['text_a'].apply(lambda x: ' '.join(jieba.lcut(x)))
            save_path = os.path.join(os.path.dirname(__file__), os.path.splitext(item)[0] + '.txt')
            files.append(save_path)
            if 'label' in df.columns: 
                df['label'] = df['label'].map({1:'__label__positive',0:'__label__negative'})
                df[['label','text_a']].to_csv(save_path ,sep=' ', header=False, index=False, quoting=csv.QUOTE_NONE, escapechar=' ')
            else:
                df[['text_a']].to_csv(save_path, sep=' ', header=False, index=False, quoting=csv.QUOTE_NONE, escapechar=' ')
    return files

if __name__ == '__main__':
    # 加载并转换目录中的语料文件
    files = convert_corpus('corpus')
    print(files)

    model = fasttext.train_supervised(files[0])

    for i, line in enumerate(open(files[-1])):
        label,prob = model.predict(line.strip())
        print(f'text:{ "".join(line.strip().split()) } \nlabel:{label}\nprob:{prob}')
        if i > 5:
            break
    


    