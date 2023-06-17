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
            # 文本分词
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
    # # 模型训练
    # model = fasttext.train_supervised(files[-1])
    model = fasttext.train_supervised(files[-1],lr=0.5, epoch=25, wordNgrams=2, bucket=200000, dim=50, loss='ova')
    # # 模型保存
    # model.save_model('text_classfication.mod')
    # 模型加载
    model = fasttext.load_model('text_classfication.mod')
    # 模型推理
    label,prob = model.predict('效果 很 一般')
    print(f'text:{ "效果很一般" } \nlabel:{label}\nprob:{prob}')
    # 模型测试
    result = model.test(files[0], k=-1)
    print(result)


    