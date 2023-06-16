import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from model import BertTextClassfication
from utils import generate_position_embedding
from wobert import WoBertTokenizer

def load_corpus(corpus_dir):
    """
    加载语料目录中文件
    """
    corpus = {}
    for item in os.listdir(corpus_dir):
        item_path = os.path.join(corpus_dir, item)
        if os.path.isfile(item_path):
            df = pd.read_csv(item_path, sep=r'\t', engine='python')
            if 'label' in df.columns: 
                corpus[item] = df[['label','text_a']].values
            else:
                corpus[item] = df[['text_a']].values
    return corpus


def build_dataloader(dataset, tokenizer, batch_size=32):
    """
    根据DataSet生成DataLoader
    """
    def collcate_fun(batch_data):
        batch_data = np.array(batch_data)
        # 模型输入项
        input_data = tokenizer(list(batch_data[:,1]), padding=True, return_tensors='pt')
        # label
        labels = torch.LongTensor(batch_data[:,0].astype(int))
        input_data['labels'] = labels
        return input_data

    return DataLoader(dataset, shuffle=True, batch_size=batch_size, collate_fn=collcate_fun)


if __name__ == '__main__':
    # 创建模型
    model = BertTextClassfication('junnyu/wobert_chinese_plus_base', num_labels=2)
    # 加载目录中的语料文件
    corpus = load_corpus('corpus')
    # 合并train.tsv, dev.tsv语料
    train_corpus = np.append(corpus['train.tsv'],corpus['dev.tsv'], axis=0)

    # 检测语料最大长度
    max_len = max([len(line[1]) for line in train_corpus])
    # 语料长度大于510, 使用分层位置编码
    if max_len > 510:
        # 创建分层position embedding
        hierarchical_embedding = generate_position_embedding(model.encoder)
        # 新position embedding嵌入现有bert模型
        model.encoder.embeddings.position_embeddings = hierarchical_embedding

    # Tokenizer
    tokenizer= WoBertTokenizer.from_pretrained('junnyu/wobert_chinese_plus_base')
    # 创建dataloader
    dataloader = build_dataloader(train_corpus, tokenizer, batch_size=4)

    for data in dataloader:
        break

    