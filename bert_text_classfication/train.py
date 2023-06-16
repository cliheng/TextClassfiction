import torch
import numpy as np
from torch.optim import AdamW
from tqdm import tqdm
from torch.utils.data import DataLoader

from corpus_process import load_corpus, build_dataloader
from model import BertTextClassfication
from utils import generate_position_embedding
from wobert import WoBertTokenizer

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train(model, dataloader):
    optim = AdamW(model.parameters(), lr=4e-5)
    model.train()
    pbar = tqdm(dataloader)
    for i, input_data in enumerate(pbar):
        input_data = { k:v.to(device) for k,v in input_data.items() }
        
        loss = model(**input_data)

        loss.backward()
        optim.step()

        pbar.set_description(f'train loss:{loss.item():.5f}') 

        model.zero_grad()    


if __name__ == '__main__':
    # 创建模型
    model = BertTextClassfication('junnyu/wobert_chinese_plus_base', dropout_rate=0.2, num_labels=2)
    model.to(device)
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

    # 模型训练
    train(model, dataloader)

