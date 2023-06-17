import torch
from torch.nn.modules.sparse import Embedding 


class BertHierarchicalPositionEmbedding(Embedding):
    """
    分层位置编码PositionEmbedding
    """
    def __init__(self, alpha=0.4, num_embeddings=512,embedding_dim=768):
        super().__init__(num_embeddings=num_embeddings, embedding_dim=embedding_dim)
        self.input_dim = num_embeddings
        self.alpha = alpha

    def forward(self, input):
        
        input_shape = input.shape
        seq_len = input_shape[1]
        position_ids = torch.arange(0, seq_len, dtype=torch.int64).to(input.device)

        embeddings = self.weight - self.alpha * self.weight[:1]
        embeddings = embeddings / (1 - self.alpha)
        embeddings_x = torch.index_select(embeddings, 0, torch.div(position_ids, self.input_dim, rounding_mode='trunc'))
        embeddings_y = torch.index_select(embeddings, 0, position_ids % self.input_dim)
        embeddings = self.alpha * embeddings_x + (1 - self.alpha) * embeddings_y
        return embeddings

def generate_position_embedding(bert_model):
    """
    通过bert预训练权重创建BertHierarchicalPositionEmbedding并返回
    """
    # 加载bert预训练文件中的position embedding的weight
    embedding_weight = bert_model.embeddings.position_embeddings.weight
    hierarchical_position = BertHierarchicalPositionEmbedding()
    hierarchical_position.weight.data.copy_(embedding_weight)
    # 不参与模型训练
    hierarchical_position.weight.requires_grad = False
    return hierarchical_position

