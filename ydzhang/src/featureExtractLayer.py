from torch import nn
import torch
import torch.functional as F
from src.layers import FullyConnectedLayer
class EmbeddingMLPLayer(nn.Module):
    """
    特征提取，包括稀疏类别特征　和　连续数量特征
    """
    def __init__(self,
                 feature_dim_dict,
                 embedding_size,
                 mlp_hidden_list,
                 activation = 'relu',
                 dropout = 0.1,
                 use_bn = True
                 ):
        """
        :param feature_dim_dict: dict,to indicate sparse field and dense field like {'sparse':{'field_1':4,'field_2':3,'field_3':2},'dense':{'field_4': 128,'field_5': 1}}

        """
        super(EmbeddingMLPLayer, self).__init__()
        self.feature_dim_dict = feature_dim_dict
        self.embedding_layer_dict = nn.ModuleDict()
        for sparse_feat, feat_dim in feature_dim_dict['sparse'].items():
            self.embedding_layer_dict[sparse_feat] = nn.Embedding(feat_dim, embedding_size)
        self.embedding_size = embedding_size
        self.hidden_size = embedding_size * len(feature_dim_dict['sparse']) + sum(feature_dim_dict['dense'].values())
        self.mlp_hidden_list = mlp_hidden_list
        self.mlp = FullyConnectedLayer(self.hidden_size,
                                       hidden_size= self.mlp_hidden_list,
                                       dropout_rate= dropout,
                                       batch_norm= use_bn,
                                       activation= activation)

    def forward(self, input_dict):
        """

        :param input_dict: a input dict like {'sparse': {'field_1': (batchsize, ), 'field_2': (batchsize, )}, 'dense': {'field_3': (batchsize, 1), 'field 4': (batchsize, 1)}}
        :return:
        """
        sparse_feature_embeddings = []
        for sparse_feat, feat_column in input_dict['sparse'].items():
            sparse_feature_embeddings.append(self.embedding_layer_dict[sparse_feat](feat_column))

        sparse_feature_concated = torch.cat(sparse_feature_embeddings, dim= -1)

        dense_feature_concated = torch.cat(list(input_dict['dense'].values()), dim= -1)

        feature_concated = torch.cat([sparse_feature_concated, dense_feature_concated], dim= -1)

        output = self.mlp(feature_concated)

        return output

    def extra_repr(self):
        return "EmbeddingMLPFeatureExtractor: %s with embedding size %d to hidden variable with dim %s" %(self.feature_dim_dict, self.embedding_size, self.mlp_hidden_list)

class EmbeddingConcatLayer(nn.Module):
    """
    特征提取，包括稀疏类别特征　和　连续数量特征
    """
    def __init__(self,
                 feature_dim_dict,
                 embedding_size,
                 ):
        """
        :param feature_dim_dict: dict,to indicate sparse field and dense field like {'sparse':{'field_1':4,'field_2':3,'field_3':2},'dense':{'field_4': 128,'field_5': 1}}

        """
        super(EmbeddingConcatLayer, self).__init__()
        self.feature_dim_dict = feature_dim_dict
        self.embedding_layer_dict = nn.ModuleDict()
        for sparse_feat, feat_dim in feature_dim_dict['sparse'].items():
            self.embedding_layer_dict[sparse_feat] = nn.Embedding(feat_dim, embedding_size)
        self.embedding_size = embedding_size
        self.output_size = embedding_size * len(feature_dim_dict['sparse']) + sum(feature_dim_dict['dense'].values())

    def forward(self, input_dict):
        """

        :param input_dict: a input dict like {'sparse': {'field_1': (batchsize, ), 'field_2': (batchsize, )}, 'dense': {'field_3': (batchsize, 1), 'field 4': (batchsize, 1)}}
        :return:
        """
        sparse_feature_embeddings = []
        for sparse_feat, feat_column in input_dict['sparse'].items():
            sparse_feature_embeddings.append(self.embedding_layer_dict[sparse_feat](feat_column))

        sparse_feature_concated = torch.cat(sparse_feature_embeddings, dim= -1)

        dense_feature_concated = torch.cat(list(input_dict['dense'].values()), dim= -1)

        feature_concated = torch.cat([sparse_feature_concated, dense_feature_concated], dim= -1)


        return feature_concated

    def extra_repr(self):
        return "EmbeddingConcatFeatureExtractor: %s with embedding size %d to hidden variable with dim %d" %(self.feature_dim_dict, self.embedding_size, self.output_size)

def questionFeatureMerge(title_feat, describe_feat, topic_feat):
    """

    :param title_feat: (b, d_tl)
    :param describe_feat: (b, d_ds)
    :param topic_feat: (b, d_tp)
    :return:
    """
    return torch.cat([title_feat, describe_feat, topic_feat], dim= -1)

if __name__ == '__main__':
    input_dim_dict = {'sparse': {'a': 5, 'b': 12}, 'dense': {'d': 1, 'e': 1}}
    batchsize = 5
    input = {'sparse': {'a': torch.randint(0, 5, (batchsize, )), 'b': torch.randint(0, 12, (batchsize, ))},
             'dense': {'d': torch.randn((batchsize, 1)), 'e': torch.randn((batchsize, 1))}}
    t = EmbeddingConcatLayer(input_dim_dict, 16)

    out = t(input)
    print(out.detach().numpy())