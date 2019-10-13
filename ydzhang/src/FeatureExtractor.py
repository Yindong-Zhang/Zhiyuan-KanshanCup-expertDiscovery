from torch import nn
import torch
import torch.functional as F

class SimpleFeatureExtractor(nn.Module):
    """
    特征提取，包括稀疏类别特征　和　连续数量特征
    """
    def __init__(self,
                 feature_dim_dict,
                 embedding_size,
                 output_dim,
                 ):
        """
        :param feature_dim_dict: dict,to indicate sparse field and dense field like {'sparse':{'field_1':4,'field_2':3,'field_3':2},'dense':['field_4','field_5']}

        """
        super(SimpleFeatureExtractor, self).__init__()
        self.feature_dim_dict = feature_dim_dict
        self.embedding_layer_dict = nn.ModuleDict()
        for sparse_feat, feat_dim in feature_dim_dict['sparse'].item():
            self.embedding_layer_dict[sparse_feat] = nn.Embedding(feat_dim, embedding_size)
        self.embedding_size = embedding_size
        self.hidden_size = embedding_size * len(feature_dim_dict['sparse']) + len(feature_dim_dict['dense'])
        self.output_dim = output_dim
        self.linear = nn.Linear(self.hidden_size, output_dim)

    def forward(self, input_dict):
        """

        :param input_dict: a input dict like {'sparse': {'field_1': feature_column, 'field_2': feature_column}, 'dense': {'field_3': feature column, 'field 4': feature_column'}}
        :return:
        """
        sparse_feature_embeddings = []
        for sparse_feat, feat_column in input_dict['sparse'].item():
            sparse_feature_embeddings.append(self.embedding_layer_dict[sparse_feat](feat_column))

        sparse_feature_concated = torch.cat(sparse_feature_embeddings, dim= -1)

        dense_feature_concated = torch.cat(input_dict['dense'].values(), dim= -1)

        feature_concated = torch.cat([sparse_feature_concated, dense_feature_concated], dim= -1)

        output = nn.ReLU()(self.linear(feature_concated))

        return output

    def extra_repr(self):
        return "AnswerFeature: %s with embedding size %d to hidden variable with dim %d" %(self.feature_dim_dict, self.embedding_size, self.output_dim)
