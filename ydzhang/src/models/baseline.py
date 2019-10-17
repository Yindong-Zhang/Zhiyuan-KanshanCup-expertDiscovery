from torch import nn
import torch
import torch.functional as F
from src.featureExtractLayer import EmbeddingMLPLayer
from src.layers import BiLSTMRCNN
from src.DIN import DIN
from src.attentionSequencePoolingLayer import SequenceAttentionPoolingLayer

import numpy as np
dataDir = './test'
word_embedding = np.random.randn(1024, 128)

# baseline using pooling layer to extract semantic info rather than RCNN
wv_size = 128
batchsize = 128
hist_length = 10
hist_embed_dim = 128
query_dim = 128
user_profile_dim = 512
# TODO:
query_features = {'sparse': {
    'has_describe': torch.randint(0, 2, (batchsize, ))
    },
    'dense':
        {'topic_feat': torch.randn((batchsize, wv_size)),
         'title_feat': torch.randn((batchsize, wv_size)),
         'describe_length': torch.randint(0, 256, (batchsize, 1)).float(),
         'title_length': torch.randint(0, 64, (batchsize, 1)).float()}
    }
query_features_embed = EmbeddingMLPLayer({'sparse': {'has_describe': 2}, 'dense': {'topics_feat': wv_size,
                                                                             'title_feat': wv_size,
                                                                             'describe_length': 1,
                                                                             'title_lenght': 1}},
                                   embedding_size= 32, output_dim= query_dim)(query_features)
history_feat_dict = {'sparse':{
    'is_good': 2,
    'is_recommend': 2,
    'has_picture': 2,
    'has_video': 2,
    },
    'dense': {'topic_feat': wv_size,
              'title_feat': wv_size,
              'word_count': 1,
              'num_zan': 1,
              'num_cancel_zan': 1,
              'num_comment': 1,
              'num_collect': 1,
              'num_thanks': 1,
              'num_report':1,
              'num_useless': 1,
              'num_oppose': 1}
    }
def fake_history_features(hist_length):
    answer_features = {'sparse': {
        'is_good': torch.randint(0, 2, (hist_length, )),
        'is_recommend': torch.randint(0, 2, (hist_length, )),
        'has_picture': torch.randint(0, 2, (hist_length, )),
        'has_video': torch.randint(0, 2, (hist_length, )),
    },
        'dense': {
            'topic_feat': torch.randn((hist_length, wv_size)),
            'title_feat': torch.randn((hist_length, wv_size)),
            'word_count': torch.randint(0, 256, (hist_length, 1)).float(),
            'num_zan': torch.randint(0, 2046, (hist_length, 1)).float(),
            'num_report': torch.randint(0, 2046, (hist_length, 1)).float(),
            'num_cancel_zan': torch.randint(0, 2046, (hist_length, 1)).float(),
            'num_comment': torch.randint(0, 2046, (hist_length, 1)).float(),
            'num_collect': torch.randint(0, 2046, (hist_length, 1)).float(),
            'num_thanks': torch.randint(0, 2046, (hist_length, 1)).float(),
            'num_report': torch.randint(0, 2046, (hist_length, 1)).float(),
            'num_useless': torch.randint(0, 2046, (hist_length, 1)).float(),
            'num_oppose': torch.randint(0, 2046, (hist_length, 1)).float(),
            }
    }
    return answer_features

# history input should be a list of list
hist_length_list = [np.random.randint(0, 11) for _ in range(batchsize)]
hist_feature_list = [fake_history_features(hist_length_list[i]) for i in range(batchsize)]
hist_features_extract_layer = EmbeddingMLPLayer(history_feat_dict, embedding_size= 8, output_dim= hist_embed_dim)
hist_features = torch.zeros((batchsize, hist_length, hist_embed_dim))
no_hist_embedding = nn.Parameter(torch.randn((hist_embed_dim, )))
for i in range(batchsize):
    # TODO:
    if hist_length_list[i] > 0:
        hist_features[i, :hist_length_list[i]] = hist_features_extract_layer(hist_feature_list[i])
    else:
        hist_features[i, 0] = no_hist_embedding

print(hist_features[:2])
print(hist_features.size())

hist_length = torch.LongTensor([hist_length_list, ]).reshape((-1, 1)) # shape -> (batchsize, 1)

# TODO:
user_feat_dict = {'sparse': {
    'gender': 3,
    'visit_freq': 5,
    'binary_A': 2,
    'binary_B': 2,
    'binary_C': 2,
    'binary_D': 2,
    'binary_E': 2,
    'category_A': 100,
    'category_B': 100,
    'category_C': 100,
    'category_D': 100,
    'category_E': 100,
},
    'dense': {
        'salt_value': 1,
        'follow_topics': wv_size,
        'interest_topics': wv_size,
    }
}
user_features = {
    'sparse': {
        'gender': torch.randint(0, 3, (batchsize, )),
        'visit_freq': torch.randint(0, 5, (batchsize, )),
        'binary_A': torch.randint(0, 2, (batchsize, )),
        'binary_B': torch.randint(0, 2, (batchsize, )),
        'binary_C': torch.randint(0, 2, (batchsize, )),
        'binary_D': torch.randint(0, 2, (batchsize, )),
        'binary_E': torch.randint(0, 2, (batchsize, )),
        'category_A': torch.randint(0, 100, (batchsize, )),
        'category_B': torch.randint(0, 100, (batchsize, )),
        'category_C': torch.randint(0, 100, (batchsize, )),
        'category_D': torch.randint(0, 100, (batchsize, )),
        'category_E': torch.randint(0, 100, (batchsize, )),
    },
    'dense': {
        'salt_value': torch.randn((batchsize, 1)),
        'follow_topics': torch.randn((batchsize, 128)),
        'interest_topics': torch.randn((batchsize, 128)),
    }
}
user_feature_extract_layer= EmbeddingMLPLayer(user_feat_dict, 32, user_profile_dim)
user_features_embed = user_feature_extract_layer(user_features)

DINModel = DIN(query_dim, hist_embed_dim, user_profile_dim, hidden_dim_list = [512, 1])
out = DINModel(query_features_embed, hist_features, hist_length, user_features_embed)

print(out)
