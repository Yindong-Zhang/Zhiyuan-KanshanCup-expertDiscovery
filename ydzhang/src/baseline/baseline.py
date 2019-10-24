from torch import nn
import torch
import torch.functional as F
from src.featureExtractLayer import EmbeddingMLPLayer
from src.layers import BiLSTMRCNN
from src.DIN import DIN
from torch.nn.utils.rnn import pad_sequence
import torch.optim as optim

import numpy as np

# baseline using pooling layer to extract semantic info rather than RCNN

class Model(nn.Module):
    def __init__(self, query_feat_dict, history_feat_dict, user_feat_dict,
                 query_embed_dim, hist_embed_dim, user_profile_dim,
                 max_hist_len, embed_size,
                 hidden_dim_list,
                 device, **kwargs):
        super(Model, self).__init__(**kwargs)
        self.hist_embed_dim = hist_embed_dim
        self.user_profile_dim = user_profile_dim
        self.query_embed_dim = query_embed_dim
        self.max_hist_len = max_hist_len
        self.embed_size = embed_size
        self.device = device
        self.query_features_extract_layer = EmbeddingMLPLayer(query_feat_dict,
                                                              embedding_size= self.embed_size, mlp_hidden_list= [self.query_embed_dim, ])
        self.hist_features_extract_layer = EmbeddingMLPLayer(history_feat_dict, embedding_size= self.embed_size, mlp_hidden_list= [self.hist_embed_dim, ])
        self.user_feature_extract_layer= EmbeddingMLPLayer(user_feat_dict, embedding_size= self.embed_size, mlp_hidden_list= [self.user_profile_dim, ])
        self.interaction_layer = DIN(query_embed_dim, hist_embed_dim, user_profile_dim, hidden_dim_list = hidden_dim_list, use_sigmoid= True)

    def forward(self, query_features, hist_features_list, hist_length, user_features):
        query_feat_embed = self.query_features_extract_layer(query_features)
        batchsize = len(hist_features_list)

        # hist_feat_embed_list = []
        # for i in range(batchsize):
        #     hist_feat_embed_list.append(self.hist_features_extract_layer(hist_features_list[i]))

        # TODO: debug
        hist_feat_embed = torch.zeros((batchsize, self.max_hist_len, self.hist_embed_dim)).to(self.device)
        for i ,(length,  answer_features) in enumerate(zip(hist_length, hist_features_list)):
            if length > 0:
                hist_feat_embed[i, :length] = self.hist_features_extract_layer(answer_features)
        print(hist_feat_embed.shape)

        user_feat_embed = self.user_feature_extract_layer(user_features)

        out = self.interaction_layer(query_feat_embed, hist_feat_embed, hist_length, user_feat_embed)
        return out

if __name__ == '__main__':
    wv_size = 128
    batchsize = 32
    max_hist_length = 20
    hist_embed_dim = 128
    query_embed_dim = 128
    user_profile_dim = 512


    def create_dataset(batchsize, max_hist_length, wv_size, ):
        query_features = {'sparse': {
            'has_describe': torch.randint(0, 2, (batchsize,))
        },
            'dense':
                {'topic_feat': torch.randn((batchsize, wv_size)),
                 'title_feat': torch.randn((batchsize, wv_size)),
                 'describe_length': torch.randint(0, 256, (batchsize, 1)).float(),
                 'title_length': torch.randint(0, 64, (batchsize, 1)).float()}
        }
        query_feat_dict = {'sparse': {'has_describe': 2}, 'dense': {'topics_feat': wv_size,
                                                                    'title_feat': wv_size,
                                                                    'describe_length': 1,
                                                                    'title_lenght': 1}}

        history_feat_dict = {'sparse': {
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
                      'num_report': 1,
                      'num_useless': 1,
                      'num_oppose': 1}
        }

        def fake_history_features(hist_length):
            answer_features = {'sparse': {
                'is_good': torch.randint(0, 2, (hist_length,)),
                'is_recommend': torch.randint(0, 2, (hist_length,)),
                'has_picture': torch.randint(0, 2, (hist_length,)),
                'has_video': torch.randint(0, 2, (hist_length,)),
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
                'gender': torch.randint(0, 3, (batchsize,)),
                'visit_freq': torch.randint(0, 5, (batchsize,)),
                'binary_A': torch.randint(0, 2, (batchsize,)),
                'binary_B': torch.randint(0, 2, (batchsize,)),
                'binary_C': torch.randint(0, 2, (batchsize,)),
                'binary_D': torch.randint(0, 2, (batchsize,)),
                'binary_E': torch.randint(0, 2, (batchsize,)),
                'category_A': torch.randint(0, 100, (batchsize,)),
                'category_B': torch.randint(0, 100, (batchsize,)),
                'category_C': torch.randint(0, 100, (batchsize,)),
                'category_D': torch.randint(0, 100, (batchsize,)),
                'category_E': torch.randint(0, 100, (batchsize,)),
            },
            'dense': {
                'salt_value': torch.randn((batchsize, 1)),
                'follow_topics': torch.randn((batchsize, 128)),
                'interest_topics': torch.randn((batchsize, 128)),
            }
        }
        # history input should be a list of list
        hist_length_list = [np.random.randint(0, max_hist_length + 1) for _ in range(batchsize)]
        hist_length = torch.LongTensor([hist_length_list, ]).reshape((-1, 1))  # shape -> (batchsize, 1)

        hist_features_list = [fake_history_features(hist_length_list[i]) for i in range(batchsize)]
        target = torch.randint(0, 2, (batchsize, 1)).float()
        return query_feat_dict, query_features, history_feat_dict, hist_length, hist_features_list, user_feat_dict, user_features, target


    query_feat_dict, query_features, history_feat_dict, hist_length, hist_features_list, user_feat_dict, user_features, target = create_dataset(batchsize, max_hist_length, wv_size)
    model = Model(query_feat_dict, history_feat_dict, user_feat_dict,
                  query_embed_dim, hist_embed_dim, user_profile_dim,
                  embed_size= 32,
                  max_hist_len = max_hist_length,
                  hidden_dim_list= [512, 1],
                  device= 'cpu')
    optimizer = optim.Adam(params= model.parameters(), lr = 1e-3)

    optimizer.zero_grad()
    predict = model(query_features, hist_features_list, hist_length, user_features)
    loss = nn.BCELoss()(predict, target)
    loss.backward()
    optimizer.step()
    print(predict)
