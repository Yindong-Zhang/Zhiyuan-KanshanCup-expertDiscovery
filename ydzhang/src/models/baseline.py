from torch import nn
import torch
import torch.functional as F
from src.featureExtractLayer import EmbeddingMLPLayer
from src.layers import BiLSTMRCNN
from src.DIN import DIN
import numpy as np
dataDir = './test'
word_embedding = np.random.randn(1024, 128)

# baseline using pooling layer to extract semantic info rather than RCNN
wv_size = 128
batchsize = 128
hist_length = 10
# TODO:
query_features = {'sparse': {
    'has_describe': torch.randint(0, 2, (batchsize, ))
    },
    'dense':
        {'topic_feat': torch.randn((batchsize, wv_size)),
         'title_feat': torch.randn((batchsize, wv_size)),
         'describe_length': torch.randint((0, 256, (batchsize, 1))),
         'title_length': torch.randint(0, 64, (batchsize, 1))}
    }
query_features = EmbeddingMLPLayer({'sparse': {'has_describe': 2}, 'dense': ['topics_feat', 'title_feat', 'describe_length', 'title_lenght']},
                                   embedding_size= 32, output_dim= 256)(query_features)
answer_feat_dict = {'sparse':{
    'is_good': 2,
    'is_recommend': 2,
    'has_picture': 2,
    'has_video': 2,
    },
    'dense': ['word_count', 'num_zan', 'num_cancel_zan',
              'num_comment', 'num_collect', 'num_thanks',
              'num_report', 'num_useless', 'num_oppose']
    }
answer_features = {'sparse': {
    'is_good': torch.randint(0, 2, (batchsize, )),
    'is_recomment': torch.randint(0, 2, (batchsize, )),
    'has_picture': torch.randint(0, 2, (batchsize, )),
    'has_video': torch.randint(0, 2, (batchsize, )),
},
    'dense': {
        'word_count': torch.randint(0, 256, (batchsize, 1)),
        'num_zan': torch.randint(0, 2046, (batchsize, 1)),
        'num_report': torch.randint(0, 2046, (batchsize, 1)),
        'num_cancel_zan': torch.randint(0, 2046, (batchsize, 1)),
        'num_comment': torch.randint(0, 2046, (batchsize, 1)),
        'num_collect': torch.randint(0, 2046, (batchsize, 1)),
        'num_thanks': torch.randint(0, 2046, (batchsize, 1)),
        'num_report': torch.randint(0, 2046, (batchsize, 1)),
        'num_useless': torch.randint(0, 2046, (batchsize, 1)),
        'num_oppose': torch.randint(0, 2046, (batchsize, 1)),
        }
}

# history input should be a list of list
for i in range(batchsize):
    # TODO:
    pass