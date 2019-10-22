import os
import pandas as pd
import numpy as np
import random
from math import ceil
import torch

def create_feat_dict(feat_dim_dict, inds, feat_df, array_dict):
    """

    :param feat_dim_dict: {'dense': {'feat name': 1, 'feat_name': 128}, 'sparse': {'feat name': 4, 'feat name: 12}}
    :param feat_df
    :param array_dict: dictionary of array for vector features
    :return: a feature column dictionary replace dimension with feature column
    """
    sparse_feat_dict = {feat_name: torch.LongTensor(feat_df.loc[inds, feat_name]) for feat_name in feat_dim_dict['sparse']}
    dense_feat_dict = {}
    for feat_name, feat_dim in feat_dim_dict['dense'].items():
        if feat_dim == 1:
            dense_feat_dict[feat_name] = torch.FloatTensor(feat_df.loc[inds, feat_name])
        elif feat_dim > 1:
            dense_feat_inds = feat_df.loc[inds, feat_name]
            dense_feat_dict[feat_name] = torch.FloatTensor(array_dict[feat_name][dense_feat_inds, :])
        else:
            raise Exception("illegal dimension")
    return {'dense': dense_feat_dict, 'sparse': sparse_feat_dict}


class Dataset():
    def __init__(self, dataDir, batchsize, question_feat_dict, user_feat_dict, answer_feat_dict):
        self.dataDir = dataDir
        self.invite_df = pd.read_csv(os.path.join(self.dataDir, 'invite_info_1021.csv'),
                                usecols= ['question_id', 'user_id', 'create_day', 'is_answer'],
                                     sep= '\t',
                                     # nrows= 10000,
                                     )
        self.user_df = pd.read_csv(os.path.join(self.dataDir, 'member_info_1021.csv'),
                              # usecols= ['user_id', 'gender', 'visit_freq', 'binary_A',
                              #        'binary_B', 'binary_C', 'binary_D', 'binary_E',
                              #        'category_A',
                              #        'category_B',
                              #        'category_C',
                              #        'category_D', 'category_E', 'salt_value', 'follow_topics_mp', 'interest_topics'],
                              index_col= 'user_id',
                                   sep= '\t',
                                   # nrows= 10000,
                                   )
        user_follow_topics_mp = np.load(os.path.join(self.dataDir, 'user_follow_topics_mp.npy'))
        self.user_array_dict = {'follow_topics_mp': user_follow_topics_mp}
        self.quest_df = pd.read_csv(os.path.join(self.dataDir, 'question_info_1021.csv'),
                                    # usecols= ['question_id', 'title_SW', 'title_W', 'question_topics_mp', 'title_W_ind', 'create_day',
                                    #           'has_describe', 'describe_length'],
                                    index_col= 'question_id',
                                    sep= '\t',
                                    # nrows= 10000
                                    )
        self.quest_array_dict = {'topics_mp': np.load(os.path.join(self.dataDir, 'question_topics_mp.npy'))}
        self.quest_title_array = np.load(os.path.join(self.dataDir, 'question_title_W.npy'))
        self.answer_df = pd.read_csv(os.path.join(self.dataDir, 'answer_info_1021.csv'),
                                # usecols = ['answer_id', 'question_id', 'user_id', 'create_day',
                                #          'answer_SW', 'answer_W', 'is_good', 'has_picture', 'has_video',
                                #          'word_count', 'num_zan', 'num_cancel_zan', 'num_comment', 'num_collect',
                                #          'num_thanks', 'num_report', 'num_useless', 'num_oppose', 'question_topics_mp'],
                                index_col= ['user_id', 'answer_id'],
                                     sep= '\t',
                                     # nrows= 10000,
                                     )
        user_ids = self.invite_df['user_id'].unique()
        #TODO: 利用user id 建立多重索引
        self.history_dict = {user_id: self.answer_df.loc[self.answer_df['user_id'] == user_id, ['question_id', 'answer_id', 'create_day']] for user_id in user_ids}
        self.n_samples = len(self.invite_df)
        self.batchsize = batchsize
        self.quest_feat_dict = question_feat_dict
        self.user_feat_dict = user_feat_dict
        self.answer_feat_dict = answer_feat_dict

    def __iter__(self):
        inds_list = list(range(self.n_samples))
        random.shuffle(inds_list)
        n_batches = ceil(self.n_samples / self.batchsize)
        for i in range(n_batches):
            batch_inds = inds_list[i * self.batchsize : min((i + 1 ) * self.batchsize, self.n_samples)]
            batch_invite_df= self.invite_df.loc[batch_inds, :]
            quest_ids, user_ids, invite_times, is_answer = batch_invite_df['question_id'], batch_invite_df['user_id'], batch_invite_df['create_day'], batch_invite_df['is_answer']
            quest_feats = create_feat_dict(self.quest_feat_dict, quest_ids, self.quest_df, self.quest_array_dict)

            quest_titles = torch.LongTensor(self.quest_title_array(self.quest_df.loc[quest_ids, 'title_W_ind']))

            user_feats = create_feat_dict(self.user_feat_dict, user_ids, self.user_df, self.user_array_dict)

            hist_feat_list = []
            hist_len = []
            for user_id, invite_times in zip(user_ids, invite_times):
                full_history = self.history_dict[user_id]
                hist_df = full_history.loc[hist_df['create_time'] < invite_times]
                hist_len.append(len(hist_df))
                hist_qids = hist_df['question_ids']
                hist_aids = hist_df['answer_ids']
                hist_quest_feats = torch.LongTensor(self.quest_title_array(self.quest_df.loc[hist_qids, 'title_W_ind']))
                hist_ans_feats = create_feat_dict(self.answer_feat_dict, hist_aids, self.answer_df, self.quest_array_dict)
                hist_feat_list.append((hist_quest_feats, hist_ans_feats))

            hist_len = torch.LongTensor(hist_len).reshape(-1, 1)

            yield quest_titles, quest_feats, hist_feat_list, hist_len, user_feats,


if __name__ == '__main__':
    wv_size = 64
    query_feat_dict = {'sparse': {'has_describe': 2},
                       'dense': {'topics_mp': wv_size,
                                 'describe_length': 1,
                                 # 'title_length': 1,
                                 # 'num_answers': 1,
                                 }
                       }
    history_feat_dict = {'sparse': {
        'is_good': 2,
        'is_recommend': 2,
        'has_picture': 2,
        'has_video': 2,
    },
        'dense': {'topic_mp': wv_size,
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
            'follow_topics_mp': wv_size,
        }
    }
    dataset = Dataset('../data',
                      batchsize= 32,
                      question_feat_dict= query_feat_dict,
                      user_feat_dict= user_feat_dict,
                      answer_feat_dict= history_feat_dict)
    for i, batch in enumerate(dataset):
        print(batch)
        if i > 20:
            break;

