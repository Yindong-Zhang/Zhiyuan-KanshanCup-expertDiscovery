from src.datasets import create_feat_dict
import os
import pandas as pd
import numpy as np
import random
from math import ceil
from time import time
import torch
import torch.nn.functional as F
from src.config import WVSIZE

class Dataset():
    def __init__(self, invite_df, hist_quest_array, user_df, user_array_dict,  quest_df, quest_array_dict,
                 batchsize,
                 question_feat_dict,
                 user_feat_dict,
                 context_feat_dict,
                 shuffle= True,
                 return_target= True):
        """

        :param dataDir:
        :param batchsize:
        :param question_feat_dict:
        :param user_feat_dict:
        :param answer_feat_dict:
        :param max_hist_len:
        :param day_range: a tuple of [day_first, day_last) to construct dataset.
        :param return_target: whether to return target
        """
        self.invite_df = invite_df
        self.hist_quest_array = hist_quest_array
        self.user_df = user_df
        self.user_array_dict = user_array_dict
        self.quest_df = quest_df
        self.quest_array_dict = quest_array_dict
        # self.quest_title_array = np.load(os.path.join(self.dataDir, 'question_title_W.npy'))
        self.batchsize = batchsize
        self.quest_feat_dict = question_feat_dict
        self.user_feat_dict = user_feat_dict
        self.context_feat_dict = context_feat_dict
        self.inds_list = np.arange(len(self.invite_df)).tolist()
        self.n_samples = len(self.invite_df)
        self.num_batches = ceil(self.n_samples / self.batchsize)
        self.return_target = return_target
        self.shuffle = shuffle
        if self.shuffle:
            random.shuffle(self.inds_list)

    def __iter__(self):
        n_batches = self.num_batches
        for i in range(n_batches):
            batch_inds = self.inds_list[i * self.batchsize : min((i + 1 ) * self.batchsize, self.n_samples)]
            batch_invite_df= self.invite_df.iloc[batch_inds, :]
            batch_hist_quest_idx = self.hist_quest_array[batch_inds, :]
            batch_hist_quest_tp = F.embedding(batch_hist_quest_idx, weight= self.quest_array_dict['question_topics_mp'],
                                              padding_idx= -1)
            quest_ids, user_ids, invite_times = batch_invite_df['question_id'], batch_invite_df['user_id'], batch_invite_df['create_day']
            quest_feats = create_feat_dict(self.quest_feat_dict, self.quest_df.loc[quest_ids], self.quest_array_dict)
            user_feats = create_feat_dict(self.user_feat_dict, self.user_df.loc[user_ids], self.user_array_dict)

            context_feats = create_feat_dict(self.context_feat_dict, batch_invite_df, None)

            outputs = [quest_feats, user_feats, context_feats]
            if self.return_target:
                is_answer =  batch_invite_df['is_answer']
                target = torch.FloatTensor(is_answer.values).reshape(-1, 1)
                outputs.append(target)

            yield outputs

    def __len__(self):
        return self.num_batches


def create_train_val_test_dataset(dataDir,
                                  batchsize,
                                  quest_dim_dict,
                                  user_dim_dict,
                                  context_dim_dict,
                                  max_hist_len,
                                  train_day_range,
                                  val_day_range):
    print('loading data...')
    invite_df = pd.read_csv(os.path.join(dataDir, 'invite_info_1107.csv'),
                                 # usecols=['question_id', 'user_id', 'create_day', 'is_answer'],
                                 sep='\t',
                                 # nrows= 10000,
                            index_col= 0,
                                 )
    hist_quest_array = np.load(os.path.join(dataDir, 'train_hist_quest_array.npy'))[:, :max_hist_len]
    user_df = pd.read_csv(os.path.join(dataDir, 'member_info_1106.csv'),
                               # usecols= ['user_id', 'gender', 'visit_freq', 'binary_A',
                               #        'binary_B', 'binary_C', 'binary_D', 'binary_E',
                               #        'category_A',
                               #        'category_B',
                               #        'category_C',
                               #        'category_D', 'category_E', 'salt_value', 'follow_topics_mp', 'interest_topics'],
                               index_col='user_id',
                               sep='\t',
                               # nrows= 10000,
                               )
    user_follow_topics_mp = np.load(os.path.join(dataDir, 'user_follow_topics_mp.npy'))
    user_interest_topics_wp = np.load(os.path.join(dataDir, 'interest_topic_wp.npy'))
    user_array_dict = {'follow_topics_mp': user_follow_topics_mp,
                            'interest_topics_wp': user_interest_topics_wp
                            }
    quest_df = pd.read_csv(os.path.join(dataDir, 'question_info_1106.csv'),
                                # usecols= ['question_id', 'title_SW', 'title_W', 'question_topics_mp', 'title_W_ind', 'create_day',
                                #           'has_describe', 'describe_length'],
                                index_col='question_id',
                                sep='\t',
                                # nrows= 10000
                                )
    question_topics_mp = np.load(os.path.join(dataDir, 'question_topics_mp.npy'))
    quest_array_dict = {'question_topics_mp': question_topics_mp}
    test_df = pd.read_csv(os.path.join(dataDir, 'test_invite_info_1107.csv'),
                          # usecols= ['question_id', 'user_id', 'create_day'],
                          sep= '\t')
    test_hist_quest_array = np.load(os.path.join(dataDir, 'test_hist_quest_array.npy'))[:, :max_hist_len]
    print('Load data complete.')
    train_idx = np.logical_and(invite_df['create_day'] >= train_day_range[0], invite_df['create_day'] < train_day_range[1])
    val_idx = np.logical_and(invite_df['create_day'] >= val_day_range[0], invite_df['create_day'] < val_day_range[1])
    train_invite_df = invite_df.loc[train_idx]
    val_invite_df = invite_df.loc[val_idx]
    train_hist_quest_array = hist_quest_array[train_idx]
    val_hist_quest_array = hist_quest_array[val_idx]
    train_dataset = Dataset(train_invite_df, train_hist_quest_array, user_df, user_array_dict, quest_df, quest_array_dict,
                            batchsize,
                            quest_dim_dict,
                            user_dim_dict,
                            context_dim_dict,
                            )
    val_dataset = Dataset(val_invite_df, val_hist_quest_array, user_df, user_array_dict, quest_df, quest_array_dict,
                          batchsize,
                          quest_dim_dict,
                          user_dim_dict,
                          context_dim_dict,
                          )
    test_dataset = Dataset(test_df, test_hist_quest_array, user_df, user_array_dict, quest_df, quest_array_dict,
                           batchsize,
                           quest_dim_dict,
                           user_dim_dict,
                           context_dim_dict,
                           return_target= False,
                           shuffle= False,
                           )
    return train_dataset, val_dataset, test_dataset


if __name__ == '__main__':
    wv_size = WVSIZE
    max_hist_len = 16
    query_feat_dict = {'sparse': {'has_describe': 2},
                       'dense': {'question_topics_mp': wv_size,
                                 'describe_W_length': 1,
                                 'title_W_length': 1,
                                 }
                       }
    # history_feat_dict = {'sparse': {
    #     'is_good': 2,
    #     'is_recommend': 2,
    # 'has_picture': 2,
    # 'has_video': 2,
    # },
    #     'dense': {'question_topics_mp': wv_size,
    #               'word_count': 1,
    #               'num_zan': 1,
    #               'num_comment': 1,
    #               'num_collect': 1,
    #               'num_thanks': 1,
    # 'num_report': 1,
    #              'num_useless': 1,
    #               'num_oppose': 1
    #               }
    # }
    user_feat_dict = {'sparse': {
        'gender': 3,
        'visit_freq': 5,
        'binary_A': 2,
        'binary_B': 2,
        'binary_C': 2,
        'binary_D': 2,
        'binary_E': 2,
        'category_A': 150,
        'category_B': 150,
        'category_C': 150,
        'category_D': 150,
        'category_E': 2,
    },
        'dense': {
            'salt_value': 1,
            'follow_topics_mp': wv_size,
            'interest_topics_wp': wv_size,
        }
    }
    context_feat_dict = {
        'sparse': {
            'create_hour': 25,
        },
        'dense': {
            # 'days_since_last_ans': 1,
            # 'days_since_last_ans_scaled': 1,
        }
    }
    # dataset = Dataset('../../data',
    #                   batchsize= 32,
    #                   question_feat_dict= query_feat_dict,
    #                   user_feat_dict= user_feat_dict,
    #                   answer_feat_dict= history_feat_dict,
    #                   max_hist_len = max_hist_len,
    #                   )

    train_dataset, val_dataset, test_dataset = create_train_val_test_dataset('../../data',
                                                                             batchsize= 256,
                                                                             quest_dim_dict=query_feat_dict,
                                                                             user_dim_dict=user_feat_dict,
                                                                             context_dim_dict= context_feat_dict,
                                                                             max_hist_len= 16,
                                                                             train_day_range=[3838+ 20, 3838 + 25],
                                                                             val_day_range=[3838 + 25, 3838 + 30],
                                                                             )
    t0 = time()
    for i, batch in enumerate(train_dataset):

        print(i)
        if i > 20:
            break
    t1 = time()
    print('mean iteration time: %.4f' %((t1 - t0)/ 20, ))

    t2 = time()
    for i, batch in enumerate(val_dataset):
        print(i)
        if i > 20:
            break
    t3 = time()
    print('mean iteration time: %.4f' %((t3 - t2)/ 20, ))

    t4 = time()
    for i, batch in enumerate(test_dataset):
        print(i)
        if i > 20:
            break
    t5 = time()
    print('mean iteration time: %.4f' %((t5 - t4)/ 20, ))
