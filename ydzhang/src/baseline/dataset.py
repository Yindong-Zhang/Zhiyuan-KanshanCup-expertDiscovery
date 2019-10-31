from src.datasets import create_feat_dict
import os
import pandas as pd
import numpy as np
import random
from math import ceil
import torch

class Dataset():
    def __init__(self, invite_df, user_df, user_array_dict,  quest_df, quest_array_dict, ans_df, user_hist_dict,
                 batchsize,
                 question_feat_dict,
                 user_feat_dict,
                 answer_feat_dict,
                 max_hist_len,
                 day_range):
        """

        :param dataDir:
        :param batchsize:
        :param question_feat_dict:
        :param user_feat_dict:
        :param answer_feat_dict:
        :param max_hist_len:
        :param day_range: a tuple of [day_first, day_last) to construct dataset.
        """
        self.invite_df = invite_df
        self.user_df = user_df
        self.user_array_dict = user_array_dict
        self.quest_df = quest_df
        self.quest_array_dict = quest_array_dict
        # self.quest_title_array = np.load(os.path.join(self.dataDir, 'question_title_W.npy'))
        self.answer_df = ans_df
        # 利用user id 建立多重索引
        self.max_hist_len = max_hist_len
        self.user_has_history_set = set(self.answer_df['user_id'].unique())
        self.history_dict= user_hist_dict
        self.batchsize = batchsize
        self.quest_feat_dict = question_feat_dict
        self.user_feat_dict = user_feat_dict
        self.answer_feat_dict = answer_feat_dict

        self.day_range= day_range
        self.inds_list = np.arange(len(self.invite_df))[np.logical_and(self.invite_df['create_day'] >= self.day_range[0], self.invite_df['create_day'] < self.day_range[1])].tolist()
        self.n_samples = len(self.inds_list)
        self.num_batches = ceil(self.n_samples / self.batchsize)
        random.shuffle(self.inds_list)

    def __iter__(self):
        n_batches = self.num_batches
        for i in range(n_batches):
            batch_inds = self.inds_list[i * self.batchsize : min((i + 1 ) * self.batchsize, self.n_samples)]
            batch_invite_df= self.invite_df.loc[batch_inds, :]
            quest_ids, user_ids, invite_times, is_answer = batch_invite_df['question_id'], batch_invite_df['user_id'], batch_invite_df['create_day'], batch_invite_df['is_answer']
            quest_feats = create_feat_dict(self.quest_feat_dict, quest_ids, self.quest_df, self.quest_array_dict)

            user_feats = create_feat_dict(self.user_feat_dict, user_ids, self.user_df, self.user_array_dict)

            hist_feat_list = []
            hist_len = []
            # 注意对没有历史记录的用户的处理
            for user_id, invite_time in zip(user_ids, invite_times):
                if user_id in self.user_has_history_set:
                    full_history = self.history_dict.loc[user_id]
                    hist_df = full_history.loc[full_history['create_day'] <= invite_time]
                    hist_df = hist_df.iloc[-self.max_hist_len:,]
                    hist_len.append(len(hist_df))
                    if len(hist_df) > 0:
                        hist_qids = hist_df['question_id']
                        hist_aids = hist_df.index # answer_id
                        hist_ans_feats = create_feat_dict(self.answer_feat_dict, hist_aids, self.answer_df, self.quest_array_dict)
                        hist_feat_list.append(hist_ans_feats)
                    else:
                        hist_feat_list.append((None, None))
                else:
                    hist_len.append(0)
                    hist_feat_list.append((None, None))

            hist_len = torch.LongTensor(hist_len).reshape(-1, 1)

            target = torch.FloatTensor(is_answer.values)
            yield quest_feats, hist_feat_list, hist_len, user_feats, target

    def __len__(self):
        return self.num_batches


def create_train_val_dataset(dataDir,
                             batchsize,
                             question_feat_dict,
                             user_feat_dict,
                             answer_feat_dict,
                             max_hist_len,
                             train_day_range,
                             val_day_range):
    invite_df = pd.read_csv(os.path.join(dataDir, 'invite_info_1021.csv'),
                                 usecols=['question_id', 'user_id', 'create_day', 'is_answer'],
                                 sep='\t',
                                 # nrows= 10000,
                                 )
    user_df = pd.read_csv(os.path.join(dataDir, 'member_info_1021.csv'),
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
    user_array_dict = {'follow_topics_mp': user_follow_topics_mp}
    quest_df = pd.read_csv(os.path.join(dataDir, 'question_info_1021.csv'),
                                # usecols= ['question_id', 'title_SW', 'title_W', 'question_topics_mp', 'title_W_ind', 'create_day',
                                #           'has_describe', 'describe_length'],
                                index_col='question_id',
                                sep='\t',
                                # nrows= 10000
                                )
    quest_array_dict = {'question_topics_mp': np.load(os.path.join(dataDir, 'question_topics_mp.npy'))}
    # self.quest_title_array = np.load(os.path.join(self.dataDir, 'question_title_W.npy'))
    answer_df = pd.read_csv(os.path.join(dataDir, 'answer_info_1021.csv'),
                                 usecols = ['answer_id', 'question_id', 'user_id', 'create_day',
                                          #'answer_SW', 'answer_W',
                                            'is_good', 'has_picture', 'has_video',
                                          'word_count', 'num_zan', 'num_cancel_zan', 'num_comment', 'num_collect',
                                          'num_thanks', 'num_report', 'num_useless', 'num_oppose', 'question_topics_mp'],
                                 index_col=['answer_id'],
                                 sep='\t',
                                 # nrows= 10000,
                                 )
    user_hist_dict= answer_df[['question_id', 'user_id', 'create_day']].set_index(['user_id', answer_df.index])
    print('Load data complete.')
    train_dataset = Dataset(invite_df, user_df, user_array_dict,  quest_df, quest_array_dict, answer_df, user_hist_dict,
                 batchsize,
                 question_feat_dict,
                 user_feat_dict,
                 answer_feat_dict,
                 max_hist_len,
                            train_day_range)
    val_dataset = Dataset(invite_df, user_df, user_array_dict,  quest_df, quest_array_dict, answer_df, user_hist_dict,
                 batchsize,
                 question_feat_dict,
                 user_feat_dict,
                 answer_feat_dict,
                 max_hist_len,
                            val_day_range)
    return train_dataset, val_dataset


if __name__ == '__main__':
    wv_size = 64
    max_hist_len = 16
    query_feat_dict = {'sparse': {'has_describe': 2},
                       'dense': {'question_topics_mp': wv_size,
                                 'describe_length': 1,
                                 # 'title_length': 1,
                                 # 'num_answers': 1,
                                 }
                       }
    history_feat_dict = {'sparse': {
        'is_good': 2,
        # 'is_recommend': 2,
        'has_picture': 2,
        'has_video': 2,
    },
        'dense': {'question_topics_mp': wv_size,
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
    # dataset = Dataset('../../data',
    #                   batchsize= 32,
    #                   question_feat_dict= query_feat_dict,
    #                   user_feat_dict= user_feat_dict,
    #                   answer_feat_dict= history_feat_dict,
    #                   max_hist_len = max_hist_len,
    #                   )

    train_dataset, val_dataset = create_train_val_dataset('../../data',
                                                          batchsize=32,
                                                          question_feat_dict=query_feat_dict,
                                                          user_feat_dict=user_feat_dict,
                                                          answer_feat_dict=history_feat_dict,
                                                          max_hist_len=max_hist_len,
                                                          train_day_range=[3838+ 10, 3838 + 25],
                                                          val_day_range=[3838 + 25, 3838 + 30],
                                                          )
    for i, batch in enumerate(train_dataset):
        print(batch)
        if i > 20:
            break

    for i, batch in enumerate(val_dataset):
        print(batch)
        if i > 20:
            break