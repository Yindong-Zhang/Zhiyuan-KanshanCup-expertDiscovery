from src.dataset import create_feat_dict
import os
import pandas as pd
import numpy as np
import random
from math import ceil
import torch

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
        self.quest_array_dict = {'question_topics_mp': np.load(os.path.join(self.dataDir, 'question_topics_mp.npy'))}
        self.quest_title_array = np.load(os.path.join(self.dataDir, 'question_title_W.npy'))
        self.answer_df = pd.read_csv(os.path.join(self.dataDir, 'answer_info_1021.csv'),
                                # usecols = ['answer_id', 'question_id', 'user_id', 'create_day',
                                #          'answer_SW', 'answer_W', 'is_good', 'has_picture', 'has_video',
                                #          'word_count', 'num_zan', 'num_cancel_zan', 'num_comment', 'num_collect',
                                #          'num_thanks', 'num_report', 'num_useless', 'num_oppose', 'question_topics_mp'],
                                index_col= ['answer_id'],
                                     sep= '\t',
                                     # nrows= 10000,
                                     )
        # 利用user id 建立多重索引
        # self.history_dict = {user_id: self.answer_df.loc[self.answer_df['user_id'] == user_id, ['question_id', 'answer_id', 'create_day']] for user_id in user_ids}
        # self.history_dict = pd.read_csv(os.path.join(self.dataDir, 'answer_info_1021.csv'),
        #                                 usecols=['answer_id', 'question_id', 'user_id', 'create_day'],
        #                                 index_col=['user_id', 'answer_id'],
        #                                 sep= '\t')
        self.user_has_history_set = set(self.answer_df['user_id'].unique())
        self.history_dict= self.answer_df[['question_id', 'user_id', 'create_day']].set_index(['user_id', self.answer_df.index])
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

            quest_titles = torch.LongTensor(self.quest_title_array[self.quest_df.loc[quest_ids, 'title_W_ind']])

            user_feats = create_feat_dict(self.user_feat_dict, user_ids, self.user_df, self.user_array_dict)

            hist_feat_list = []
            hist_len = []
            # 注意对没有历史记录的用户的处理
            for user_id, invite_time in zip(user_ids, invite_times):
                if user_id in self.user_has_history_set:
                    full_history = self.history_dict.loc[user_id]
                    hist_df = full_history.loc[full_history['create_day'] <= invite_time]
                    hist_len.append(len(hist_df))
                    if len(hist_df) > 0:
                        hist_qids = hist_df['question_id']
                        hist_aids = hist_df.index # answer_id
                        hist_quest_feats = torch.LongTensor(self.quest_title_array[self.quest_df.loc[hist_qids, 'title_W_ind']])
                        hist_ans_feats = create_feat_dict(self.answer_feat_dict, hist_aids, self.answer_df, self.quest_array_dict)
                        hist_feat_list.append((hist_quest_feats, hist_ans_feats))
                    else:
                        hist_feat_list.append((None, None))
                else:
                    hist_len.append(0)
                    hist_feat_list.append((None, None))

            hist_len = torch.LongTensor(hist_len).reshape(-1, 1)

            yield quest_titles, quest_feats, hist_feat_list, hist_len, user_feats,