import os
import pandas as pd
import numpy as np
import random
from math import ceil
import torch

def create_feat_dict(feat_dim_dict, inds, feat_df):
    """

    :param feat_dim_dict: {'dense': {'feat name': 1, 'feat_name': 128}, 'sparse': {'feat name': 4, 'feat name: 12}}
    :param feat_df
    :return: a feature column dictionary replace dimension with feature column
    """
    return {'dense': {feat_name: torch.FloatTensor(feat_df.loc[inds, feat_name]) for feat_name in feat_dim_dict['dense']},
    'sparse': {feat_name: torch.LongTensor(feat_df.loc[inds, feat_name]) for feat_name in feat_dim_dict['sparse']}}


class Dataset():
    def __init__(self, dataDir, batchsize, question_feat_dict, user_feat_dict, answer_feat_dict):
        self.dataDir = dataDir
        self.invite_df = pd.read_csv(os.path.join(self.dataDir, 'invite_info.csv'),
                                names= ['question_id', 'user_id', 'create_time', 'isAnswer'])
        self.user_df = pd.read_csv(os.path.join(self.dataDir, 'user_info.csv'),
                              names= ['user_id', 'gender', 'visit_freq', 'binary_A',
                                     'binary_B', 'binary_C', 'binary_D', 'binary_E',
                                     'category_A' 'category_B', 'category_C',
                                     'category_D', 'category_E', 'salt', 'follow_topics', 'interest_topics'],
                              index_col= 'user_id')
        self.quest_df = pd.read_csv(os.path.join(self.dataDir, 'question_info.csv'),
                                    names = ['question_id', 'title_SW', 'title_W', 'question_topics', 'create_day'],
                                    index_col= 'question_id')
        # TODO: pad title sequence to fix length sequence
        self.quest_title_sr = self.quest_df['title_W']
        self.answer_df = pd.read_csv(os.path.join(self.dataDir, 'answer_info.csv'),
                                names = ['answer_id', 'question_id', 'user_id', 'create_time',
                                         'ans_SW', 'ans_W', 'is_good', 'has_picture', 'has_video',
                                         'word_count', 'num_zan', 'num_cancel', 'num_comment', 'num_collect',
                                         'num_thanks', 'num_report', 'num_useless', 'num_oppos'],
                                index_col= 'answer_id')
        user_ids = self.invite_df['user_id'].unique()
        self.history_dict = {user_id: self.answer_df.loc[self.answer_df['user_id'] == user_id, ['question_id', 'answer_id', 'create_time']] for user_id in user_ids}
        self.n_samples = len(self.invite_df)
        self.batchsize = batchsize

        self.quest_feat_dict = question_feat_dict
        self.user_feat_dict = user_feat_dict
        self.answer_feat_dict = answer_feat_dict

    def __iter__(self):
        inds_list = random.shuffle(range(self.n_samples))
        n_batches = ceil(self.n_samples / self.batchsize)
        for i in range(n_batches):
            batch_inds = inds_list[i * self.batchsize : min((i + 1 ) * self.batchsize, self.n_samples)]
            batch_invite_df= self.invite_df.loc[batch_inds, :]
            quest_ids, user_ids, invite_times, isAnswers = batch_invite_df['question_id'], batch_invite_df['user_id'], batch_invite_df['create_time'], batch_invite_df['isAnswer']
            quest_feats = create_feat_dict(self.quest_feat_dict, quest_ids, self.quest_df)

            quest_titles = torch.LongTensor(self.quest_title_sr.loc[quest_ids])

            user_feats = create_feat_dict(self.user_feat_dict, user_ids, self.user_df)

            hist_feat_list = []
            hist_len = []
            for user_id, invite_times in zip(user_ids, invite_times):
                full_history = self.history_dict[user_id]
                hist_df = full_history.loc[hist_df['create_time'] < invite_times]
                hist_len.append(len(hist_df))
                hist_qids = hist_df['question_ids']
                hist_aids = hist_df['answer_ids']
                hist_quest_feats = torch.LongTensor(self.quest_title_sr.loc[hist_qids])
                hist_ans_feats = create_feat_dict(self.answer_feat_dict, hist_aids, self.answer_df)
                hist_feat_list.append((hist_quest_feats, hist_ans_feats))

            hist_len = torch.LongTensor(hist_len)

            yield quest_titles, quest_feats, hist_feat_list, hist_len, user_feats,


