import os
import pandas as pd
import numpy as np
import random
from math import ceil
import torch
from src.config import WVSIZE
def create_feat_dict(feat_dim_dict, batch_feat_df, array_dict):
    """

    :param feat_dim_dict: {'dense': {'feat name': 1, 'feat_name': 128}, 'sparse': {'feat name': 4, 'feat name: 12}}
    :param batch_feat_df
    :param array_dict: dictionary of array for vector features
    :return: a feature column dictionary replace dimension with feature column
    """
    sparse_feat_dict = {feat_name: torch.LongTensor(batch_feat_df[feat_name].values) for feat_name in feat_dim_dict['sparse']}
    dense_feat_dict = {}
    for feat_name, feat_dim in feat_dim_dict['dense'].items():
        if feat_dim == 1:
            dense_feat_dict[feat_name] = torch.FloatTensor(batch_feat_df[feat_name].values).reshape(-1, 1)
        elif feat_dim > 1:
            dense_feat_inds = batch_feat_df[feat_name]
            dense_feat_dict[feat_name] = torch.FloatTensor(array_dict[feat_name][dense_feat_inds, :])
        else:
            raise Exception("illegal dimension")
    return {'dense': dense_feat_dict, 'sparse': sparse_feat_dict}


class Dataset():
    def __init__(self, invite_df, user_df, user_array_dict,  quest_df, quest_array_dict, ans_df, user_hist_dict,
                 batchsize,
                 question_feat_dict,
                 user_feat_dict,
                 max_hist_len,
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
        self.user_df = user_df
        self.user_array_dict = user_array_dict
        self.quest_df = quest_df
        self.quest_array_dict = quest_array_dict        # self.quest_title_array = np.load(os.path.join(self.dataDir, 'question_title_W.npy'))
        self.answer_df = ans_df
        # 利用user id 建立多重索引
        self.user_has_history_set = set(self.answer_df['user_id'].unique())
        self.history_dict= user_hist_dict
        self.batchsize = batchsize
        self.quest_feat_dict = question_feat_dict
        self.user_feat_dict = user_feat_dict

        self.max_hist_len = max_hist_len

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
            quest_ids, user_ids, invite_times = batch_invite_df['question_id'], batch_invite_df['user_id'], batch_invite_df['create_day']
            quest_feats = create_feat_dict(self.quest_feat_dict, quest_ids, self.quest_df, self.quest_array_dict)
            user_feats = create_feat_dict(self.user_feat_dict, user_ids, self.user_df, self.user_array_dict)

            hist_topics_array = torch.zeros((self.batchsize, self.max_hist_len, WVSIZE))
            hist_lens = []
            # 注意对没有历史记录的用户的处理
            for i, (user_id, invite_time) in enumerate(zip(user_ids, invite_times)):
                if user_id in self.user_has_history_set:
                    full_history = self.history_dict.loc[user_id]
                    hist_df = full_history.loc[full_history['create_day'] <= invite_time]
                    hist_df = hist_df.iloc[-self.max_hist_len:]
                    hist_len = len(hist_df)
                    hist_lens.append(hist_len)
                    if hist_len > 0:
                        hist_qids = hist_df['question_id']
                        quest_topic_inds = self.quest_df.loc[hist_qids, 'question_topics_mp']
                        hist_topics_array[i, :hist_len] = torch.FloatTensor(self.quest_array_dict['question_topics_mp'][quest_topic_inds, :])
                else:
                    hist_lens.append(0)

            hist_lens = torch.LongTensor(hist_lens).reshape(-1, 1)

            outputs = [quest_feats, user_feats, hist_topics_array, hist_lens]
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
    user_interest_topics_wp = np.load(os.path.join(dataDir, 'interest_topic_wp.npy'))
    user_array_dict = {'follow_topics_mp': user_follow_topics_mp,
                            'interest_topics_wp': user_interest_topics_wp
                            }
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
    test_df = pd.read_csv(os.path.join(dataDir, 'invite_info_evaluate_1021.csv'),
                          usecols= ['question_id', 'user_id', 'create_day'],
                          sep= '\t')
    print('Load data complete.')
    train_invite_df = invite_df.loc[np.logical_and(invite_df['create_day'] >= train_day_range[0], invite_df['create_day'] < train_day_range[1])]
    val_invite_df = invite_df.loc[np.logical_and(invite_df['create_day'] >= val_day_range[0], invite_df['create_day'] < val_day_range[1])]
    train_dataset = Dataset(train_invite_df, user_df, user_array_dict, quest_df, quest_array_dict, answer_df, user_hist_dict,
                            batchsize,
                            quest_dim_dict,
                            user_dim_dict,
                            max_hist_len,
                            )
    val_dataset = Dataset(val_invite_df, user_df, user_array_dict, quest_df, quest_array_dict, answer_df, user_hist_dict,
                          batchsize,
                          quest_dim_dict,
                          user_dim_dict,
                          max_hist_len,
                          )
    test_dataset = Dataset(test_df, user_df, user_array_dict, quest_df, quest_array_dict, answer_df, user_hist_dict,
                           batchsize,
                           quest_dim_dict,
                           user_dim_dict,
                           max_hist_len,
                           return_target= False,
                           shuffle= False,
                           )
    return train_dataset, val_dataset, test_dataset



if __name__ == '__main__':
    wv_size = 64
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
    dataset = Dataset('../data',
                      batchsize= 32,
                      question_feat_dict= query_feat_dict,
                      user_feat_dict= user_feat_dict,
                      answer_feat_dict= history_feat_dict)
    for i, batch in enumerate(dataset):
        print(batch)
        if i > 20:
            break;

