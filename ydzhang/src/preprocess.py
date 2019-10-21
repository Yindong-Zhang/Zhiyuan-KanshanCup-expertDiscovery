# TODO: max pooling topics
# TODO: change time unit to day
import pandas as pd
import numpy as np
import os
dataDir = '../data'

# invite_df = pd.read_csv(os.path.join(dataDir, 'invite_info_1020.csv'),
#                                 names= ['question_id', 'user_id', 'create_time', 'isAnswer'])
user_df = pd.read_csv(os.path.join(dataDir, 'member_info_1020.csv'),
                              names= ['user_id', 'gender', 'visit_freq', 'binary_A',
                                     'binary_B', 'binary_C', 'binary_D', 'binary_E',
                                     'category_A' 'category_B', 'category_C',
                                     'category_D', 'category_E', 'salt', 'follow_topics', 'interest_topics'],
                              index_col= 'user_id')
topic_ids = pd.read_csv(os.path.join(dataDir, 'topic_vectors_64d.txt'),
                        names = ['topic_id', 'vector'],
                        usecols = ['topic_id', ],
                        index_col= 'topic_id',
                        sep = '\t',
                        squeeze= True)
topic_ids['index'] = np.arange(len(topic_ids))
topic_vec = np.load(os.path.join(dataDir, 'topic_array.npy'))
def map_topics(s):
    id_list = s.split(',')
    ind_list = topic_ids[id_list]
    topic_array = topic_vec[ind_list]
    topic_mp = np.max(topic_array, axis= 0)
    return topic_mp
user_df['follow_topics_mp'] = user_df['follow_topics'].apply(map_topics)
