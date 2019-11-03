from src.baseline.dataset import create_train_val_test_dataset
from src.baseline.baseline import  Baseline
from time import time
from sklearn.metrics import roc_auc_score
import pandas as pd
import os
from src.config import DAYFIRST, PROJECTPATH
import torch
from torch import optim, nn
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--epoches', type= int, default= 1, help= 'training epoches')
parser.add_argument('--print_every', type= int, default = 32, help= 'print intervals.')
parser.add_argument('--patience', type= int, default= 4, help= 'patience epoche for early stopping')
args = parser.parse_args()

wv_size = 64
max_hist_len = 16
query_embed_dim = 256
hist_embed_dim= 128
user_embed_dim= 512
batchsize = 256
dataDir = os.path.join(PROJECTPATH, 'data')
configStr= 'test-baseline'

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
    }
}

chkpt_path =  os.path.join(PROJECTPATH, 'chkpt', configStr)
train_dataset, val_dataset, test_dataset = create_train_val_test_dataset(dataDir,
                                                           batchsize= batchsize,
                                                           quest_dim_dict= query_feat_dict,
                                                           user_dim_dict= user_feat_dict,
                                                           train_day_range= [DAYFIRST + 10, DAYFIRST + 25],
                                                           val_day_range= [DAYFIRST + 25, DAYFIRST + 30],
                                                           )

model = Baseline(query_feat_dict, user_feat_dict,
              query_embed_dim, user_embed_dim,
              embed_size=16,
              hidden_dim_list=[1024, 20, 1],
              device='cpu')

optimizer = optim.Adam(params=model.parameters(), lr=5e-2)

def loop_dataset(model, dataset, optimizer= None):
    num_batches = len(dataset)
    mean_loss = 0
    mean_auc = 0
    for i, batch in enumerate(dataset):

        quest_feats, user_feats, target = batch

        predict = model(quest_feats, user_feats)
        loss = nn.BCELoss()(predict, target)
        auc_score = roc_auc_score(target, predict.detach())

        if optimizer is not None:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        mean_loss = (i * mean_loss + loss) / (i + 1)
        mean_auc = (i * mean_auc + auc_score) / (i + 1)

        if i % args.print_every == 0:
            print("%d / %d: loss %.4f auc %.4f" %(i, num_batches, mean_loss, mean_auc))

        if i > 64:
            break

    return mean_loss, mean_auc

best_pfms = 0
count = 0
for epoch in range(args.epoches):
    print("training epoch %d ..." %(epoch, ))
    t0 = time()
    model.train()
    train_loss, train_auc = loop_dataset(model, train_dataset, optimizer)
    t1 = time()
    print("training epoch %d in %d minutes: loss %.4f auc %.4f\n" %(epoch, (t1 - t0) / 60, train_loss, train_auc))

    print("validation epoch %d ..." %(epoch, ))
    t2 = time()
    model.eval()
    val_loss, val_auc = loop_dataset(model, val_dataset)
    t3 =time()
    print("valid epoch %d in %d minutes: loss %.4f auc %.4f\n" %(epoch, (t3 - t2) / 60, val_loss, val_auc))

    if epoch > 2:
        break

    if val_auc > best_pfms:
        best_pfms = val_auc
        torch.save(model, chkpt_path)
        count = 0
    else:
        count += 1
        if count > args.patience:
            print("early stopping at epoch %d: performance %.4f." %(epoch, best_pfms))

model = torch.load(chkpt_path)


res_list = []
for i, batch in enumerate(test_dataset):
    quest_feats, user_feats = batch

    predict = model(quest_feats, user_feats)
    res_list.extend(predict.detach().numpy().squeeze().tolist())

test_df = pd.read_csv(os.path.join(dataDir, 'invite_info_evaluate_1021.csv'),
                      sep= '\t',
                      usecols= ['question_id', 'user_id', 'create_time'])
test_df['predict_value'] = res_list

test_df.to_csv(os.path.join(dataDir, 'result_1103.csv'), sep= '\t', header= False, index= False)