from src.baseline.dataset import create_train_val_dataset
from src.baseline.baseline import Model
from sklearn.metrics import auc
from src.config import DAYFIRST
from torch import optim, nn
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--epoches', type= int, default= 10000, help= 'training epoches')
parser.add_argument('--print_every', type= int, default = 32, help= 'print intervals.')
args = parser.parse_args()

wv_size = 64
max_hist_len = 16
query_embed_dim = 128
hist_embed_dim= 128
user_embed_dim= 128
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
# dataset = Dataset('../../data',
#                   batchsize= 32,
#                   question_feat_dict= query_feat_dict,
#                   user_feat_dict= user_feat_dict,
#                   answer_feat_dict= history_feat_dict,
#                   max_hist_len = max_hist_len,
#                   )

train_dataset, val_dataset = create_train_val_dataset('../../data',
                  batchsize= 32,
                  question_feat_dict= query_feat_dict,
                  user_feat_dict= user_feat_dict,
                  answer_feat_dict= history_feat_dict,
                  max_hist_len = max_hist_len,
                                                      train_day_range= [DAYFIRST + 10, DAYFIRST + 25],
                                                      val_day_range= [DAYFIRST + 25, DAYFIRST + 30],
                                                      )

model = Model(query_feat_dict, history_feat_dict, user_feat_dict,
              query_embed_dim, hist_embed_dim, user_embed_dim,
              embed_size=16,
              max_hist_len=max_hist_len,
              hidden_dim_list=[512, 1],
              device='cpu')
optimizer = optim.Adam(params=model.parameters(), lr=1e-3)

def loop_dataset(model, dataset, optimizer= None):
    num_batches = len(dataset)
    mean_loss = 0
    mean_auc = 0
    for i, batch in enumerate(dataset):

        quest_feats, hist_feat_list, hist_len, user_feats, target = batch

        optimizer.zero_grad()
        predict = model(quest_feats, hist_feat_list, hist_len, user_feats)
        loss = nn.BCELoss()(predict, target)
        auc_score = auc(predict, target)

        if optimizer is not None:
            loss.backward()
            optimizer.step()

        mean_loss = (i * mean_loss + loss) / (i + 1)
        mean_auc = (i * mean_auc + auc_score) / (i + 1)

        if i % args.print_every == 0:
            print("%d / %d: loss %.4f auc %。4f" %(i, num_batches, mean_loss, mean_auc))

        if i > 20:
            break

    return mean_loss, mean_auc

for epoch in range(args.epoches):
    print("training epoch %d ..." %(epoch, ))
    train_loss, train_auc = loop_dataset(model, train_dataset, optimizer)
    print("training epoch %d: loss %.4f auc %.4f\n" %(epoch, train_loss, train_auc))

    print("validation epoch %d ..." %(epoch, ))
    val_loss, val_auc = loop_dataset(model, val_dataset)
    print("valid epoch %d: loss %.4f auc %.4f\n" %(epoch, val_loss, val_auc))

    if epoch > 2:
        break

