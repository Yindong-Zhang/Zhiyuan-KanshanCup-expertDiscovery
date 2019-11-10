
def move_feat_dict_to_gpu(features_dict):
    for type, type_dict in features_dict.items():
        for feat, column in type_dict.items():
            type_dict[feat] = column.cuda()
        features_dict[type] = type_dict
    return features_dict