import logging
import os
import random

import numpy as np
import torch

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


def print_args(args):
    logger.info("=== args ===")
    for k, v in args.__dict__.items():
        logger.info(f"{k}: {v}")


def get_model_path_name(args):
    source_model_dir_path = os.path.join(args.save_model_dir,
                                         f"{os.path.basename(args.source_dir)}-{args.way_num}-{args.shot_num}")
    suffix = ""
    if args.post_avg_by_class:
        suffix = suffix + "_post_avg_by_class"
    if args.prior_avg_by_class:
        suffix = suffix + "_prior_avg_by_class"
    
    source_model_path = os.path.join(source_model_dir_path, f"model{suffix}.pth")
    return source_model_path


def set_seeds(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)


def get_labels(path):
    labels = []
    for line in open(path, "r"):
        labels.append(line.strip())
    return labels


def get_map(label_list, merge_BI=True):
    if merge_BI:
        label_map = {}
        for label in label_list:
            if label.startswith('B-') or label.startswith('I-'):
                label = 'I-' + label[2:]
            if label not in label_map:
                label_map[label] = len(label_map)
        return label_map


def get_evaluate_fpr(y_pred, y_true, O_id=0):
    prior_pred = []
    prior_true = []
    ent_pred = []
    ent_true = []
    
    for _idx, value in enumerate(y_pred):
        if value != O_id:
            ent_pred.append((_idx, value))
            prior_pred.append((_idx, 1))
    
    for _idx, value in enumerate(y_true):
        if value != O_id:
            ent_true.append((_idx, value))
            prior_true.append((_idx, 1))
    
    open("cls_test_record.txt", 'a').write(f"ent_pred:{ent_pred}\nent_true:{ent_true}\n\n")
    
    R = set(ent_pred)
    T = set(ent_true)
    X = len(R & T)
    Y = len(R)
    Z = len(T)
    if Y == 0 or Z == 0:
        ent_f1, ent_precision, ent_recall = 0, 0, 0
    else:
        ent_f1, ent_precision, ent_recall = 2 * X / (Y + Z), X / Y, X / Z
    
    R = set(prior_pred)
    T = set(prior_true)
    X = len(R & T)
    Y = len(R)
    Z = len(T)
    if Y == 0 or Z == 0:
        prior_f1, prior_precision, prior_recall = 0, 0, 0
    else:
        prior_f1, prior_precision, prior_recall = 2 * X / (Y + Z), X / Y, X / Z
    
    return prior_f1, prior_precision, prior_recall, ent_f1, ent_precision, ent_recall


def draw_distributions(args, label_list, vector_list,
                       no_ent_vector, ent_avg_vector,
                       test_vectors=None):
    from sklearn.decomposition import PCA
    from matplotlib import pyplot
    
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
    pca = PCA(n_components=2)
    model = pca.fit(test_vectors)
    
    support_result = model.transform(vector_list)
    if no_ent_vector and ent_avg_vector:
        no_ent_avg = model.transform(no_ent_vector[None, :])
        ent_avg = model.transform(ent_avg_vector[None, :])
    
    pyplot.figure(figsize=(24, 12))
    
    if test_vectors != None:
        pyplot.subplot(2, 1, 1)
        test_result = model.transform(test_vectors)
        pyplot.scatter(test_result[:, 0], test_result[:, 1], marker='.',
                       color=[colors[0]] * len(test_result))
        pyplot.subplot(2, 1, 2)
    
    pyplot.scatter(support_result[:, 0], support_result[:, 1], marker='.',
                   color=[colors[1 + int(i)] for i in label_list])
    if no_ent_vector and ent_avg_vector:
        pyplot.scatter(no_ent_avg[0, 0], no_ent_avg[0, 1], marker='x',
                       color="r")
        pyplot.scatter(ent_avg[0, 0], ent_avg[0, 1], marker='x',
                       color="b")
    
    # for i, label in enumerate(label_list):
    #     pyplot.annotate(int(label), xy=(result[i, 0], result[i, 1]), color=colors[int(label)])
    
    # save_name = f"{args.support_file_name}_{os.path.basename(os.path.join(args.source_model_path))[6:-4]}"
    save_name = "test"
    
    pyplot.savefig(f"{save_name}.jpg")
    pyplot.show()


def get_entity_vectors(input_ids, entity_vectors):
    valid_vecs = None
    _, s_, s_, hidden_size = entity_vectors.shape
    
    for ent_vec, input_id in zip(entity_vectors, input_ids):
        valid_token_len = len(torch.where(input_id != 0)[0])  # delete paddings
        
        # filter [cls] / [sep] / start_id > end_id
        pred_mask = torch.triu(torch.ones(s_, s_), diagonal=0)
        pred_mask[0] = 0
        pred_mask[:, 0] = 0
        pred_mask[valid_token_len - 1:] = 0
        pred_mask[:, valid_token_len - 1:] = 0
        
        if valid_vecs == None:
            valid_vecs = ent_vec[pred_mask != 0].reshape(-1, hidden_size)
        else:
            valid_vecs = torch.vstack((valid_vecs,
                                       ent_vec[pred_mask != 0].reshape(-1, hidden_size)))
    
    return valid_vecs
