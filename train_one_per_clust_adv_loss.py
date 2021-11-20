import os.path
import math
import random
from re import L

from scipy.sparse import csr_matrix
import scipy.sparse as sp
import seaborn as sn
import pandas as pd
import numpy as np
from seaborn.distributions import ecdfplot

import tensorflow as tf

from sklearn import metrics
import matplotlib.pyplot as plt
from tensorflow.python.ops.gen_linalg_ops import matrix_triangular_solve

from constants import *
from utils.data import load_data, get_test_edges, get_false_edges, sparse_to_tuple, get_complete_cora_data, get_complete_data
from networks.gae_model import GAEModel    
from loss import total_loss, topological_loss

from utils.save_model import save_gae_model_embs as save_model
from utils.utils import *

from trainers.AdvLossTrainer import TrainerWithAdvLoss

import time

# set the seed
tf.random.set_seed(SEED)
random.seed(SEED)
np.random.RandomState(SEED)

# Include standard modules
import getopt, sys

# Get full command-line arguments
full_cmd_arguments = sys.argv

argument_list = full_cmd_arguments[1:]

print(argument_list)

short_options = "d:c:"
long_options = ["dataset=", "clusters=", "mse", "test", "nonneggrad", "fc", "constlabel"]

TEST, ADV_LOSS = False, False

try:
    arguments, values = getopt.getopt(argument_list, short_options, long_options)
except getopt.error as err:
    # Output error, and return with an error code
    print (str(err))
    sys.exit(2)

# Evaluate given options
for current_argument, current_value in arguments:
    if current_argument in ("-d", "--dataset"):
        DATASET_NAME = current_value
        print("DATASET_NAME:", current_value)
    
    elif current_argument in ("-c", "--clusters"):
        N_CLUSTERS = current_value
        print("Num Clusters:", N_CLUSTERS)
    
    elif current_argument in ("--constlabel"):
        LABEL_OF_ALL_1, ADV_LOSS = True, True
        print("CONST LABEL:", True)
    
    elif current_argument in ("--mse"):
        print("USING MSE LOSS WITH CONSTANT LABEL")
        LABEL_OF_ALL_1, MSE_LOSS, ADV_LOSS = True, True, True    

    elif current_argument in ("--test"):
        EPOCHS = 2
        TEST = True
        print("Testing train: n epochs:", EPOCHS)
    
    elif current_argument in ("--adv"):
        print("training the shared model with adv loss function")
        ADV_LOSS = True

    elif current_argument in ("--firstshared"):
        ADV_LOSS = False
        SHARE_FIRST = True

if LABEL_OF_ALL_1 and NON_NEG_GRAD:
    raise Exception("ONLY ONE OF THE TWO CAN BE TRUE: LABEL_OF_ALL_1, NON_NEG_GRAD")

print("MSE_LOSS", MSE_LOSS, "LABEL_OF_ALL_1", LABEL_OF_ALL_1)


"""
returns a np array containing for each test edge:
    n_from_couple, n_to_couple, from_clust, to_clust \
    where n_from_couple and n_to_couple are respectively the clust of the first node and the clust of the second node
    from_clust are to_clust
"""
def prepare_edges(matrix, clust_to_node):
    matrix = matrix + matrix.T
    n_clusters = len(clust_to_node)

    edges = None

    for clust_from in range(n_clusters):
        for clust_to in range(clust_from, n_clusters):
            matrix_c1_c2 = matrix[clust_to_node[clust_from], :][:, clust_to_node[clust_to]]
            
            edges_c1_c2, _, _ = sparse_to_tuple(matrix_c1_c2)
            labels_c1_c2 = [[clust_from, clust_to]]*edges_c1_c2.shape[0]

            edges_lables_c1_c2 = np.concatenate((edges_c1_c2, labels_c1_c2), 1)
            if edges is None:
                edges = edges_lables_c1_c2
            else:
                edges = np.concatenate((edges, edges_lables_c1_c2), 0)

    return edges


def test(models, features, test_edges, false_test_edges, dataset):
    features_tensors = [convert_sparse_matrix_to_sparse_tensor(features[i]) for i in range(len(features))] 
    
    embs = []
    for i in range(len(models)):
        embs.append(models[i](features_tensors[i]))

    preds_all, labels_all = None, None

    for clust_from in range(len(models)):
        for clust_to in range(clust_from, len(models)):
            
            edges_from_to_idxs = (test_edges[:, 2] == clust_from) * (test_edges[:,3] == clust_to)
            
            edges_from_to = test_edges[edges_from_to_idxs]

            false_edges_from_to_idxs = (false_test_edges[:, 2] == clust_from) * (false_test_edges[:, 3] == clust_to)
            false_edges_from_to = false_test_edges[false_edges_from_to_idxs][:, :2]

            preds, labels = get_predictions_and_labels(embs[clust_from], edges_from_to, false_edges_from_to, BATCH_SIZE, embs[clust_to])

            if preds_all is None:
                preds_all, labels_all = preds, labels
            else:
                preds_all = tf.concat([preds_all, preds], 0)
                labels_all = np.concatenate((labels_all, labels),0)
            print("preds", preds.shape, "all", preds_all.shape)
            print("labels", labels.shape, "all", labels_all.shape)


    name = f"single_models_adv_loss"
    if TRAIN_ALSO_CLASSIFIER:
        name += "_train_classifier"
    if LABEL_OF_ALL_1 :
        name += "_const_labels"

    cms = plot_cf_matrix(labels_all, preds_all, name, TEST)

    roc_score = metrics.roc_auc_score(labels_all, preds_all)
    ap_score = metrics.average_precision_score(labels_all, preds_all)


    return roc_score, ap_score, cms

def get_test_valid_false_edges(complete_data):
    adj_train, _, test_matrix, valid_matrix  = complete_data

    train_edges, _, _ = sparse_to_tuple(adj_train)
    test_edges, _, _ = sparse_to_tuple(test_matrix)
    valid_edges, _, _ = sparse_to_tuple(valid_matrix)
    
    data = [1]*(len(train_edges) + len(test_edges) + len(valid_edges))
    indexes = np.concatenate((train_edges,test_edges, valid_edges), 0)

    complete_adj = csr_matrix((data, (indexes[:,0], indexes[:,1])), shape = adj_train.shape)

    false_edges = get_false_edges(complete_adj, test_edges.shape[0] + valid_edges.shape[0])

    # split the false edges into test and validation
    valid_false_edges = false_edges[:valid_edges.shape[0]]
    test_false_edges = false_edges[valid_edges.shape[0]:]

    test_ones = [1]*test_false_edges.shape[0]
    valid_ones = [1]*valid_false_edges.shape[0]

    # build the sparse matrices relative to the test and valid false edges
    test_false_matrix = csr_matrix((test_ones, (test_false_edges[:,0], test_false_edges[:,1])), adj_train.shape)
    valid_false_matrix = csr_matrix((valid_ones, (valid_false_edges[:,0], valid_false_edges[:,1])), adj_train.shape)

    return test_false_matrix, valid_false_matrix

if __name__ == "__main__":
    
    # load data
    single_data = load_data(DATASET_NAME, N_CLUSTERS, get_couples = False)
    complete_data  = get_complete_data(DATASET_NAME, N_CLUSTERS, leave_intra_clust_edges=LEAVE_INTRA_CLUSTERS)

    adjs, features, tests_single, valids_single, clust_to_node_single, node_to_clust_single, _ = single_data
    _, _, test_matrix, _ = complete_data

    test_false_matrix, valid_false_matrix = get_test_valid_false_edges(complete_data)
    
    models = []
    for i in range(len(adjs)):
        
        ith_trainer = TrainerWithAdvLoss(f"single_model_adv_loss_{i}", BATCH_SIZE, PATIENCE, EPOCHS, LR, N_CLUSTERS, i, False, True, NUM_EPOCHS_ADV_LOSS)
        
        ith_valid_false_edges, _, _ = sparse_to_tuple(valid_false_matrix[clust_to_node_single[i], :][:, clust_to_node_single[i]])
        ith_valid_edges, _, _ = sparse_to_tuple(valids_single[i])
        
        ith_trainer.prepare_data(adjs[i], features[i], ith_valid_edges, ith_valid_false_edges)

        ith_trainer.train()

        models.append(ith_trainer.model)


    test_edges = prepare_edges(test_matrix, clust_to_node_single)
    test_false_edges = prepare_edges(test_false_matrix, clust_to_node_single)

    roc_score, ap_score, cms = test(models, features, test_edges, test_false_edges, DATASET_NAME)

    print(cms)
        

