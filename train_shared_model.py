from operator import pos
import os  
import os.path
import math
import random
from re import L

import time

import concurrent
import itertools
from matplotlib import cm
from scipy import sparse

from scipy.sparse import csr_matrix
import scipy.sparse as sp
import seaborn as sn
import networkx as nx
import pandas as pd
import numpy as np

import tensorflow as tf

from sklearn import metrics
import matplotlib.pyplot as plt
from tensorflow.python.keras.utils.tf_utils import convert_inner_node_data

from constants import *
from utils.data import load_data, get_test_edges, get_false_edges, sparse_to_tuple, get_complete_cora_data, get_complete_data
from networks.shared_model import LastSharedWithAdversarialLoss, LastShared, FirstShared    
from loss import total_loss, topological_loss
from utils.utils import convert_sparse_matrix_to_sparse_tensor, compute_adj_norm, plot_cf_matrix, plot

from trainers.SharedLayerTrainer import SharedTrainer
from trainers.AdvLossTrainer import SharedTrainerWithAdvLoss

# set the seed
tf.random.set_seed(SEED)
random.seed(SEED)
np.random.RandomState(SEED)

from utils.save_model import save_shared_model_embs as save_model


# Include standard modules
import getopt, sys

# Get full command-line arguments
full_cmd_arguments = sys.argv

argument_list = full_cmd_arguments[1:]

print(argument_list)

short_options = "d:c:"
long_options = ["dataset=", "clusters=", "mse", "test", "fc", "constlabel", "adv", "firstshared"]

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
        EPOCHS = 6
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

def get_edges_formatted(matrix:sp.csr_matrix, clust_to_node:dict, n_clusters):
    # ensure that if there is an edge [i,j] than there is also [j, i]
    matrix = matrix + matrix.T
    edges = None
    for clust_1 in range(n_clusters):
        for clust_2 in range(clust_1, n_clusters):
            matrix_c1_c2 = matrix[clust_to_node[clust_1], :][:, clust_to_node[clust_2]]
            edges_c1_c2, _, _ = sparse_to_tuple(matrix_c1_c2)

            if edges_c1_c2.shape[0]>0: 
                from_to_clust = np.array([[clust_1, clust_2]]*len(edges_c1_c2))

                edges_c1_c2 = np.concatenate((edges_c1_c2, from_to_clust), 1)

                if edges is None:
                    edges = edges_c1_c2
                else:
                    edges = np.concatenate((edges, edges_c1_c2))

    
    return edges
        

if __name__ == "__main__":

    # load data
    data = load_data(DATASET_NAME, N_CLUSTERS)
    complete_data  = get_complete_data(DATASET_NAME, N_CLUSTERS, LEAVE_INTRA_CLUSTERS)

    # extact the informations stored into data
    adjs, features_, tests, valids, clust_to_node, node_to_clust, com_idx_to_clust_idx = data

    # get the false edges and save them for each couple of clusters
    complete_train_matrix, _, complete_test_matrix, complete_valid_matrix  = complete_data
    
    # turn the dictionary into a list of features
    features = [features_[i] for i in range(len(features_))]

    train_edges = get_edges_formatted(complete_train_matrix, clust_to_node, N_CLUSTERS)
    valid_edges = get_edges_formatted(complete_valid_matrix, clust_to_node, N_CLUSTERS)
    test_edges = get_edges_formatted(complete_test_matrix, clust_to_node, N_CLUSTERS)


    test_false_matrix, valid_false_matrix = get_test_valid_false_edges(complete_data)
    
    # in the case of train false edges I use as true matrix the one of training and not the complete
    # one, in order to not invalidate the traininig by knowing which are the real false edges
    train_false_edges = get_false_edges(complete_train_matrix, train_edges.shape[0])
    train_false_matrix = sp.csr_matrix((np.ones(train_false_edges.shape[0]), (train_false_edges[:,0], train_false_edges[:,1])), complete_train_matrix.shape)

    # get the edges ready with the indexes inside the cluster and the indication of which cluster
    train_false_edges = get_edges_formatted(train_false_matrix, clust_to_node, N_CLUSTERS)
    valid_false_edges = get_edges_formatted(valid_false_matrix, clust_to_node, N_CLUSTERS)
    test_false_edges = get_edges_formatted(test_false_matrix, clust_to_node, N_CLUSTERS)

    adj_train_norm_list = [compute_adj_norm(adjs[clust] + adjs[clust].T) for clust in range(N_CLUSTERS)]

    if ADV_LOSS:
        trainer = SharedTrainerWithAdvLoss(False, "SharedTrainerWithAdvLoss", BATCH_SIZE, PATIENCE, EPOCHS, LR, N_CLUSTERS, DATASET_NAME, False, LABEL_OF_ALL_1, 5)
    else:
        trainer = SharedTrainer(not SHARE_FIRST, "SharedModel", BATCH_SIZE, PATIENCE, EPOCHS, LR, N_CLUSTERS, DATASET_NAME)

    trainer.prepare_data(adj_train_norm_list, features, 
        train_edges, train_false_edges, valid_edges, valid_false_edges)

    trainer.train()

    execution_time = trainer.execution_time

    cms, _, _ = trainer.test(test_edges, test_false_edges, TEST)

    f1s, precs, recs = [], [], []
    for cm in cms:
        tp, fp, fn = cm[1][1], cm[0][1], cm[1][0]
        precs.append(tp/(tp+fp))
        recs.append(tp/(tp+fn))
        f1s.append(2*precs[-1]*recs[-1]/(precs[-1]+recs[-1])) 

    print(f1s)

    if not TEST:
        with open(f"results/{trainer.name}.txt", "a") as fout:
            fout.write(f"precs: {precs}\n")
            fout.write(f"recs: {recs}\n")
            fout.write(f"f1s: {f1s}\n")
            fout.write(f"time: {execution_time}\n")
            fout.write("-"*10 + "\n")

