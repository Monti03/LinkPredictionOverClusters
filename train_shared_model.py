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
from networks.shared_model import LastShared, FirstShared    
from loss import total_loss, topological_loss

# set the seed
tf.random.set_seed(SEED)
random.seed(SEED)
np.random.RandomState(SEED)

from utils.save_model import save_shared_model_embs as save_model

from train_couples import train as complete_train
from train_couples import test as complete_test

# Include standard modules
import getopt, sys

# Get full command-line arguments
full_cmd_arguments = sys.argv

argument_list = full_cmd_arguments[1:]

print(argument_list)

short_options = "d:c:"
long_options = ["dataset=", "clusters=", "mse", "test", "nonneggrad", "fc", "constlabel"]

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
    elif current_argument in ("-c", "--clusters"):
        N_CLUSTERS = current_value
    elif current_argument in ("--test"):
        EPOCHS = 5
        print("Testing train: n epochs:", EPOCHS)
        
        
# convert sparse matrix to sparse tensor
def convert_sparse_matrix_to_sparse_tensor(sparse_mx):
    if not sp.isspmatrix_coo(sparse_mx):
        sparse_mx = sparse_mx.tocoo()
    indices_matrix = np.vstack((sparse_mx.row, sparse_mx.col)).transpose()
    values = sparse_mx.data
    shape = sparse_mx.shape
    sparse_tensor = tf.SparseTensor(indices=indices_matrix, values=values, dense_shape=shape)
    return tf.cast(sparse_tensor, dtype=tf.float32)


def comp_idxs_to_clusts_idxs_multi_thread(edges, node_to_clust, clust_to_node, com_idx_to_clust_idx):
    executor = concurrent.futures.ProcessPoolExecutor()
    
    n_edges = edges.shape[0]
    n_batch = n_edges//7

    batches = [edges[i*n_batch: (i+1)*n_batch] for i in range(7)]
    batches.append(edges[7*n_batch:])

    converted_edges_list = [executor.submit(comp_idxs_to_clusts_idxs, batches[i], node_to_clust, clust_to_node, com_idx_to_clust_idx) for i in range(8)]
    res = concurrent.futures.wait(converted_edges_list)
    print(res)
    converted_edges = []
    for i in range(8):
        print(converted_edges_list.__class__)
        converted_edges += converted_edges_list[i].result()
        pass
    return np.array(converted_edges)

def comp_idxs_to_clusts_idxs(edges, node_to_clust, clust_to_node, com_idx_to_clust_idx):
    print("start comp_idxs_to_clusts_idxs")
    converted_edges = []
    for edge in edges:
        comp_idx_from, comp_idx_to = edge[0], edge[1]
        
        from_clust, to_clust = node_to_clust[comp_idx_from], node_to_clust[comp_idx_to]

        clust_idx_from, clust_idx_to = com_idx_to_clust_idx[comp_idx_from], com_idx_to_clust_idx[comp_idx_to]

        if(from_clust < to_clust):
            converted_edges.append([clust_idx_from, clust_idx_to, from_clust, to_clust])        
        elif(from_clust > to_clust):
            converted_edges.append([clust_idx_to, clust_idx_from, to_clust, from_clust])        

    print("end comp_idxs_to_clusts_idxs")

    return converted_edges

def train_preds(embs_from, embs_to, tmp_train_edges):
    # get the train edges of this batch
    tmp_from = tmp_train_edges[:,0]
    tmp_to = tmp_train_edges[:,1]

    # gather the embeddings of the nodes relative to the trained edges
    embs_from_ = tf.gather(embs_from, tmp_from)
    embs_to_ = tf.gather(embs_to, tmp_to)

    # obtain the logits by a scalar product of the embs of the two nodes of the corresponding edge
    train_pos_pred = tf.linalg.diag_part(tf.matmul(embs_from_, embs_to_, transpose_b=True))
    
    return train_pos_pred

def train(features_list, adj_train_list, adj_train_norm_list, train_edges, valid_edges, valid_false_edges, intra_train_edges, intra_valid_edges, intra_valid_false_edges, complete_adj, share_first, clust_to_node, com_idx_to_clust_idx, node_to_clust=None):
    n_nodes = [cluster.shape[0] for cluster in features_list]
    print(n_nodes)
    condition = MATRIX_OPERATIONS
    for clust_1 in n_nodes:
        for clust_2 in n_nodes:
            condition = condition and (clust_1 * clust_2 <= BATCH_SIZE * BATCH_SIZE)
            print(condition)
            if not condition:
                print(clust_1, clust_2, clust_2*clust_1)

    if condition:
        print("MATRIX TRAIN")
        return matrix_train(features_list, adj_train_list, adj_train_norm_list, train_edges, valid_edges, valid_false_edges, intra_train_edges, intra_valid_edges, intra_valid_false_edges, complete_adj, share_first, clust_to_node, com_idx_to_clust_idx, node_to_clust)
    else:
        print("BATCHED TRAIN")
        return batched_train(features_list, adj_train_list, adj_train_norm_list, train_edges, valid_edges, valid_false_edges, intra_train_edges, intra_valid_edges, intra_valid_false_edges, complete_adj, share_first, clust_to_node, com_idx_to_clust_idx, node_to_clust)


def matrix_train(features_list, adj_train_list, adj_train_norm_list, train_edges, valid_edges, valid_false_edges, intra_train_edges, intra_valid_edges, intra_valid_false_edges, complete_adj, share_first, clust_to_node, com_idx_to_clust_idx, node_to_clust=None):
    train_accs = []
    train_losses = []

    valid_accs = []
    valid_losses = []
    
    patience = 0

    n_clusters = len(features_list)

    assert n_clusters == len(adj_train_list) and n_clusters == len(adj_train_norm_list)
    assert n_clusters == len(train_edges) and n_clusters == len(valid_edges) and n_clusters == len(valid_false_edges)

    # define the optimizer
    optimizer = tf.keras.optimizers.Adam(learning_rate=LR)
    
    # get the number of nodes per cluster
    n_nodes = [adj_train.shape[0] for adj_train in adj_train_list]

    # convert the normalized adj and the features to tensors
    adj_train_norm_tensor = [convert_sparse_matrix_to_sparse_tensor(adj_train_norm) for adj_train_norm in adj_train_norm_list]
    feature_tensor = [convert_sparse_matrix_to_sparse_tensor(features) for features in features_list]

    # initialize the model    
    if share_first:
        model = FirstShared(adj_train_norm_tensor)
    else:
        model = LastShared(adj_train_norm_tensor)
    
    # flatten the matrix edges
    valid_edges_indeces = [[x[0]*n_nodes[cluster] + x[1] for x in valid_edges[cluster]] for cluster in range(n_clusters)]
    valid_false_edges_indeces = [[x[0]*n_nodes[cluster] + x[1] for x in valid_false_edges[cluster]] for cluster in range(n_clusters)]

    intra_valid_edges = comp_idxs_to_clusts_idxs_multi_thread(intra_valid_edges, node_to_clust, clust_to_node, com_idx_to_clust_idx)
    intra_valid_false_edges = comp_idxs_to_clusts_idxs_multi_thread(intra_valid_false_edges, node_to_clust, clust_to_node, com_idx_to_clust_idx)
    

    print(f"valid_edges_indeces: {[len(i) for i in valid_edges_indeces]}")
    print(f"valid_false_edges_indeces: {[len(i) for i in valid_false_edges_indeces]}")

    # get random false edges to train
    train_false_edges = [get_false_edges(adj_train_list[cluster], len(train_edges[cluster]), node_to_clust=node_to_clust) for cluster in range(n_clusters)]
    intra_train_false_edges = get_false_edges(complete_adj, intra_train_edges.shape[0], node_to_clust=None)

    intra_train_false_edges = comp_idxs_to_clusts_idxs_multi_thread(intra_train_false_edges, node_to_clust, clust_to_node, com_idx_to_clust_idx)
    intra_train_edges = comp_idxs_to_clusts_idxs_multi_thread(intra_train_edges, node_to_clust, clust_to_node, com_idx_to_clust_idx)


    print("train_false_edges", [clust_false_edges.shape[0] for clust_false_edges in train_false_edges])
    print("train_pos_edges", [len(edges) for edges in train_edges])

    
    n_epochs = 0
    for epoch in range(EPOCHS):
        n_epochs = epoch
        epoch_losses = []
        epoch_accs = []

        train_pred, tmp_train_y  = None, None
        clusts_loss = [] 

        with tf.GradientTape() as tape:  
            
            embs_list = []

            for clust in range(n_clusters):
                
                # get the train edges
                clust_train_edges = train_edges[clust]
                clust_train_false_edges = train_false_edges[clust]
                print("predicting")
                # forward pass -> obtain the embeddings of the nodes
                embs = model(feature_tensor[clust], cluster=clust, training=True)
                
                # save embs to use them in the prediction of the edges between the clusters
                embs_list.append(embs)

                complete_graph_preds = tf.matmul(embs, embs, transpose_b=True)
                train_pos_pred = tf.gather_nd(complete_graph_preds, clust_train_edges)
                
                train_neg_pred = tf.gather_nd(complete_graph_preds, clust_train_false_edges)

                # concatenate the results
                clust_train_pred = tf.concat((train_pos_pred, train_neg_pred), -1)
                clust_tmp_train_y = tf.concat([tf.ones(train_pos_pred.shape[0]),tf.zeros(train_neg_pred.shape[0])], -1)

                if train_pred is None:
                    train_pred = clust_train_pred
                    tmp_train_y = clust_tmp_train_y

                else:
                    train_pred = tf.concat((train_pred, clust_train_pred), -1)
                    tmp_train_y = tf.concat((tmp_train_y, clust_tmp_train_y), -1)

                # get loss
                clust_loss = topological_loss(clust_tmp_train_y, clust_train_pred)
                print(f"clust_loss {clust} : {clust_loss}")

                clusts_loss.append(clust_loss)
                
            # take the predictions for  the train edges between different clusters
            for cluster_1 in range(n_clusters):
                for cluster_2 in range(cluster_1+1, n_clusters):
                    # get the indices of the edges starting from clust1 and ending in clust2
                    tmp_train_edges_1 = intra_train_edges[:,2]==cluster_1
                    tmp_train_edges_2 = intra_train_edges[:,3]==cluster_2
                    tmp_train_edges_1_2 = tmp_train_edges_1 * tmp_train_edges_2

                    # get the right edges from clust1 to clust2
                    tmp_train_edges_1_2 = intra_train_edges[tmp_train_edges_1_2]
                    
                    train_pos_pred = None
                    if(tmp_train_edges_1_2.shape[0] > 0):
                        # get the train edges
                        complete_intra_clust_preds = tf.matmul(embs_list[cluster_1], embs_list[cluster_2], transpose_b=True)
                        train_pos_pred = tf.gather_nd(complete_intra_clust_preds, tmp_train_edges_1_2[:,:2])

                    tmp_train_edges_1 = intra_train_false_edges[:,2]==cluster_1
                    tmp_train_edges_2 = intra_train_false_edges[:,3]==cluster_2
                    tmp_train_edges_1_2 = tmp_train_edges_1 * tmp_train_edges_2

                    tmp_train_edges_1_2 = intra_train_false_edges[tmp_train_edges_1_2]

                    train_neg_pred = None
                    if(tmp_train_edges_1_2.shape[0]>0):
                        # get the train edges 
                        complete_intra_clust_preds = tf.matmul(embs_list[cluster_1], embs_list[cluster_2], transpose_b=True)
                        train_neg_pred = tf.gather_nd(complete_intra_clust_preds, tmp_train_edges_1_2[:,:2])
                        

                    clust_train_pred = None
                    clust_tmp_train_y = None
                    if(train_neg_pred is not None and train_pos_pred is not None):
                        # concatenate the results
                        clust_train_pred = tf.concat((train_pos_pred, train_neg_pred), -1)
                        clust_tmp_train_y = tf.concat([tf.ones(train_pos_pred.shape[0]),tf.zeros(train_neg_pred.shape[0])], -1)

                    elif(train_pos_pred is not None):
                        clust_train_pred = train_pos_pred
                        clust_tmp_train_y = tf.ones(train_pos_pred.shape[0])

                    elif(train_neg_pred is not None):
                        clust_train_pred = train_neg_pred
                        clust_tmp_train_y = tf.ones(train_neg_pred.shape[0])
                    else:
                        continue

                    if train_pred is None:
                        train_pred = clust_train_pred
                        tmp_train_y = clust_tmp_train_y

                    else:
                        train_pred = tf.concat((train_pred, clust_train_pred), -1)
                        tmp_train_y = tf.concat((tmp_train_y, clust_tmp_train_y), -1)                

            print("batch ground truth", tmp_train_y.shape)
            print("batch prediction", train_pred.shape)
            
            loss = topological_loss(tmp_train_y, train_pred)
            
            # get gradient from loss 
            grad = tape.gradient(loss, model.trainable_variables)

            # get acc
            ta = tf.keras.metrics.Accuracy()
            ta.update_state(tmp_train_y, tf.round(tf.nn.sigmoid(train_pred)))
            train_acc = ta.result().numpy()
        
            print("batch train_acc:", train_acc)

            # optimize the weights
            optimizer.apply_gradients(zip(grad, model.trainable_variables))

            epoch_losses.append(loss.numpy())
            epoch_accs.append(train_acc)

        train_losses.append(sum(epoch_losses)/len(epoch_losses))
        train_accs.append(sum(epoch_accs)/len(epoch_accs))
        print(f"\ntrain_loss: {train_losses[-1]}")
        print(f"train_acc: {train_accs[-1]}")

        # save memory
        grad = None
        train_pred = None

        print("------------\nvalidation\n------------")

        tot_valid_pred, tot_valid_y = None, None
        clusts_loss = []
        embs_list = []
        for clust in range(n_clusters):

            embs = model(feature_tensor[clust], cluster=clust,  training=False)
            embs_list.append(embs)

            complete_graph_preds = tf.matmul(embs, embs, transpose_b=True)
            if (valid_edges[clust][:,:2].shape[0]> 0):
                valid_pred_p = tf.gather_nd(complete_graph_preds, valid_edges[clust][:,:2])            
            else:
                valid_pred_p = None
            
            if (valid_false_edges[clust][:, :2].shape[0]> 0):
                valid_pred_n = tf.gather_nd(complete_graph_preds, valid_false_edges[clust][:, :2])
            else:
                valid_pred_n = None

            if valid_pred_p is not None and valid_pred_n is not None:
                valid_pred = tf.concat([valid_pred_p, valid_pred_n], 0)
            elif valid_pred_n is not None:
                valid_pred = valid_pred_n
            else:
                valid_pred = valid_pred_p

            valid_y = [1]*len(valid_edges[clust]) + [0]*len(valid_false_edges[clust])
            valid_y = tf.convert_to_tensor(valid_y, dtype=tf.float32)

            clust_loss = topological_loss(valid_y, valid_pred)
            clusts_loss.append(clust_loss)
            
            va = tf.keras.metrics.Accuracy()
            va.update_state(valid_y, tf.round(tf.nn.sigmoid(valid_pred)))
            clust_valid_acc = va.result().numpy()

            print(f"clust_loss {clust} : {clust_loss}")
            print(f"clust_acc {clust} : {clust_valid_acc}")

            if tot_valid_pred is None:
                tot_valid_pred = valid_pred
                tot_valid_y = valid_y
            else:
                tot_valid_pred = tf.concat([tot_valid_pred, valid_pred], -1)
                tot_valid_y = tf.concat([tot_valid_y, valid_y], -1)

        # VALIDATION OF EDGES AMONG DIFFERENT CLUSTERS
        print("----validation different clusters----")
        for cluster_1 in range(n_clusters):
            for cluster_2 in range(cluster_1+1, n_clusters):
                
                complete_intra_clust_preds = tf.matmul(embs_list[cluster_1], embs_list[cluster_2], transpose_b = True)

                tmp_valid_edges_1 = intra_valid_edges[:,2]==cluster_1
                tmp_valid_edges_2 = intra_valid_edges[:,3]==cluster_2
                tmp_valid_edges_1_2 = tmp_valid_edges_1 * tmp_valid_edges_2

                # get the right edges from clust1 to clust2
                tmp_valid_edges_1_2 = intra_valid_edges[tmp_valid_edges_1_2]
                
                valid_pos_pred = None
                if(tmp_valid_edges_1_2.shape[0] > 0):
                    valid_pos_pred = tf.gather_nd(complete_intra_clust_preds, tmp_valid_edges_1_2[:,:2])

                tmp_valid_edges_1 = intra_valid_false_edges[:,2]==cluster_1
                tmp_valid_edges_2 = intra_valid_false_edges[:,3]==cluster_2
                tmp_valid_edges_1_2 = tmp_valid_edges_1 * tmp_valid_edges_2

                # get the right edges from clust1 to clust2
                tmp_valid_edges_1_2 = intra_valid_false_edges[tmp_valid_edges_1_2]

                valid_neg_pred = None
                if(tmp_valid_edges_1_2.shape[0] > 0):
                    valid_neg_pred = tf.gather_nd(complete_intra_clust_preds,  tmp_valid_edges_1_2[:,:2])
                
                clust_valid_pred = None
                clust_tmp_valid_y = None
                if(valid_neg_pred is not None and valid_pos_pred is not None):
                    # concatenate the results
                    print("valid_pos_pred", valid_pos_pred.shape)
                    print("valid_pos_neg", valid_neg_pred.shape)

                    clust_valid_pred = tf.concat((valid_pos_pred, valid_neg_pred), -1)
                    clust_tmp_valid_y = tf.concat([tf.ones(valid_pos_pred.shape[0]),tf.zeros(valid_neg_pred.shape[0])], -1)

                elif(valid_pos_pred is not None):
                    clust_valid_pred = valid_pos_pred
                    clust_tmp_valid_y = tf.ones(valid_pos_pred.shape[0])

                elif(valid_neg_pred is not None):
                    clust_valid_pred = valid_neg_pred
                    clust_tmp_valid_y = tf.ones(valid_neg_pred.shape[0])
                else:
                    continue

                if tot_valid_pred is None:
                    tot_valid_pred = clust_valid_pred
                    tot_valid_y = clust_tmp_valid_y

                else:
                    tot_valid_pred = tf.concat((tot_valid_pred, clust_valid_pred), -1)
                    tot_valid_y = tf.concat((tot_valid_y, clust_tmp_valid_y), -1)

        #loss_weights = tf.convert_to_tensor([[-0.1]*len(valid_edges[clust]) + [1]*len(valid_false_edges[clust])])

        print("valid truth shape", tot_valid_y.shape)
        print("valid pred shape", tot_valid_pred.shape)

        #valid_loss = topological_loss(tot_valid_y, tot_valid_pred)
        valid_loss = tf.reduce_mean(clusts_loss)

        va = tf.keras.metrics.Accuracy()
        va.update_state(tot_valid_y, tf.round(tf.nn.sigmoid(tot_valid_pred)))
        valid_acc = va.result().numpy()

        print(f"valid_loss: {valid_loss.numpy()}")
        print(f"valid_acc: {valid_acc}")
        print(f"valid_losses_len: {len(valid_losses)}")
        print(f"patience: {patience}")
        
        valid_loss_np = valid_loss.numpy()

        if(len(valid_losses) > 0 and min(valid_losses) < valid_loss_np):
            print("increase patience")
            print(min(valid_losses), valid_loss_np)
            patience += 1
        else:
            print("zero patience")
            if(len(valid_losses)>0):
                print(min(valid_losses), valid_loss_np)
            patience = 0
        print(patience)
        valid_losses.append(valid_loss.numpy())
        valid_accs.append(valid_acc)
        
        if(patience > PATIENCE):
            print("breaking")
            break
        print("#"*20)

    model_name  = "first" if share_first else "last"

    plot(train_losses, valid_losses, "loss", f"shared_{model_name}")
    plot(train_accs, valid_accs, "acc", f"shared_{model_name}")

    with open("plots/n_epochs.txt", "a") as fout:
        fout.write(f"{model_name}, {DATASET_NAME}, {n_epochs}\n")

    return model

def batched_train(features_list, adj_train_list, adj_train_norm_list, train_edges, valid_edges, valid_false_edges, intra_train_edges, intra_valid_edges, intra_valid_false_edges, complete_adj, share_first, clust_to_node, com_idx_to_clust_idx, node_to_clust=None):
    train_accs = []
    train_losses = []

    valid_accs = []
    valid_losses = []
    
    patience = 0

    n_clusters = len(features_list)

    assert n_clusters == len(adj_train_list) and n_clusters == len(adj_train_norm_list)
    assert n_clusters == len(train_edges) and n_clusters == len(valid_edges) and n_clusters == len(valid_false_edges)

    # define the optimizer
    optimizer = tf.keras.optimizers.Adam(learning_rate=LR)
    
    # get the number of nodes per cluster
    n_nodes = [adj_train.shape[0] for adj_train in adj_train_list]

    # convert the normalized adj and the features to tensors
    adj_train_norm_tensor = [convert_sparse_matrix_to_sparse_tensor(adj_train_norm) for adj_train_norm in adj_train_norm_list]
    feature_tensor = [convert_sparse_matrix_to_sparse_tensor(features) for features in features_list]

    # initialize the model    
    if share_first:
        model = FirstShared(adj_train_norm_tensor)
    else:
        model = LastShared(adj_train_norm_tensor)
    
    # flatten the matrix edges
    valid_edges_indeces = [[x[0]*n_nodes[cluster] + x[1] for x in valid_edges[cluster]] for cluster in range(n_clusters)]
    valid_false_edges_indeces = [[x[0]*n_nodes[cluster] + x[1] for x in valid_false_edges[cluster]] for cluster in range(n_clusters)]

    intra_valid_edges = comp_idxs_to_clusts_idxs_multi_thread(intra_valid_edges, node_to_clust, clust_to_node, com_idx_to_clust_idx)
    intra_valid_false_edges = comp_idxs_to_clusts_idxs_multi_thread(intra_valid_false_edges, node_to_clust, clust_to_node, com_idx_to_clust_idx)
    

    print(f"valid_edges_indeces: {[len(i) for i in valid_edges_indeces]}")
    print(f"valid_false_edges_indeces: {[len(i) for i in valid_false_edges_indeces]}")

    # get random false edges to train
    train_false_edges = [get_false_edges(adj_train_list[cluster], len(train_edges[cluster]), node_to_clust=node_to_clust) for cluster in range(n_clusters)]
    intra_train_false_edges = get_false_edges(complete_adj, intra_train_edges.shape[0], node_to_clust=None)

    intra_train_false_edges = comp_idxs_to_clusts_idxs_multi_thread(intra_train_false_edges, node_to_clust, clust_to_node, com_idx_to_clust_idx)
    intra_train_edges = comp_idxs_to_clusts_idxs_multi_thread(intra_train_edges, node_to_clust, clust_to_node, com_idx_to_clust_idx)


    print("train_false_edges", [clust_false_edges.shape[0] for clust_false_edges in train_false_edges])
    print("train_pos_edges", [len(edges) for edges in train_edges])

    batch_size = BATCH_SIZE
    n_epochs = 0
    for epoch in range(EPOCHS):
        n_epochs = epoch
        epoch_losses = []
        epoch_accs = []

        max_batches = max([len(edges) for edges in train_edges]) // batch_size + 1

        batch_sizes = [len(edges)//max_batches for edges in train_edges]
        intra_batch_size = intra_train_edges.shape[0]//max_batches

        print("batch sizes", batch_sizes)

        # shuffle data in order to have different batches in different epochs
        #train_edges = [np.random.shuffle(clust_train_edges) for clust_train_edges in train_edges]
        #train_false_edges = [np.random.shuffle(clust_train_false_edges) for clust_train_false_edges in train_false_edges]

        for i in range(max_batches):

            train_pred, tmp_train_y  = None, None
            clusts_loss = [] 

            with tf.GradientTape() as tape:  
                
                embs_list = []

                for clust in range(n_clusters):
                    
                    # get the train edges
                    clust_train_edges = train_edges[clust]
                    clust_train_false_edges = train_false_edges[clust]
                    print("predicting")
                    # forward pass -> obtain the embeddings of the nodes
                    embs = model(feature_tensor[clust], cluster=clust, training=True)
                    
                    # save embs to use them in the prediction of the edges between the clusters
                    embs_list.append(embs)

                    # get the train edges of this batch
                    tmp_train_edges = clust_train_edges[i * batch_sizes[clust]: (i+1) * batch_sizes[clust] ]
                    
                    # get the train edges of this batch
                    tmp_from = tmp_train_edges[:,0]
                    tmp_to = tmp_train_edges[:,1]

                    # gather the embeddings of the nodes relative to the trained edges
                    embs_from_ = tf.gather(embs, tmp_from)
                    embs_to_ = tf.gather(embs, tmp_to)

                    # obtain the logits by a scalar product of the embs of the two nodes of the corresponding edge
                    train_pos_pred = tf.linalg.diag_part(tf.matmul(embs_from_, embs_to_, transpose_b=True))
                    
                    # same above, but for the false edges
                    tmp_train_false_edges = clust_train_false_edges[i * batch_sizes[clust]: (i+1) * batch_sizes[clust]]
                    
                    # get the train edges of this batch
                    tmp_from = tmp_train_false_edges[:,0]
                    tmp_to = tmp_train_false_edges[:,1]

                    # gather the embeddings of the nodes relative to the trained edges
                    embs_from_ = tf.gather(embs, tmp_from)
                    embs_to_ = tf.gather(embs, tmp_to)

                    # obtain the logits by a scalar product of the embs of the two nodes of the corresponding edge
                    train_neg_pred = tf.linalg.diag_part(tf.matmul(embs_from_, embs_to_, transpose_b=True))

                    # concatenate the results
                    clust_train_pred = tf.concat((train_pos_pred, train_neg_pred), -1)
                    clust_tmp_train_y = tf.concat([tf.ones(train_pos_pred.shape[0]),tf.zeros(train_neg_pred.shape[0])], -1)

                    if train_pred is None:
                        train_pred = clust_train_pred
                        tmp_train_y = clust_tmp_train_y

                    else:
                        train_pred = tf.concat((train_pred, clust_train_pred), -1)
                        tmp_train_y = tf.concat((tmp_train_y, clust_tmp_train_y), -1)

                    # get loss
                    clust_loss = topological_loss(clust_tmp_train_y, clust_train_pred)
                    print(f"clust_loss {clust} : {clust_loss}")

                    clusts_loss.append(clust_loss)
                
                    clust_loss = topological_loss(clust_tmp_train_y, clust_train_pred)

                

                tmp_train_edges = intra_train_edges[i * intra_batch_size: (i+1) * intra_batch_size]
                tmp_train_false_edges = intra_train_false_edges[i * intra_batch_size: (i+1) * intra_batch_size]
                
                for cluster_1 in range(n_clusters):
                    for cluster_2 in range(cluster_1+1, n_clusters):
                        # get the indices of the edges starting from clust1 and ending in clust2
                        tmp_train_edges_1 = tmp_train_edges[:,2]==cluster_1
                        tmp_train_edges_2 = tmp_train_edges[:,3]==cluster_2
                        tmp_train_edges_1_2 = tmp_train_edges_1 * tmp_train_edges_2

                        # get the right edges from clust1 to clust2
                        tmp_train_edges_1_2 = tmp_train_edges[tmp_train_edges_1_2]
                        
                        train_pos_pred = None
                        if(tmp_train_edges_1_2.shape[0] > 0):
                            # get the train edges of this batch
                            tmp_from = tmp_train_edges_1_2[:,0]
                            tmp_to = tmp_train_edges_1_2[:,1]

                            # gather the embeddings of the nodes relative to the trained edges
                            embs_from_ = tf.gather(embs_list[cluster_1], tmp_from)
                            embs_to_ = tf.gather(embs_list[cluster_2], tmp_to)

                            # obtain the logits by a scalar product of the embs of the two nodes of the corresponding edge
                            train_pos_pred = tf.linalg.diag_part(tf.matmul(embs_from_, embs_to_, transpose_b=True))

                        tmp_train_edges_1 = tmp_train_false_edges[:,2]==cluster_1
                        tmp_train_edges_2 = tmp_train_false_edges[:,3]==cluster_2
                        tmp_train_edges_1_2 = tmp_train_edges_1 * tmp_train_edges_2

                        tmp_train_edges_1_2 = tmp_train_false_edges[tmp_train_edges_1_2]

                        train_neg_pred = None
                        if(tmp_train_edges_1_2.shape[0]>0):
                            # get the train edges of this batch
                            tmp_from = tmp_train_edges_1_2[:,0]
                            tmp_to = tmp_train_edges_1_2[:,1]

                            # gather the embeddings of the nodes relative to the trained edges
                            embs_from_ = tf.gather(embs_list[cluster_1], tmp_from)
                            embs_to_ = tf.gather(embs_list[cluster_2], tmp_to)

                            # obtain the logits by a scalar product of the embs of the two nodes of the corresponding edge
                            train_neg_pred = tf.linalg.diag_part(tf.matmul(embs_from_, embs_to_, transpose_b=True))
                            

                        clust_train_pred = None
                        clust_tmp_train_y = None
                        if(train_neg_pred is not None and train_pos_pred is not None):
                            # concatenate the results
                            clust_train_pred = tf.concat((train_pos_pred, train_neg_pred), -1)
                            clust_tmp_train_y = tf.concat([tf.ones(train_pos_pred.shape[0]),tf.zeros(train_neg_pred.shape[0])], -1)

                        elif(train_pos_pred is not None):
                            clust_train_pred = train_pos_pred
                            clust_tmp_train_y = tf.ones(train_pos_pred.shape[0])

                        elif(train_neg_pred is not None):
                            clust_train_pred = train_neg_pred
                            clust_tmp_train_y = tf.ones(train_neg_pred.shape[0])
                        else:
                            continue

                        if train_pred is None:
                            train_pred = clust_train_pred
                            tmp_train_y = clust_tmp_train_y

                        else:
                            train_pred = tf.concat((train_pred, clust_train_pred), -1)
                            tmp_train_y = tf.concat((tmp_train_y, clust_tmp_train_y), -1)                

                print("batch ground truth", tmp_train_y.shape)
                print("batch prediction", train_pred.shape)
                
                loss = topological_loss(tmp_train_y, train_pred)
                #loss = tf.reduce_mean(clusts_loss)
                # get gradient from loss 
                grad = tape.gradient(loss, model.trainable_variables)

            # get acc
            ta = tf.keras.metrics.Accuracy()
            ta.update_state(tmp_train_y, tf.round(tf.nn.sigmoid(train_pred)))
            train_acc = ta.result().numpy()
        
            print("batch train_acc:", train_acc)

            # optimize the weights
            optimizer.apply_gradients(zip(grad, model.trainable_variables))

            epoch_losses.append(loss.numpy())
            epoch_accs.append(train_acc)

        train_losses.append(sum(epoch_losses)/len(epoch_losses))
        train_accs.append(sum(epoch_accs)/len(epoch_accs))
        print(f"\ntrain_loss: {train_losses[-1]}")
        print(f"train_acc: {train_accs[-1]}")

        # save memory
        grad = None
        train_pred = None

        print("------------\nvalidation\n------------")

        tot_valid_pred, tot_valid_y = None, None
        clusts_loss = []
        embs_list = []
        for clust in range(n_clusters):

            embs = model(feature_tensor[clust], cluster=clust,  training=False)
            embs_list.append(embs)

            valid_pred_p=None
            for i in range(0, len(valid_edges[clust]), BATCH_SIZE):
                tmp_valid_edges = valid_edges[clust][i: i+BATCH_SIZE]
                tmp_from = tmp_valid_edges[:,0]
                tmp_to = tmp_valid_edges[:,1]

                embs_from = tf.gather(embs, tmp_from)
                embs_to = tf.gather(embs, tmp_to)
                if(valid_pred_p is None):
                    valid_pred_p = tf.linalg.diag_part(tf.matmul(embs_from, embs_to, transpose_b=True))
                else:
                    batch_logits = tf.linalg.diag_part(tf.matmul(embs_from, embs_to, transpose_b=True))
                    valid_pred_p = tf.concat((valid_pred_p, batch_logits), -1)
            
            valid_pred_n = None
            for i in range(0, len(valid_false_edges[clust]), BATCH_SIZE):
                tmp_valid_edges = valid_false_edges[clust][i: i+BATCH_SIZE]
                tmp_from = tmp_valid_edges[:,0]
                tmp_to = tmp_valid_edges[:,1]

                embs_from = tf.gather(embs, tmp_from)
                embs_to = tf.gather(embs, tmp_to)
                if(valid_pred_n is None):
                    valid_pred_n = tf.linalg.diag_part(tf.matmul(embs_from, embs_to, transpose_b=True))
                else:
                    batch_logits = tf.linalg.diag_part(tf.matmul(embs_from, embs_to, transpose_b=True))
                    valid_pred_n = tf.concat((valid_pred_n, batch_logits), -1)

            if valid_pred_p is not None and valid_pred_n is not None:
                valid_pred = tf.concat([valid_pred_p, valid_pred_n], 0)
            elif valid_pred_n is not None:
                valid_pred = valid_pred_n
            else:
                valid_pred = valid_pred_p

            valid_y = [1]*len(valid_edges[clust]) + [0]*len(valid_false_edges[clust])
            valid_y = tf.convert_to_tensor(valid_y, dtype=tf.float32)

            clust_loss = topological_loss(valid_y, valid_pred)
            clusts_loss.append(clust_loss)
            
            va = tf.keras.metrics.Accuracy()
            va.update_state(valid_y, tf.round(tf.nn.sigmoid(valid_pred)))
            clust_valid_acc = va.result().numpy()

            print(f"clust_loss {clust} : {clust_loss}")
            print(f"clust_acc {clust} : {clust_valid_acc}")

            if tot_valid_pred is None:
                tot_valid_pred = valid_pred
                tot_valid_y = valid_y
            else:
                tot_valid_pred = tf.concat([tot_valid_pred, valid_pred], -1)
                tot_valid_y = tf.concat([tot_valid_y, valid_y], -1)

        # VALIDATION OF EDGES AMONG DIFFERENT CLUSTERS
        print("----validation different clusters----")
        for cluster_1 in range(n_clusters):
            for cluster_2 in range(cluster_1+1, n_clusters):
                embs_from, embs_to = embs_list[cluster_1], embs_list[cluster_2] 

                tmp_valid_edges_1 = intra_valid_edges[:,2]==cluster_1
                tmp_valid_edges_2 = intra_valid_edges[:,3]==cluster_2
                tmp_valid_edges_1_2 = tmp_valid_edges_1 * tmp_valid_edges_2

                # get the right edges from clust1 to clust2
                tmp_valid_edges_1_2 = intra_valid_edges[tmp_valid_edges_1_2]
                
                valid_pos_pred = None
                if(tmp_valid_edges_1_2.shape[0] > 0):
                    valid_pos_pred = train_preds(embs_list[cluster_1], embs_list[cluster_2], tmp_valid_edges_1_2)

                tmp_valid_edges_1 = intra_valid_false_edges[:,2]==cluster_1
                tmp_valid_edges_2 = intra_valid_false_edges[:,3]==cluster_2
                tmp_valid_edges_1_2 = tmp_valid_edges_1 * tmp_valid_edges_2

                # get the right edges from clust1 to clust2
                tmp_valid_edges_1_2 = intra_valid_false_edges[tmp_valid_edges_1_2]
                
                valid_neg_pred = None
                if(tmp_valid_edges_1_2.shape[0] > 0):
                    valid_neg_pred = train_preds(embs_list[cluster_1], embs_list[cluster_2], tmp_valid_edges_1_2)
                
                clust_valid_pred = None
                clust_tmp_valid_y = None
                if(valid_neg_pred is not None and valid_pos_pred is not None):
                    # concatenate the results
                    clust_valid_pred = tf.concat((valid_pos_pred, valid_neg_pred), -1)
                    clust_tmp_valid_y = tf.concat([tf.ones(valid_pos_pred.shape[0]),tf.zeros(valid_neg_pred.shape[0])], -1)

                elif(valid_pos_pred is not None):
                    clust_valid_pred = valid_pos_pred
                    clust_tmp_valid_y = tf.ones(valid_pos_pred.shape[0])

                elif(valid_neg_pred is not None):
                    clust_valid_pred = valid_neg_pred
                    clust_tmp_valid_y = tf.ones(valid_neg_pred.shape[0])
                else:
                    continue

                if tot_valid_pred is None:
                    tot_valid_pred = clust_valid_pred
                    tot_valid_y = clust_tmp_valid_y

                else:
                    tot_valid_pred = tf.concat((tot_valid_pred, clust_valid_pred), -1)
                    tot_valid_y = tf.concat((tot_valid_y, clust_tmp_valid_y), -1)

        #loss_weights = tf.convert_to_tensor([[-0.1]*len(valid_edges[clust]) + [1]*len(valid_false_edges[clust])])

        print("valid truth shape", tot_valid_y.shape)
        print("valid pred shape", tot_valid_pred.shape)

        #valid_loss = topological_loss(tot_valid_y, tot_valid_pred)
        valid_loss = tf.reduce_mean(clusts_loss)

        va = tf.keras.metrics.Accuracy()
        va.update_state(tot_valid_y, tf.round(tf.nn.sigmoid(tot_valid_pred)))
        valid_acc = va.result().numpy()

        print(f"valid_loss: {valid_loss.numpy()}")
        print(f"valid_acc: {valid_acc}")
        print(f"valid_losses_len: {len(valid_losses)}")
        print(f"patience: {patience}")
        
        valid_loss_np = valid_loss.numpy()

        if(len(valid_losses) > 0 and min(valid_losses) < valid_loss_np):
            print("increase patience")
            print(min(valid_losses), valid_loss_np)
            patience += 1
        else:
            print("zero patience")
            if(len(valid_losses)>0):
                print(min(valid_losses), valid_loss_np)
            patience = 0
        print(patience)
        valid_losses.append(valid_loss.numpy())
        valid_accs.append(valid_acc)
        
        if(patience > PATIENCE):
            print("breaking")
            break
        print("#"*20)

    model_name  = "first" if share_first else "last"

    plot(train_losses, valid_losses, "loss", f"shared_{model_name}")
    plot(train_accs, valid_accs, "acc", f"shared_{model_name}")

    with open("plots/n_epochs.txt", "a") as fout:
        fout.write(f"{model_name}, {DATASET_NAME}, {n_epochs}\n")

    return model

def plot(train, valid, name, clust_id):
    plt.clf()
    plt.plot(train, label=f"train_{name}")
    plt.plot(valid, label=f"valid_{name}")
    plt.ylabel(f"{name}s")
    plt.xlabel("epochs")
    plt.legend([f"train_{name}", f"valid_{name}"])
    plt.savefig(f"plots/{name}_{clust_id}.png")


def get_preds(embs, edges_pos, edges_neg, embs_1=None):
    
    embs_1 = embs if embs_1 is None else embs_1

    valid_pred_p=None
    for i in range(0, len(edges_pos), BATCH_SIZE):
        tmp_valid_edges = edges_pos[i: i+BATCH_SIZE]
        tmp_from = tmp_valid_edges[:,0]
        tmp_to = tmp_valid_edges[:,1]

        embs_from = tf.gather(embs, tmp_from)
        embs_to = tf.gather(embs_1, tmp_to)
        if(valid_pred_p is None):
            valid_pred_p = tf.linalg.diag_part(tf.matmul(embs_from, embs_to, transpose_b=True))
        else:
            batch_logits = tf.linalg.diag_part(tf.matmul(embs_from, embs_to, transpose_b=True))
            valid_pred_p = tf.concat((valid_pred_p, batch_logits), -1)
    
    valid_pred_n = None
    for i in range(0, len(edges_neg), BATCH_SIZE):
        tmp_valid_edges = edges_neg[i: i+BATCH_SIZE]
        tmp_from = tmp_valid_edges[:,0]
        tmp_to = tmp_valid_edges[:,1]

        embs_from = tf.gather(embs, tmp_from)
        embs_to = tf.gather(embs_1, tmp_to)
        if(valid_pred_n is None):
            valid_pred_n = tf.linalg.diag_part(tf.matmul(embs_from, embs_to, transpose_b=True))
        else:
            batch_logits = tf.linalg.diag_part(tf.matmul(embs_from, embs_to, transpose_b=True))
            valid_pred_n = tf.concat((valid_pred_n, batch_logits), -1)

    if(valid_pred_n is not None and valid_pred_p is not None):
        preds_all = tf.concat([valid_pred_p, valid_pred_n], 0)
        labels_all = np.hstack([np.ones(len(edges_pos)), np.zeros(len(edges_neg))])
    elif(valid_pred_p is not None):
        preds_all = valid_pred_p
        labels_all = np.ones(len(edges_pos))
    else:
        preds_all = valid_pred_n
        labels_all = np.ones(len(edges_neg))

    return preds_all, labels_all

def get_scores(labels_all, preds_all, dataset, clust):    
    roc_score = metrics.roc_auc_score(labels_all, preds_all)
    ap_score = metrics.average_precision_score(labels_all, preds_all)

    cms = []

    for t in [0.5, 0.6, 0.7]:

        plt.clf()
        cm = metrics.confusion_matrix(labels_all, np.where(preds_all > t, 1, 0))
        
        cms.append(cm)

        df_cm = pd.DataFrame(cm, index = [i for i in "01"],
                    columns = [i for i in "01"])
        plt.figure(figsize = (10,7))
        sn.heatmap(df_cm, annot=True, fmt='g')
        plt.xlabel("predicted labels")
        plt.ylabel("true labels")
        plt.savefig(f"plots/conf_matrix_{dataset}_{clust}_{t}.png")
        plt.close()

    print(f"roc_score: {roc_score}")
    print(f"ap_score: {ap_score}")

    roc_curve_plot(labels_all, preds_all, roc_score, dataset, clust)

    return roc_score, ap_score, cms

def roc_curve_plot(testy, y_pred, roc_score, dataset, clust):
    
    lr_fpr, lr_tpr, _ = metrics.roc_curve(testy, y_pred)
    
    plt.clf()
    plt.plot(lr_fpr, lr_tpr, label='ROC AUC=%.3f' % (roc_score))

    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')

    plt.legend()
    plt.savefig(f"plots/roc_score_{dataset}_{clust}")


# test_p: [[from, to, clust_from, clust_to]]
def test_in_between_edges(features_list, model, test_p, test_n, dataset, model_name):

    features = [convert_sparse_matrix_to_sparse_tensor(features) for features in features_list]

    embs_list = [model(features[i], cluster=i, training=False) for i in range(len(features_list))]

    labels_all, preds_all = None, None

    for clust_1 in range(len(features_list)):
        for clust_2 in range(len(features_list)):   
            # skip edges inside the clusters
            if clust_1 == clust_2:
                continue

            test_p_clust_1 = test_p[test_p[:, 2] == clust_1]
            test_p_clust_1_2 = test_p_clust_1[test_p_clust_1[:, 3] == clust_2]
            test_p_clust_1_2 = test_p_clust_1_2[:, 0:2]

            test_n_clust_1 = test_n[test_n[:, 2] == clust_1]
            test_n_clust_1_2 = test_n_clust_1[test_n_clust_1[:, 3] == clust_2]
            test_n_clust_1_2 = test_n_clust_1_2[:, 0:2]
                        
            preds, labels = get_preds(embs_list[clust_1], test_p_clust_1_2, test_n_clust_1_2, embs_1=embs_list[clust_2])

            if(labels_all is None):
                preds_all = preds
                labels_all = labels
            else:
                preds_all = tf.concat([preds_all, preds], 0)
                labels_all = np.hstack([labels_all, labels])

    auc, ap, cms = get_scores(labels_all, preds_all, dataset, f"{model_name}_intra_clust")

    return ap, auc, cms



def test(features_list, model, test_p, test_n, dataset, clust):
    #n_nodes = features.shape[0]
    features = [convert_sparse_matrix_to_sparse_tensor(features) for features in features_list]

    embs_list = [model(features[i], cluster=i, training=False) for i in range(len(features_list))]

    first_layer_output = [model.conv_1[i](features[i]).numpy() for i in range(len(features_list))]

    for clust in range(len(features_list)):
        np.savetxt(f"first_layer_out/{len(features_list)}/{dataset}_shared_{len(features_list)}_{clust}.csv", first_layer_output[clust], delimiter=",")
        check = np.loadtxt(f"first_layer_out/{len(features_list)}/{dataset}_shared_{len(features_list)}_{clust}.csv", delimiter=',')

        print(np.sum(check - first_layer_output[clust]))

    labels_all, preds_all = None, None

    for i in range(len(features_list)):

        preds, labels = get_preds(embs_list[i], test_p[i], test_n[i])

        if(labels_all is None):
            preds_all = preds
            labels_all = labels
        else:
            preds_all = tf.concat([preds_all, preds], 0)
            labels_all = np.hstack([labels_all, labels])

    auc, ap, cms = get_scores(labels_all, preds_all, dataset, clust)

    return ap, auc, cms

# compute Ã = D^{1/2}(A+I)D^{1/2}
def compute_adj_norm(adj):
    
    adj_I = adj + sp.eye(adj.shape[0])

    D = np.sum(adj_I, axis=1)
    D_power = sp.diags(np.asarray(np.power(D, -0.5)).reshape(-1))

    adj_norm = D_power.dot(adj_I).dot(D_power)

    return adj_norm

def complete_graph(node_to_clust):
    clust = "complete"
    adj_train, features, test_matrix, valid_matrix  = get_complete_data(DATASET_NAME, N_CLUSTERS, leave_intra_clust_edges=LEAVE_INTRA_CLUSTERS)

    train_edges, _, _ = sparse_to_tuple(adj_train)
    test_edges, _, _ = sparse_to_tuple(test_matrix)
    valid_edges, _, _ = sparse_to_tuple(valid_matrix)

    data = [1]*(len(train_edges) + len(test_edges) + len(valid_edges))
    indexes = np.concatenate((train_edges,test_edges, valid_edges), 0)

    complete_adj = csr_matrix((data, (indexes[:,0], indexes[:,1])), shape = adj_train.shape)

    node_to_clust_tmp = node_to_clust
    if(LEAVE_INTRA_CLUSTERS):
        # so the false edges that we will build are also between the clusters
        node_to_clust_tmp = None

    false_edges = get_false_edges(complete_adj, test_edges.shape[0] + valid_edges.shape[0], node_to_clust_tmp)
    valid_false_edges = false_edges[:valid_edges.shape[0]]
    test_false_edges = false_edges[valid_edges.shape[0]:]


    # since get_test_edges returns a triu, we sum to its transpose 
    adj_train = adj_train + adj_train.T

    # get normalized adj
    adj_train_norm = compute_adj_norm(adj_train)
    
    print(f"valid_edges: {valid_edges.shape[0]}")
    print(f"valid_false_edges: {valid_false_edges.shape[0]}")

    start_time = time.time()
    
    # start training
    #model = complete_train(features, adj_train, adj_train_norm, train_edges, valid_edges, valid_false_edges, clust, node_to_clust=node_to_clust)

    execution_time = time.time() - start_time

    #model.save_weights(f"weights/{DATASET_NAME}_{clust}")

    test_ap, test_auc = 0, 0

    same_clust_test = [node_to_clust[edge[0]] == node_to_clust[edge[1]] for edge in test_edges]
    same_clust_false_test = [node_to_clust[edge[0]] == node_to_clust[edge[1]] for edge in test_false_edges]

    diff_clust_test = list(map(lambda x: not x, same_clust_test))
    diff_clust_false_test = list(map(lambda x: not x, same_clust_false_test))


    #_, _, _ = complete_test(features, model, test_edges[same_clust_test], test_false_edges[same_clust_false_test], DATASET_NAME, clust+"_same_clust")
    #_, _, _ = complete_test(features, model, test_edges[diff_clust_test], test_false_edges[diff_clust_false_test], DATASET_NAME, clust+"_intra_clust")
    #test_ap, test_auc, cms = complete_test(features, model, test_edges, test_false_edges, DATASET_NAME, clust)

    #f1s, precs, recs = [], [], []
    #for cm in cms:
    #tp, fp, fn = cm[1][1], cm[0][1], cm[1][0]
    #precs.append(tp/(tp+fp))
    #recs.append(tp/(tp+fn))
    #f1s.append(2*precs[-1]*recs[-1]/(precs[-1]+recs[-1])) 

    """with open(f"results/{DATASET_NAME}_complete_model.txt", "a") as fout:
        fout.write(f"precs: {precs}\n")
        fout.write(f"recs: {recs}\n")
        fout.write(f"f1s: {f1s}\n")
        fout.write(f"time: {execution_time}\n")
        fout.write("-"*10 + "\n")"""


    
    test_ones = [1]*test_false_edges.shape[0]
    valid_ones = [1]*valid_false_edges.shape[0]

    test_false_matrix = csr_matrix((test_ones, (test_false_edges[:,0], test_false_edges[:,1])), adj_train.shape)
    valid_false_matrix = csr_matrix((valid_ones, (valid_false_edges[:,0], valid_false_edges[:,1])), adj_train.shape)

    return test_false_matrix, valid_false_matrix, test_ap, test_auc, execution_time

def get_intra_edges(matrix, clust_to_node):
    print("start get_intra_edges")
    n_clusters = len(clust_to_node.keys())
    

    intra_matrix = sp.csr_matrix(matrix.shape)
    for i in range(n_clusters):
        in_clust = clust_to_node[i]

        in_clust_0 = []
        for i in range(len(in_clust)):
            in_clust_0 += in_clust
        

        in_clust_0 = np.array(in_clust_0)
        in_clust_1 = np.array([-1] * (len(in_clust)*len(in_clust)))
        base_idx = np.array([0 + len(in_clust)*i for i in range(len(in_clust))])    

        for i in range(len(in_clust)):
            in_clust_1[base_idx] = in_clust[i]
            base_idx += 1

        tmp_matrix = matrix.copy()
        print(in_clust_0.shape)
        print(in_clust_1.shape)

        tmp_matrix[in_clust_0, in_clust_1] = 0

        intra_matrix[in_clust, :] = tmp_matrix[in_clust, :]
        intra_matrix[:, in_clust] = tmp_matrix[:, in_clust]
    print("end get_intra_edges")
    
    return intra_matrix

if __name__ == "__main__":
    
    # load data : adj, features, node labels and number of clusters
    adjs, features_, tests, valids, clust_to_node, node_to_clust, com_idx_to_clust_idx = load_data(DATASET_NAME)

    # turn the dictionary into a list of features
    features_ = [features_[i] for i in range(len(features_))]

    # train the complete graph in order to compare the results
    # get also the test false edges
    test_false_matrix, valid_false_matrix, test_ap, test_auc, execution_time = complete_graph(node_to_clust)

    n_test_edges = []
    n_valid_edges = []

    test_aps = [test_ap]
    test_aucs = [test_auc]
    execution_times = [execution_time]

    test_intra_cluster_aps = []
    test_intra_cluster_aucs = []

    subset_lenghts = []
    
    adj_train_list, adj_train_norm_list = [], []
    train_edges_list, test_edges_list, valid_edges_list = [], [], []
    test_false_edges_list, valid_false_edges_list = [], []

    for clust in range(len(adjs)):
        print("\n")

        adj_train = adjs[clust]

        train_edges, _, _ = sparse_to_tuple(adj_train)
        test_edges, _, _ = sparse_to_tuple(tests[clust])
        valid_edges, _, _ = sparse_to_tuple(valids[clust])
        
        n_test_edges.append(test_edges.shape[0])
        n_valid_edges.append(valid_edges.shape[0])


        test_false_matrix_c = test_false_matrix[clust_to_node[clust], :]
        test_false_matrix_c = test_false_matrix_c[:, clust_to_node[clust]]

        valid_false_matrix_c = valid_false_matrix[clust_to_node[clust], :]
        valid_false_matrix_c = valid_false_matrix_c[:, clust_to_node[clust]]
        
        test_false_edges, _, _ = sparse_to_tuple(test_false_matrix_c)
        valid_false_edges, _, _ = sparse_to_tuple(valid_false_matrix_c)

        subset_lenghts.append((len(valid_edges), len(valid_false_edges), len(test_edges), len(test_false_edges)))

        # since get_test_edges returns a triu, we sum to its transpose 
        adj_train = adj_train + adj_train.T

        # get normalized adj
        adj_train_norm = compute_adj_norm(adj_train)

        adj_train_list.append(adj_train)
        adj_train_norm_list.append(adj_train_norm)

        train_edges_list.append(train_edges)
        valid_edges_list.append(valid_edges)
        test_edges_list.append(test_edges)

        test_false_edges_list.append(test_false_edges)
        valid_false_edges_list.append(valid_false_edges)

    for share_first in [False]:
        
        adj_train, _, test_matrix, valid_matrix  = get_complete_data(DATASET_NAME, N_CLUSTERS, leave_intra_clust_edges=LEAVE_INTRA_CLUSTERS)

        intra_train_matrix = get_intra_edges(adj_train, clust_to_node)
        intra_valid_matrix = get_intra_edges(valid_matrix, clust_to_node)

        intra_train_edges, _, _ = sparse_to_tuple(intra_train_matrix)
        intra_valid_edges, _, _ = sparse_to_tuple(intra_valid_matrix)
        intra_valid_false_edges, _, _ = sparse_to_tuple(valid_false_matrix)
    

        start_time = time.time()
        #model = train(features_, adj_train_list, adj_train_norm_list, train_edges_list, valid_edges_list, valid_false_edges_list, share_first, node_to_clust=None)
        model = train(features_, adj_train_list, adj_train_norm_list, train_edges_list, valid_edges_list, valid_false_edges_list, 
                        intra_train_edges, intra_valid_edges, intra_valid_false_edges, adj_train, share_first, clust_to_node, com_idx_to_clust_idx, node_to_clust=node_to_clust)
        
        execution_times.append(time.time()-start_time)

        model_name = "first" if share_first else "last"

        #model.save(f"weights/{DATASET_NAME}/{N_CLUSTERS}/share_{model_name}")

        test_ap, test_auc, cms_inside = test(features_, model, test_edges_list, test_false_edges_list, DATASET_NAME, f"share_{model_name}")
        
        save_model(model, [convert_sparse_matrix_to_sparse_tensor(features) for features in features_], DATASET_NAME, f"share_{model_name}")


        test_aps.append(test_ap)
        test_aucs.append(test_auc)

        if(LEAVE_INTRA_CLUSTERS):

            clust_idxs = []
            for i in range(adj_train.shape[0]):
                clust = node_to_clust[i]
                clust_nodes = clust_to_node[clust]

                clust_idx = clust_nodes.index(i)
                clust_idxs.append(clust_idx)
            
            

            pos_edges = []
            neg_edges = []
            for clust in range(len(features_)):
                # take the edges starting from clust (and ending in any node, also the clust itself)
                # since when testing we ignore the edges among the same cluster
                tmp_test = test_matrix[clust_to_node[clust], :]
                tmp_test, _, _ = sparse_to_tuple(tmp_test)

                for edge in range(tmp_test.shape[0]):
                    # idx of the destination in the complete graph
                    to_idx_comp = tmp_test[edge][1] 
                    # idx of the destination in its clust
                    to_idx_clust = clust_idxs[to_idx_comp]
                    # clust of the destination node
                    to_clust = node_to_clust[to_idx_comp]

                    if(to_clust != clust):
                        pos_edges.append([tmp_test[edge][0], to_idx_clust, clust, to_clust])


                tmp_test_false = test_false_matrix[clust_to_node[clust], :]
                tmp_test_false, _, _ = sparse_to_tuple(tmp_test_false)

                for edge in range(tmp_test_false.shape[0]):
                    # idx of the destination in the complete graph
                    to_idx_comp = tmp_test_false[edge][1] 
                    # idx of the destination in its clust
                    to_idx_clust = clust_idxs[to_idx_comp]
                    # clust of the destination node
                    to_clust = node_to_clust[to_idx_comp]

                    if(to_clust != clust):
                        neg_edges.append([tmp_test_false[edge][0], to_idx_clust, clust, to_clust])

            pos_edges = np.array(pos_edges)
            neg_edges = np.array(neg_edges)

            print("pos_edges", pos_edges.shape)

            test_ap, test_auc, cms_between = test_in_between_edges(features_, model, pos_edges, neg_edges, DATASET_NAME, "share_first" if share_first else "share_last")

            test_intra_cluster_aps.append(test_ap)
            test_intra_cluster_aucs.append(test_auc)

            model_name = "share_first" if share_first else "share_last"
            model_name += "_all_test"
            
            print("before for")

            f1s, precs, recs = [], [], []

            ts = [0.5, 0.6, 0.7]
            for t in range(len(cms_inside)):
                print("inside for")
                
                cm = cms_inside[t] + cms_between[t]

                tp, fp, fn = cm[1][1], cm[0][1], cm[1][0]
                precs.append(tp/(tp+fp))
                recs.append(tp/(tp+fn))
                f1s.append(2*precs[-1]*recs[-1]/(precs[-1]+recs[-1]))

                df_cm = pd.DataFrame(cm, index = [i for i in "01"],
                    columns = [i for i in "01"])
                plt.figure(figsize = (10,7))
                sn.heatmap(df_cm, annot=True, fmt='g')
                plt.xlabel("predicted labels")
                plt.ylabel("true labels")
                plt.savefig(f"plots/conf_matrix_{DATASET_NAME}_{model_name}_{ts[t]}.png")
                plt.close()
            
            with open(f"results/{DATASET_NAME}_{model_name}.txt", "a") as fout:
                fout.write(f"precs: {precs}\n")
                fout.write(f"recs: {recs}\n")
                fout.write(f"f1s: {f1s}\n")
                fout.write(f"times: {execution_times[-1]}\n")
                fout.write("-"*10 + "\n")


    print(f"test ap: {test_aps}")
    print(f"test auc: {test_aucs}")
    print(f"test_intra_cluster_aps: {test_intra_cluster_aps}")
    print(f"test_intra_cluster_aucs: {test_intra_cluster_aucs}")
    print()
    print(f"n_test_edges: {n_test_edges}")
    print(f"n_valid_edges: {n_valid_edges}")
    print(f"execution_times: {execution_times}")