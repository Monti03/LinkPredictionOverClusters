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
from sklearn import metrics


import tensorflow as tf

from tensorflow.python.ops.gen_linalg_ops import matrix_triangular_solve

from constants import *
from utils.data import load_data, get_test_edges, get_false_edges, sparse_to_tuple, get_complete_cora_data, get_complete_data
from networks.gae_model import GAEModel    
from loss import total_loss, topological_loss
from utils.utils import convert_sparse_matrix_to_sparse_tensor, compute_adj_norm, plot_cf_matrix, plot

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
TEST = False

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
        N_CLUSTERS = int(current_value)
    elif current_argument in ("--test"):    
        EPOCHS = 2
        TEST = True
        print("Testing train: n epochs:", EPOCHS)


def get_predictions(embs_from, edges, embs_to=None):
    if embs_to is None:
        embs_to = embs_from

    predictions = None
    for i in range(0, len(edges), BATCH_SIZE):
        
        tmp_edges = edges[i: i+BATCH_SIZE]
        tmp_from = tmp_edges[:,0]
        tmp_to = tmp_edges[:,1]

        tmp_embs_from = tf.gather(embs_from, tmp_from)
        tmp_embs_to = tf.gather(embs_to, tmp_to)

        if(predictions is None):
            predictions = tf.linalg.diag_part(tf.matmul(tmp_embs_from, tmp_embs_to, transpose_b=True))
        else:
            batch_logits = tf.linalg.diag_part(tf.matmul(tmp_embs_from, embs_to, transpose_b=True))
            predictions = tf.concat((predictions, batch_logits), -1)

    return predictions

def get_predictions_and_labels(embs_from, edges_pos, edges_neg, embs_to=None):
    
    edges_pos = np.array(edges_pos)
    edges_neg = np.array(edges_neg)

    preds_pos = get_predictions(embs_from, edges_pos, embs_to=embs_to)
    preds_neg = get_predictions(embs_from, edges_neg, embs_to=embs_to)


    if (preds_pos is not None and preds_neg is not None):
        preds_all = tf.concat([preds_pos, preds_neg], 0)
    elif(preds_neg is not None):
        preds_all = preds_neg
    else: 
        preds_all = preds_pos

    labels_all = np.hstack([np.ones(len(edges_pos)), np.zeros(len(edges_neg))])

    return preds_all, labels_all


def get_predictions_and_labels_from_same_clusts_avg_couples(features_tensors, models, test_p, test_n, clust_to_nodes):
    # test the edges inside a cluster with all the models that consider that cluster
    counter = 0 
    clust_to_models = {}
    
    labels_all, preds_all = None, None

    n_original_clusters = (1 + math.sqrt(1 + 8*(len(features_tensors))))/2
    assert n_original_clusters == int(n_original_clusters)

    for clust_1 in range(int(n_original_clusters)):
        for clust_2 in range(clust_1+1, int(n_original_clusters)):
            
            if clust_1 not in clust_to_models:
                clust_to_models[clust_1] = []
            if clust_2 not in clust_to_models:
                clust_to_models[clust_2] = []

            clust_to_models[clust_1].append(counter)
            clust_to_models[clust_2].append(counter)

            counter += 1


    for clust in range(int(n_original_clusters)):
        # get positive edges to predict
        test_p_1 = test_p[:,2] == clust
        test_p_2 = test_p[:,3] == clust
        test_p_1_2_comp = test_p[test_p_1*test_p_2]
        

        # get negative edges to predict
        test_n_1 = test_n[:,2] == clust
        test_n_2 = test_n[:,3] == clust
        test_n_1_2 = test_n[test_n_1*test_n_2]
        test_n_1_2_comp = test_n_1_2[:, :2]

        preds, labels  = None, None

        for idx in clust_to_models[clust]:
            embs = models[idx](features_tensors[idx])
            nodes = clust_to_nodes[idx]

            test_p_1_2 = [[nodes.index(edge[0]), nodes.index(edge[1])] for edge in test_p_1_2_comp]
            test_n_1_2 = [[nodes.index(edge[0]), nodes.index(edge[1])] for edge in test_n_1_2_comp]

        
            tmp_preds, tmp_labels = get_predictions_and_labels(embs, test_p_1_2, test_n_1_2)
            
            if(preds is None):
                preds = tmp_preds
                labels = tmp_labels
            else:
                preds += tmp_preds

        preds = preds / len(clust_to_models[clust])
        if(labels_all is None):
            preds_all = preds
            labels_all = labels
        else:
            preds_all = tf.concat([preds_all, preds], 0)
            labels_all = np.hstack([labels_all, labels])
    return preds_all, labels_all

def get_predictions_and_labels_from_diff_clusts(features_tensors, models, test_p, test_n):
    # identifies the idx of the couple model that is resposable
    # for the couple clust_1, clust_2
    counter = 0

    labels_all, preds_all = None, None

    for clust_1 in range(N_CLUSTERS):
        for clust_2 in range(clust_1+1, N_CLUSTERS):
            # get positive edges to predict
            test_p_1 = test_p[:,2] == clust_1
            test_p_2 = test_p[:,3] == clust_2
            test_p_1_2 = test_p[test_p_1*test_p_2]
            test_p_1_2 = test_p_1_2[:, :2]

            # get negative edges to predict
            test_n_1 = test_n[:,2] == clust_1
            test_n_2 = test_n[:,3] == clust_2
            test_n_1_2 = test_n[test_n_1*test_n_2]
            test_n_1_2 = test_n_1_2[:, :2]
            
            # get the embeddings
            embs = models[counter](features_tensors[counter])

            preds, labels = get_predictions_and_labels(embs, test_p_1_2, test_n_1_2)

            if(labels_all is None and preds is not None):
                preds_all = preds
                labels_all = labels
            elif preds is not None:
                preds_all = tf.concat([preds_all, preds], 0)
                labels_all = np.hstack([labels_all, labels])

            counter += 1
    return preds_all, labels_all

def convert_edges_to_clust_idxs(edges, clust_to_node, node_to_clust):
    if len(edges) == 0:
        return np.array([])
    converted_edges = []
    clust_single_first = node_to_clust[edges[0][0]]
    
    print("len:", len(clust_to_node[clust_single_first]))
    
    for edge in edges:
        from_idx, to_idx = edge[0], edge[1] 
        clust_single = node_to_clust[from_idx]
        
        assert clust_single == node_to_clust[to_idx] and clust_single_first == clust_single

        from_idx_clust, to_idx_clust = clust_to_node[clust_single].index(from_idx), clust_to_node[clust_single].index(to_idx) 
        converted_edges.append([from_idx_clust, to_idx_clust])
    
    return np.array(converted_edges)

"""
    test_p: [[idx_1, idx_2, clust_1, clust_2]]
        in the case of test without single models, we have that idx_1 and idx_2 are the indeces of the nodes
        inside the model trained over clust_1 clust_2

        in the case of test with single models, we have that idx_1 and idx_2 are the indices of the nodes 
        inside the model trained only over clust_1 = clust_2
    
    same for test_n
"""
def test(features, models, test_p, test_n, dataset, clust_to_nodes = None, single_models = None, single_features = None, single_clust_to_node = None, single_node_to_clust = None):
    features_tensors = [convert_sparse_matrix_to_sparse_tensor(features[i]) for i in range(len(features))] 

    single_models_condition = single_models is not None and single_features is not None and single_clust_to_node is not None and single_node_to_clust is not None
    couple_models_condition = clust_to_nodes is not None

    assert single_models_condition or couple_models_condition
     
    # predict edges between different clusters
    print("different cluster prediction")
    preds_all, labels_all = get_predictions_and_labels_from_diff_clusts(features_tensors, models, test_p, test_n)

    
    if(couple_models_condition):
        print("same cluster prediction + couple models")
        preds, labels = get_predictions_and_labels_from_same_clusts_avg_couples(features_tensors, models, test_p, test_n, clust_to_nodes)
        
        preds_all = tf.concat([preds_all, preds], 0)
        labels_all = np.hstack([labels_all, labels])

    elif(single_models_condition):
        print("same cluster prediction + single models")

        # test the edges inside a cluster with a model trained only for that cluster
        
        single_features = [convert_sparse_matrix_to_sparse_tensor(single_features[i]) for i in range(len(single_features))]
        
        for clust in range(len(single_models)):
            # get positive edges to predict
            test_p_1 = test_p[:,2] == clust
            test_p_2 = test_p[:,3] == clust
            test_p_1_2 = test_p[test_p_1*test_p_2]
            test_p_1_2 = test_p_1_2[:, :2]

            test_p_clust = convert_edges_to_clust_idxs(test_p_1_2, clust_to_node_single, node_to_clust_single)

            # get negative edges to predict
            test_n_1 = test_n[:,2] == clust
            test_n_2 = test_n[:,3] == clust
            test_n_1_2 = test_n[test_n_1*test_n_2]
            test_n_1_2 = test_n_1_2[:, :2]

            test_n_clust = convert_edges_to_clust_idxs(test_n_1_2, clust_to_node_single, node_to_clust_single)

            # since the couples clust are sorted in descending order, I have to take the real value of clust
            # for the single models

            # I am sure that this is the clust that every node in test_p_clust and test_n_clust shares cause of an
            # assert in the convert_edges_to_clust_idxs function
            real_clust = node_to_clust_single[test_p_1_2[0][0]]

            embs = single_models[real_clust](single_features[real_clust])

            preds, labels = get_predictions_and_labels(embs, test_p_clust, test_n_clust)

            preds_all = tf.concat([preds_all, preds], 0)
            labels_all = np.hstack([labels_all, labels])
    else:
        raise Exception("I couldnt test over the edges inside the same cluster")

    print(preds_all.shape, labels_all.shape)

    name = f"{dataset}_only_couples" if couple_models_condition else f"{dataset}_couple_for_diff_clusts_single_for_same_clust"
    cms = plot_cf_matrix(labels_all, preds_all, name, TEST)

    roc_score = metrics.roc_auc_score(labels_all, preds_all)
    ap_score = metrics.average_precision_score(labels_all, preds_all)



    return roc_score, ap_score, cms
    
"""
returns a np array containing for each test edge:
    n_from_couple, n_to_couple, from_clust, to_clust \
    where n_from_couple and n_to_couple are respectively the clust of the first node and the clust of the second node
    from_clust are to_clust 
        - if the cluster are different -> the idx of the two nodes wrt the couple model
        - if the cluster is the same  -> idx of the nodes in the complete graph

"""
def build_test_edges(test_edges, node_to_clust, clust_to_node, n_original_clusters):
    test_edges_clusts = []

    diff_clust, same_clust = 0, 0

    for test_edge in test_edges:
        n_from_comp = test_edge[0]
        n_to_comp = test_edge[1]

        from_clust = node_to_clust[n_from_comp]
        to_clust = node_to_clust[n_to_comp]

        if(from_clust != to_clust):
            min_clust = min(from_clust, to_clust)

            # obtain the idx of the couple model that handles such edge
            base_counter = 0.5*min_clust*(2*n_original_clusters-min_clust-1) - 1
            assert base_counter == int(base_counter)
            counter = int(base_counter) + (max(from_clust, to_clust)-min(from_clust, to_clust))

            couple_nodes = clust_to_node[counter]

            n_from_couple = couple_nodes.index(n_from_comp)
            n_to_couple = couple_nodes.index(n_to_comp)

            if(from_clust < to_clust):
                test_edges_clusts.append([n_from_couple, n_to_couple, from_clust, to_clust])
            else:
                test_edges_clusts.append([n_to_couple, n_from_couple, to_clust, from_clust])

            assert test_edges_clusts[-1][2] != test_edges_clusts[-1][3]

            diff_clust += 1

        else:
            # if dealing with edges in the same cluster by not using single models, 
            # I pass to the test the idxs of the nodes wrt the complete graph. It will than
            # associate the right index to the nodes wrt the model it will use since 
            # a cluster can be predicted by multiple models. 

            test_edges_clusts.append([n_from_comp, n_to_comp, from_clust, to_clust])

            same_clust += 1

    assert len(test_edges_clusts) ==  test_edges.shape[0]
    print("same_clust", same_clust, "diff_clust", diff_clust)
    return np.array(test_edges_clusts)



class Trainer():
    

    def __init__(self, model_name:str, batch_size:int, train_patience:int, max_epochs:int, lr:float):
        self.model_name = model_name
        self.batch_size = batch_size
        self.train_patience = train_patience
        self.max_epochs = max_epochs
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=lr)

    def prepare_data(self, adj_train:sp.csr_matrix, features:sp.csr_matrix, valid_edges:np.ndarray, valid_false_edges:np.ndarray):
        self.train_accs, self.train_losses = [], []

        self.valid_accs, self.valid_losses = [], [] 

        # get the normalized adj matrix
        adj_train_norm = compute_adj_norm(adj_train)

        # convert the normalized adj and the features to tensors
        self.adj_train_norm_tensor = convert_sparse_matrix_to_sparse_tensor(adj_train_norm)
        self.feature_tensor = convert_sparse_matrix_to_sparse_tensor(features)

        # get the positive train edges
        self.train_edges, _, _ = sparse_to_tuple(adj_train)
        
        # train false edges are chosen here because we can also considere false 
        # some edges that are in valid or test set as true edges
        # get the train false edges
        self.train_false_edges = get_false_edges(adj_train, len(self.train_edges))

        self.valid_edges = valid_edges
        self.valid_false_edges = valid_false_edges

        # define the model
        self.model = GAEModel(self.adj_train_norm_tensor)

            

    def __batch_loop(self, training:bool, batch_edges:np.ndarray, batch_false_edges:np.ndarray):
        # forward pass
        embs = self.model(self.feature_tensor, training=training)
        
        node_from = batch_edges[:,0]
        node_to = batch_edges[:,1]

        # get the embedding of the nodes inside the batch_edges
        embs_from = tf.gather(embs, node_from)
        embs_to = tf.gather(embs, node_to)

        # get the predictions of batch_edges
        pos_preds = tf.linalg.diag_part(tf.matmul(embs_from, embs_to, transpose_b=True))
        
        # same for the false edges
        node_from = batch_false_edges[:,0]
        node_to = batch_false_edges[:,1]

        embs_from = tf.gather(embs, node_from)
        embs_to = tf.gather(embs, node_to)

        neg_preds = tf.linalg.diag_part(tf.matmul(embs_from, embs_to, transpose_b=True))
    
        # concatenate the predictions and build the gt
        predictions = tf.concat((pos_preds, neg_preds), 0)
        batch_gt = tf.concat([tf.ones(pos_preds.shape[0]),tf.zeros(neg_preds.shape[0])], 0)

        # get loss
        loss = topological_loss(batch_gt, predictions)

        # get acc
        ta = tf.keras.metrics.Accuracy()
        ta.update_state(batch_gt, tf.round(tf.nn.sigmoid(predictions)))
        acc = ta.result().numpy()

        return acc, loss


    def __train_loop(self):

        # shuffle data in order to have different batches in different epochs
        np.random.shuffle(self.train_edges)      
        np.random.shuffle(self.train_false_edges)
        
        batch_losses, batch_accs = [], []

        for i in range(0, min(len(self.train_edges), self.train_false_edges.shape[0]), self.batch_size):

            batch_edges = self.train_edges[i: i+self.batch_size]    
            batch_false_edges = self.train_false_edges[i: i+self.batch_size]            
        
            with tf.GradientTape() as tape:  
                acc, loss = self.__batch_loop(True, batch_edges, batch_false_edges)
        
                grad = tape.gradient(loss, self.model.trainable_variables)
            
                # optimize the weights
                self.optimizer.apply_gradients(zip(grad, self.model.trainable_variables))

                batch_losses.append(loss.numpy())
                batch_accs.append(acc)

        self.train_losses.append(sum(batch_losses)/len(batch_accs))
        self.train_accs.append(sum(batch_accs)/len(batch_accs))


    def __valid_loop(self):        
        acc, loss = self.__batch_loop(False, self.valid_edges, self.valid_false_edges)

        self.valid_accs.append(acc)
        self.valid_losses.append(loss.numpy())

        return loss

    def train(self):
        start_time = time.time()
        patience, n_epoch = 0, 0
        while patience <= self.train_patience and n_epoch < self.max_epochs :
            self.__train_loop()
            loss = self.__valid_loop()
            if (loss > min(self.valid_losses)):
                patience += 1
            else:
                patience = 0

            n_epoch += 1    

            print(f"Epoch: {n_epoch}") 
            print(f"TRAIN_RESULTS:\n\ttrain_acc: {self.train_accs[-1]}\t train_loss: {self.train_losses[-1]}")
            print(f"VALID_RESULTS:\n\tvalid_acc: {self.valid_accs[-1]}\t valid_loss: {self.valid_losses[-1]}")
            print(f"------------------")
        self.execution_time = time.time() - start_time


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

def get_per_cluster_false_edges(valid_false_matrix, clust_to_node):
    valid_false_edges_list = []
    
    for clust in range(len(clust_to_node)):

        # get the sub matrix containing only the nodes inside the cluster
        clust_valid_false_matrix = valid_false_matrix[clust_to_node[clust], :][:, clust_to_node[clust]]
        
        # get the false edges
        valid_false_edges, _, _ = sparse_to_tuple(clust_valid_false_matrix)

        valid_false_edges_list.append(valid_false_edges)
    
    return valid_false_edges_list    

def train_all_the_models(data:tuple, model_type:str, valid_false_edges_list:list):
    adjs, features, _, valids, _, _, _ = data

    # train each on each cluster
    models, execution_times = [], []
    for cluster_id in range(len(adjs)):
        # define the trainer
        trainer = Trainer(f"{model_type}_{cluster_id}", batch_size=BATCH_SIZE, train_patience=PATIENCE, max_epochs=EPOCHS, lr=LR)
        
        # set up the data to train the model
        valid_edges, _, _ = sparse_to_tuple(valids[cluster_id])
        trainer.prepare_data(adjs[cluster_id], features[cluster_id], valid_edges, valid_false_edges_list[cluster_id])

        # train the model
        trainer.train()
        execution_times.append(trainer.execution_time)
        models.append(trainer.model)

    return models, execution_times

if __name__ == "__main__":
    
    # load data
    single_data = load_data(DATASET_NAME, N_CLUSTERS, get_couples = False)
    couple_data = load_data(DATASET_NAME, N_CLUSTERS, get_couples = True)
    complete_data  = get_complete_data(DATASET_NAME, N_CLUSTERS, leave_intra_clust_edges=LEAVE_INTRA_CLUSTERS)

    # node_to_clust_single and node_to_clust_couple both return values in a range [0, nclusters], indeed the only
    # difference is that in the second one we have a different order for the clusters since when we build the couples
    # we sort the clusters so that we have more couples for the small clusters than for the large ones.
    adjs_single, features_single, tests_single, valids_single, clust_to_node_single, node_to_clust_single, _ = single_data
    adjs_couple, features_couple, tests_couple, valids_couple, clust_to_node_couple, node_to_clust_couple, _ = couple_data

    # get the false edges and save them for each couple of clusters
    test_false_matrix, valid_false_matrix = get_test_valid_false_edges(complete_data)
    valid_false_edges_list_couples = get_per_cluster_false_edges(valid_false_matrix, clust_to_node_couple)
    valid_false_edges_list_singles = get_per_cluster_false_edges(valid_false_matrix, clust_to_node_single)

    # obtain the list of edges from the matrices
    _, _, test_matrix, _ = complete_data 
    test_edges, _, _ = sparse_to_tuple(test_matrix)
    test_false_edges, _, _ = sparse_to_tuple(test_false_matrix)

    test_p = build_test_edges(test_edges, node_to_clust_couple, clust_to_node_couple, N_CLUSTERS)
    test_n = build_test_edges(test_false_edges, node_to_clust_couple, clust_to_node_couple, N_CLUSTERS)

    if COUPLE_AND_SINGLE:

        couple_models, execution_times_couple = train_all_the_models(couple_data, "couple", valid_false_edges_list_couples)
        single_models, execution_times_single = train_all_the_models(single_data, "single", valid_false_edges_list_singles)

        _, _, cms_both = test(features_couple, couple_models, test_p, test_n, DATASET_NAME, single_models=single_models, single_features=features_single, single_clust_to_node=clust_to_node_single, single_node_to_clust=node_to_clust_single)
        print("test both end")
        _, _, cms_only_couples = test(features_couple, couple_models, test_p, test_n, DATASET_NAME, clust_to_nodes = clust_to_node_couple)
        print("test only couples end")
        
        f1s, precs, recs = [], [], []
        for cm in cms_both:
            tp, fp, fn = cm[1][1], cm[0][1], cm[1][0]
            precs.append(tp/(tp+fp))
            recs.append(tp/(tp+fn))
            f1s.append(2*precs[-1]*recs[-1]/(precs[-1]+recs[-1])) 

        print(f1s)

        if not TEST:
            with open(f"results/{DATASET_NAME}_couple_for_diff_clusts_single_for_same_clust.txt", "a") as fout:
                fout.write(f"precs: {precs}\n")
                fout.write(f"recs: {recs}\n")
                fout.write(f"f1s: {f1s}\n")
                fout.write(f"time: {sum(execution_times_couple) + sum(execution_times_single)}\n")
                fout.write("-"*10 + "\n")


            f1s, precs, recs = [], [], []
            for cm in cms_only_couples:
                tp, fp, fn = cm[1][1], cm[0][1], cm[1][0]
                precs.append(tp/(tp+fp))
                recs.append(tp/(tp+fn))
                f1s.append(2*precs[-1]*recs[-1]/(precs[-1]+recs[-1])) 

            with open(f"results/{DATASET_NAME}_only_couples.txt", "a") as fout:
                fout.write(f"precs: {precs}\n")
                fout.write(f"recs: {recs}\n")
                fout.write(f"f1s: {f1s}\n")
                fout.write(f"time: {sum(execution_times_couple)}\n")
                fout.write("-"*10 + "\n")


            print(execution_times_couple)
            print(execution_times_single)


    elif COUPLES_TRAIN:
        couple_models, execution_times = train_all_the_models(couple_data, "couple", valid_false_edges_list_couples)
        _, _, cms_only_couples = test(features_couple, couple_models, test_p, test_n, DATASET_NAME, clust_to_nodes = clust_to_node_couple)

    else:
        raise Exception("No modality selected")

