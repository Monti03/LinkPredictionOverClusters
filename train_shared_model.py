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
        EPOCHS = 10
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

class SharedTrainer():

    def __init__(self, share_last:bool, model_name:str, batch_size:int, train_patience:int, max_epochs:int, lr:float, n_clusters:int, dataset:str):
        self.model_name = model_name
        self.batch_size = batch_size
        self.train_patience = train_patience
        self.max_epochs = max_epochs
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
        self.n_clusters = n_clusters
        self.share_last = share_last
        self.dataset = dataset

    def prepare_data(self, adj_train_norm_list:list, features_list:list, 
        train_edges:np.ndarray, train_false_edges:np.ndarray, valid_edges:np.ndarray, valid_false_edges:np.ndarray):
        
        self.train_accs, self.train_losses = [], []
        self.valid_accs, self.valid_losses = [], [] 

        # convert the normalized adj and the features to tensors
        self.adj_train_norm_tensor = [convert_sparse_matrix_to_sparse_tensor(adj_train_norm) for adj_train_norm in adj_train_norm_list]
        self.features_tensor = [convert_sparse_matrix_to_sparse_tensor(features) for features in features_list]

        # get the positive train edges
        self.train_edges = train_edges
        self.valid_edges = valid_edges
        self.train_false_edges = train_false_edges
        self.valid_false_edges = valid_false_edges
        
        print(self.valid_edges.shape)
        print(self.valid_false_edges.shape)

        self.initialize_model()


    def initialize_model(self):
        if self.share_last:
            self.model = LastShared(self.adj_train_norm_tensor)
        else:
            self.model = FirstShared(self.adj_train_norm_tensor)
        

    def __batch_loop(self, embs_clust_from:tf.Tensor, embs_clust_to:tf.Tensor, batch_edges:np.ndarray, batch_false_edges:np.ndarray):
        
        node_from = batch_edges[:,0]
        node_to = batch_edges[:,1]

        # get the embedding of the nodes inside the batch_edges
        embs_from = tf.gather(embs_clust_from, node_from)
        embs_to = tf.gather(embs_clust_to, node_to)

        # get the predictions of batch_edges
        pos_preds = tf.linalg.diag_part(tf.matmul(embs_from, embs_to, transpose_b=True))
        
        # same for the false edges
        node_from = batch_false_edges[:,0]
        node_to = batch_false_edges[:,1]

        embs_from = tf.gather(embs_clust_from, node_from)
        embs_to = tf.gather(embs_clust_to, node_to)

        neg_preds = tf.linalg.diag_part(tf.matmul(embs_from, embs_to, transpose_b=True))
    
        # concatenate the predictions and build the gt
        predictions = tf.concat((pos_preds, neg_preds), 0)
        batch_gt = tf.concat([tf.ones(pos_preds.shape[0]),tf.zeros(neg_preds.shape[0])], 0)


        return predictions, batch_gt

    def __compute_predictions(self, batch_edges:np.ndarray, batch_false_edges:np.ndarray, clust_1:int, clust_2:int, embs_1:tf.Tensor, embs_2:tf.Tensor):
        # get the indices of the edges starting from clust1 and ending in clust2
        batch_edges_c1_c2 = (batch_edges[:,2]==clust_1) * (batch_edges[:,3]==clust_2)
        # get the right edges from clust1 to clust2 and take only the first two columns
        batch_edges_c1_c2 = batch_edges[batch_edges_c1_c2][:,:2]

        # get the indices of the false edges starting from clust1 and ending in clust2
        batch_false_edges_c1_c2 = (batch_false_edges[:,2]==clust_1) * (batch_false_edges[:,3]==clust_2)
        # get the right false edges from clust1 to clust2 and take only the first two columns
        batch_false_edges_c1_c2 = batch_false_edges[batch_false_edges_c1_c2][:,:2]

        return self.__batch_loop(embs_1, embs_2, batch_edges_c1_c2, batch_false_edges_c1_c2)

    def train_loop(self):
        # TODO we can change all this, we can pass the train edges to [from, to, clust, clust]
        # so that we can use the same cicle for between and intra clusts
        batch_losses, batch_accs = [], []

        # shuffle data in order to have different batches in different epochs
        np.random.shuffle(self.train_edges)      
        np.random.shuffle(self.train_false_edges)

        for i in range(0, min(len(self.train_edges), self.train_false_edges.shape[0]), self.batch_size):

            batch_edges = self.train_edges[i: i+self.batch_size]    
            batch_false_edges = self.train_false_edges[i: i+self.batch_size]   

            with tf.GradientTape() as tape:  
                clusts_embs = []

                batch_predictions, batch_gt = None, None

                # compute the embeddings for each node in each cluster
                for clust in range(self.n_clusters):
                    embs = self.model(self.features_tensor[clust], clust, training=True)
                    clusts_embs.append(embs)

                # for each couple compute the predictions over the train edges
                for clust_1 in range(self.n_clusters):
                    for clust_2 in range(clust_1, self.n_clusters):
                        batch_preds_c1_c2, batch_gt_c1_c2 = self.__compute_predictions(batch_edges, 
                            batch_false_edges, clust_1, clust_2, clusts_embs[clust_1], clusts_embs[clust_2])

                        # append the predictions and the gt to the ones already computed in this batch
                        if batch_predictions is None:
                            batch_predictions, batch_gt = batch_preds_c1_c2, batch_gt_c1_c2
                        else: 
                            batch_predictions = tf.concat((batch_predictions, batch_preds_c1_c2), -1)
                            batch_gt = tf.concat((batch_gt, batch_gt_c1_c2), -1)  

                # compute loss and gradient
                loss = topological_loss(batch_gt, batch_predictions)
                grad = tape.gradient(loss, self.model.trainable_variables)

                # get acc
                ta = tf.keras.metrics.Accuracy()
                ta.update_state(batch_gt, tf.round(tf.nn.sigmoid(batch_predictions)))
                acc = ta.result().numpy()
                
                # optimize the weights
                self.optimizer.apply_gradients(zip(grad, self.model.trainable_variables))

                batch_losses.append(loss.numpy())
                batch_accs.append(acc)

        self.train_losses.append(sum(batch_losses)/len(batch_accs))
        self.train_accs.append(sum(batch_accs)/len(batch_accs))


    def valid_loop(self):  

        batch_predictions, batch_gt = None, None

        clusts_embs = []
        # compute the embeddings for each node in each cluster
        for clust in range(self.n_clusters):
            embs = self.model(self.features_tensor[clust], clust, training=False)
            clusts_embs.append(embs)

        # for each couple compute the predictions over the train edges
        for clust_1 in range(self.n_clusters):
            for clust_2 in range(clust_1, self.n_clusters):
                batch_preds_c1_c2, batch_gt_c1_c2 = self.__compute_predictions(self.valid_edges, 
                            self.valid_false_edges, clust_1, clust_2, clusts_embs[clust_1], clusts_embs[clust_2])

                # append the predictions and the gt to the ones already computed in this batch
                if batch_predictions is None:
                    batch_predictions, batch_gt = batch_preds_c1_c2, batch_gt_c1_c2
                else: 
                    batch_predictions = tf.concat((batch_predictions, batch_preds_c1_c2), -1)
                    batch_gt = tf.concat((batch_gt, batch_gt_c1_c2), -1)  
        # compute loss and gradient
        loss = topological_loss(batch_gt, batch_predictions)

        # get acc
        ta = tf.keras.metrics.Accuracy()
        ta.update_state(batch_gt, tf.round(tf.nn.sigmoid(batch_predictions)))
        acc = ta.result().numpy()
        
        self.valid_accs.append(acc)
        self.valid_losses.append(loss.numpy())

        return loss

    def train(self):
        start_time = time.time()
        patience, n_epoch = 0, 0
        while patience <= self.train_patience and n_epoch < self.max_epochs :
            self.train_loop()
            loss = self.valid_loop()
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

    def test(self, test_edges, test_false_edges):
        test_predictions, test_gt = None, None

        clusts_embs = []
        # compute the embeddings for each node in each cluster
        for clust in range(self.n_clusters):
            embs = self.model(self.features_tensor[clust], clust, training=False)
            clusts_embs.append(embs)

        # for each couple compute the predictions over the train edges
        for clust_1 in range(self.n_clusters):
            for clust_2 in range(clust_1, self.n_clusters):
                batch_preds_c1_c2, batch_gt_c1_c2 = self.__compute_predictions(test_edges, 
                            test_false_edges, clust_1, clust_2, clusts_embs[clust_1], clusts_embs[clust_2])

                # append the predictions and the gt to the ones already computed in this batch
                if test_predictions is None:
                    test_predictions, test_gt = batch_preds_c1_c2, batch_gt_c1_c2
                else: 
                    test_predictions = tf.concat((test_predictions, batch_preds_c1_c2), -1)
                    test_gt = tf.concat((test_gt, batch_gt_c1_c2), -1)  

        self.name = f"{self.dataset}_share_last" if self.share_last else f"{self.dataset}_share_first"
        cms = plot_cf_matrix(test_gt, test_predictions, self.name, TEST)

        roc_score = metrics.roc_auc_score(test_gt, test_predictions)
        ap_score = metrics.average_precision_score(test_gt, test_predictions)

        return cms, roc_score, ap_score

class SharedTrainerWithAdvLoss(SharedTrainer):

    def __init__(self, share_last:bool, model_name:str, batch_size:int, train_patience:int, 
        max_epochs:int, lr:float, n_clusters:int, dataset:str, adv_train_also_classifier:bool = False, 
        adv_const_labesl:bool=True, n_adv_epochs:int=5):

        super(SharedTrainerWithAdvLoss, self).__init__(share_last, model_name, batch_size, train_patience, max_epochs, lr, n_clusters, dataset)  

        self.adv_train_also_classifier = adv_train_also_classifier
        self.adv_const_labesl = adv_const_labesl
        self.n_adv_epochs = n_adv_epochs

    def initialize_model(self):
        self.model = LastSharedWithAdversarialLoss(self.adj_train_norm_tensor)   
        n_nodes = [adj.shape[0] for adj in self.adj_train_norm_tensor]

        self.cluster_labels = []
        for i in range(self.n_clusters):
            if LABEL_OF_ALL_1:
                ith_label = [1/self.n_clusters]*self.n_clusters
            else:
                ith_label = [0]*n_clusters
                ith_label[i] = 1
            self.cluster_labels += [ith_label] * n_nodes[i]   

    def __adv_backprop(self):
        with tf.GradientTape() as tape:  
            predicted_classes = []
            # predict the classes for each node
            for clust in range(self.n_clusters):
                predicted_classes.append(self.model(self.features_tensor[clust], cluster=clust, training=True, predict_cluster=True))

            predicted_clusters = tf.concat(predicted_classes, 0)
            
            if self.adv_train_also_classifier:
                trainable_variables = self.model.trainable_variables
            else: 
                trainable_variables = self.model.conv_1.trainable_variables


            if self.adv_const_labesl:
                loss = tf.keras.losses.MeanSquaredError()(self.cluster_labels, predicted_clusters)
                grad = tape.gradient(loss, trainable_variables)
            else:
                loss = tf.keras.losses.CategoricalCrossentropy()(self.cluster_labels, predicted_clusters)
                grad = tape.gradient(-loss, trainable_variables)

            self.optimizer.apply_gradients(zip(grad, trainable_variables))
            

    def train(self):
        start_time = time.time()
        patience, n_epoch = 0, 0
        while patience <= self.train_patience and n_epoch < self.max_epochs:
            if n_epoch % self.n_adv_epochs != 0 or n_epoch == 0:
                self.train_loop()
                loss = self.valid_loop()
                if (loss > min(self.valid_losses)):
                    patience += 1
                else:
                    patience = 0    

                print(f"Epoch: {n_epoch}") 
                print(f"TRAIN_RESULTS:\n\ttrain_acc: {self.train_accs[-1]}\t train_loss: {self.train_losses[-1]}")
                print(f"VALID_RESULTS:\n\tvalid_acc: {self.valid_accs[-1]}\t valid_loss: {self.valid_losses[-1]}")
                print(f"------------------")
            else:
                print(f"ADV EPOCH: {n_epoch}") 
                self.__adv_backprop()
            
            n_epoch += 1

        self.execution_time = time.time() - start_time

        

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

    cms, _, _ = trainer.test(test_edges, test_false_edges)

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

