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
from data import load_data, get_test_edges, get_false_edges, sparse_to_tuple, get_complete_cora_data, get_complete_data
from metrics import clustering_metrics
from base_model import MyModel    
from loss import total_loss, topological_loss

import time

# set the seed
tf.random.set_seed(SEED)
random.seed(SEED)
np.random.RandomState(SEED)


from link_prediction import train as matrix_train

# convert sparse matrix to sparse tensor
def convert_sparse_matrix_to_sparse_tensor(sparse_mx):
    if not sp.isspmatrix_coo(sparse_mx):
        sparse_mx = sparse_mx.tocoo()
    indices_matrix = np.vstack((sparse_mx.row, sparse_mx.col)).transpose()
    values = sparse_mx.data
    shape = sparse_mx.shape
    sparse_tensor = tf.SparseTensor(indices=indices_matrix, values=values, dense_shape=shape)
    return tf.cast(sparse_tensor, dtype=tf.float32)

def train(features, adj_train, adj_train_norm, train_edges, valid_edges, valid_false_edges, clust_id, node_to_clust=None):
    train_accs, train_losses = [], []

    valid_accs, valid_losses = [], [] 

    optimizer = tf.keras.optimizers.Adam(learning_rate=LR)
    
    n_nodes = adj_train.shape[0]

    # convert the normalized adj and the features to tensors
    adj_train_norm_tensor = convert_sparse_matrix_to_sparse_tensor(adj_train_norm)
    feature_tensor = convert_sparse_matrix_to_sparse_tensor(features)

    # get train ground truth 
    train_y = np.reshape(adj_train.toarray(), (n_nodes*n_nodes))
    train_y = tf.convert_to_tensor(train_y, dtype=tf.float32)

    # define the model
    model = MyModel(adj_train_norm_tensor)

    # initialize the patience
    patience = 0
    
    train_false_edges = get_false_edges(adj_train, len(train_edges), node_to_clust=node_to_clust)

    valid_edges_indeces = [x[0]*n_nodes + x[1] for x in valid_edges]
    valid_false_edges_indeces = [x[0]*n_nodes + x[1] for x in valid_false_edges]


    print(f"valid_edges_indeces: {len(valid_edges_indeces)}")
    print(f"valid_false_edges_indeces: {len(valid_false_edges_indeces)}")


    print("train_false_edges", train_false_edges.shape[0])
    print("train_pos_edges", len(train_edges))
    train_y_pos_edges = tf.convert_to_tensor([1.0]*len(train_edges))
    train_y_neg_edges = tf.convert_to_tensor([0.0]*train_false_edges.shape[0])

    tmp_train_y = tf.concat([train_y_pos_edges, train_y_neg_edges], 0)

    batch_size = BATCH_SIZE
    n_epochs = 0
    for epoch in range(EPOCHS):
        n_epochs = epoch
        print(f"epoch: {epoch}, clust: {clust_id}")

        epoch_losses = []
        epoch_accs = []


        for i in range(0, min(len(train_edges), train_false_edges.shape[0]), batch_size):
            
            # shuffle data in order to have different batches in different epochs
            np.random.shuffle(train_edges)
            np.random.shuffle(train_false_edges)


            with tf.GradientTape() as tape:    
                # forward pass
                embs = model(feature_tensor, training=True)
                print(f"forward pass: {i}, {min(len(train_edges), train_false_edges.shape[0])}")
                
                tmp_train_edges = train_edges[i: i+batch_size]
                tmp_from = tmp_train_edges[:,0]
                tmp_to = tmp_train_edges[:,1]

                embs_from = tf.gather(embs, tmp_from)
                embs_to = tf.gather(embs, tmp_to)
                train_pos_pred = tf.linalg.diag_part(tf.matmul(embs_from, embs_to, transpose_b=True))
                
                tmp_train_edges = train_false_edges[i: i+batch_size]
                tmp_from = tmp_train_edges[:,0]
                tmp_to = tmp_train_edges[:,1]

                embs_from = tf.gather(embs, tmp_from)
                embs_to = tf.gather(embs, tmp_to)
                train_neg_pred = tf.linalg.diag_part(tf.matmul(embs_from, embs_to, transpose_b=True))
            
                train_pred = tf.concat((train_pos_pred, train_neg_pred), 0)
                tmp_train_y = tf.concat([tf.ones(train_pos_pred.shape[0]),tf.zeros(train_neg_pred.shape[0])], 0)
                
                loss_weights = tf.convert_to_tensor([[-0.1]*train_pos_pred.shape[0] + [1]*train_neg_pred.shape[0]])
                """ print(loss_weights.shape)
                print(train_pred.shape)
                print(tmp_train_y.shape) """

                # get loss
                loss = topological_loss(tmp_train_y, train_pred, loss_weights)#total_loss(tmp_train_y, train_pred, logvar, mu, n_nodes, top_loss_norm)
                print("calculated loss")
                # get gradient from loss 
                grad = tape.gradient(loss, model.trainable_variables)

                # get acc
                ta = tf.keras.metrics.Accuracy()
                ta.update_state(tmp_train_y, tf.round(tf.nn.sigmoid(train_pred)))
                train_acc = ta.result().numpy()

                # optimize the weights
                optimizer.apply_gradients(zip(grad, model.trainable_variables))

                epoch_losses.append(loss.numpy())
                epoch_accs.append(train_acc)

        train_losses.append(sum(epoch_losses)/len(epoch_losses))
        train_accs.append(sum(epoch_accs)/len(epoch_accs))
        print(f"train_loss: {train_losses[-1]}")
        print(f"train_acc: {train_accs[-1]}")

        # save memory
        grad = None
        train_pred = None

        embs = model(feature_tensor, training=False)
        
        valid_pred_p=None
        for i in range(0, len(valid_edges), BATCH_SIZE):
            tmp_valid_edges = valid_edges[i: i+BATCH_SIZE]
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
        for i in range(0, len(valid_false_edges), BATCH_SIZE):
            tmp_valid_edges = valid_false_edges[i: i+BATCH_SIZE]
            tmp_from = tmp_valid_edges[:,0]
            tmp_to = tmp_valid_edges[:,1]

            embs_from = tf.gather(embs, tmp_from)
            embs_to = tf.gather(embs, tmp_to)
            if(valid_pred_n is None):
                valid_pred_n = tf.linalg.diag_part(tf.matmul(embs_from, embs_to, transpose_b=True))
            else:
                batch_logits = tf.linalg.diag_part(tf.matmul(embs_from, embs_to, transpose_b=True))
                valid_pred_n = tf.concat((valid_pred_n, batch_logits), -1)

        
        print("len(valid_edges)", len(valid_edges))
        valid_pred = tf.concat([valid_pred_p, valid_pred_n], 0)
        
        valid_y = [1]*len(valid_edges) + [0]*len(valid_false_edges)
        valid_y = tf.convert_to_tensor(valid_y, dtype=tf.float32)

        loss_weights = tf.convert_to_tensor([[-0.1]*len(valid_edges) + [1]*len(valid_false_edges)])

        valid_loss = topological_loss(valid_y, valid_pred, loss_weights)


        va = tf.keras.metrics.Accuracy()
        va.update_state(valid_y, tf.round(tf.nn.sigmoid(valid_pred)))
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

    plot(train_losses, valid_losses, "loss", clust_id)
    plot(train_accs, valid_accs, "acc", clust_id)

    with open("plots/n_epochs.txt", "a") as fout:
        fout.write(f"clust_{clust_id}, {DATASET_NAME}, {n_epochs}\n")

    return model

def plot(train, valid, name, clust_id):
    plt.clf()
    plt.plot(train, label=f"train_{name}")
    plt.plot(valid, label=f"valid_{name}")
    plt.ylabel(f"{name}s")
    plt.xlabel("epochs")
    plt.legend([f"train_{name}", f"valid_{name}"])
    plt.savefig(f"plots/{name}_{clust_id}.png")

def plot_cf_matrix(labels, preds, name):
    cms = []
    for t in [0.5, 0.6, 0.7]:

        plt.clf()
        cm = metrics.confusion_matrix(labels, np.where(preds > t, 1, 0))

        cms.append(cm)

        df_cm = pd.DataFrame(cm, index = [i for i in "01"],
                    columns = [i for i in "01"])
        plt.figure(figsize = (10,7))
        sn.heatmap(df_cm, annot=True, fmt='g')
        plt.xlabel("predicted labels")
        plt.ylabel("true labels")
        plt.savefig(f"plots/conf_matrix_{name}_{t}.png")

    return cms


def get_scores(embs, edges_pos, edges_neg, dataset, clust):
    
    valid_pred_p=None
    for i in range(0, len(edges_pos), BATCH_SIZE):
        tmp_valid_edges = edges_pos[i: i+BATCH_SIZE]
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
    for i in range(0, len(edges_neg), BATCH_SIZE):
        tmp_valid_edges = edges_neg[i: i+BATCH_SIZE]
        tmp_from = tmp_valid_edges[:,0]
        tmp_to = tmp_valid_edges[:,1]

        embs_from = tf.gather(embs, tmp_from)
        embs_to = tf.gather(embs, tmp_to)
        if(valid_pred_n is None):
            valid_pred_n = tf.linalg.diag_part(tf.matmul(embs_from, embs_to, transpose_b=True))
        else:
            batch_logits = tf.linalg.diag_part(tf.matmul(embs_from, embs_to, transpose_b=True))
            valid_pred_n = tf.concat((valid_pred_n, batch_logits), -1)

    preds_all = tf.sigmoid(tf.concat([valid_pred_p, valid_pred_n], 0)).numpy()

    labels_all = np.hstack([np.ones(len(edges_pos)), np.zeros(len(edges_neg))])

    roc_score = metrics.roc_auc_score(labels_all, preds_all)
    ap_score = metrics.average_precision_score(labels_all, preds_all)

    cms = plot_cf_matrix(labels_all, preds_all, f"{dataset}_{clust}")

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

def test(features, model, test_p, test_n, dataset, clust):
    n_nodes = features.shape[0]
    feature_tensor = convert_sparse_matrix_to_sparse_tensor(features)

    embs = model(feature_tensor, training=False)

    #pred = np.reshape(pred.numpy(), (n_nodes, n_nodes))

    auc, ap, cms = get_scores(embs, test_p, test_n, dataset, clust)

    return ap, auc, cms

def test_with_intra_edges_to_zero(features, models, test_ps, test_ns, dataset, model_name, test_p_intra, test_n_intra):
    
    features = [convert_sparse_matrix_to_sparse_tensor(features[i]) for i in range(len(features))] 

    labels_all, preds_all = None, None

    for i in range(len(features)):
        embs = models[i](features[i])
        preds, labels = get_preds(embs, test_ps[i], test_ns[i])
        if labels_all is None:
            labels_all = labels
            preds_all = preds
        else:
            preds_all = tf.concat((preds_all, preds), -1)
            
            labels_all = np.hstack([labels_all, labels])
            
    preds_all = tf.sigmoid(preds_all)
    
    preds_all = tf.concat((preds_all, [0]*test_p_intra, [0]*test_n_intra), -1)
    labels_all = np.hstack([labels_all, [1]*test_p_intra, [0]*test_n_intra])

    roc_score = metrics.roc_auc_score(labels_all, preds_all)
    ap_score = metrics.average_precision_score(labels_all, preds_all)
    
    plot_cf_matrix(labels_all, preds_all, f"{dataset}_{model_name}")

    print(f"roc_score: {roc_score}")
    print(f"ap_score: {ap_score}")

    roc_curve_plot(labels_all, preds_all, roc_score, dataset, model_name)

    return roc_score, ap_score

def get_preds(embs, edges_pos, edges_neg, embs_1=None):
    
    edges_pos = np.array(edges_pos)
    edges_neg = np.array(edges_neg)


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

    preds_all = tf.concat([valid_pred_p, valid_pred_n], 0)

    labels_all = np.hstack([np.ones(len(edges_pos)), np.zeros(len(edges_neg))])

    return preds_all, labels_all

def single_cluster_test_couple_models(features_tensors, models, test_p, test_n, clust_to_nodes):
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

        
            tmp_preds, tmp_labels = get_preds(embs, test_p_1_2, test_n_1_2)
            
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

def couple_test(features_tensors, models, test_p, test_n):
    counter = 0

    labels_all, preds_all = None, None
    
    for clust_1 in range(len(features_tensors)):
        for clust_2 in range(clust_1+1, len(features_tensors)):
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

            embs = models[counter](features_tensors[counter])

            preds, labels = get_preds(embs, test_p_1_2, test_n_1_2)

            if(labels_all is None):
                preds_all = preds
                labels_all = labels
            else:
                preds_all = tf.concat([preds_all, preds], 0)
                labels_all = np.hstack([labels_all, labels])

            counter += 1
    return preds_all, labels_all

def convert_edges_to_clust_idxs(edges, clust_to_node, node_to_clust):
    positive_edges = []
    clust_single_first = node_to_clust[edges[0][0]]
    
    print("len:", len(clust_to_node[clust_single_first]))
    
    for edge in edges:
        from_idx, to_idx = edge[0], edge[1] 
        clust_single = node_to_clust[from_idx]
        
        assert clust_single == node_to_clust[to_idx] and clust_single_first == clust_single

        from_idx_clust, to_idx_clust = clust_to_node[clust_single].index(from_idx), clust_to_node[clust_single].index(to_idx) 
        positive_edges.append([from_idx_clust, to_idx_clust])
    
    return np.array(positive_edges)

"""
    test_p: [[idx_1, idx_2, clust_1, clust_2]]
        in the case of test without single models, we have that idx_1 and idx_2 are the indeces of the nodes
        inside the model trained over clust_1 clust_2

        in the case of test with single models, we have that idx_1 and idx_2 are the indices of the nodes 
        inside the model trained only over clust_1 = clust_2
    
    same for test_n
"""
def couple_and_single_test(features, models, test_p, test_n, dataset, clust_to_nodes = None, single_models = None, single_features = None, single_clust_to_node = None, single_node_to_clust = None):
    features_tensors = [convert_sparse_matrix_to_sparse_tensor(features[i]) for i in range(len(features))] 

    single_models_condition = single_models is not None and single_features is not None and single_clust_to_node is not None and single_node_to_clust is not None
    couple_models_condition = clust_to_nodes is not None

    assert single_models_condition or couple_models_condition
     
    # predict edges between different clusters
    preds_all, labels_all = couple_test(features_tensors, models, test_p, test_n)

    
    if(couple_models_condition):
        preds, labels = single_cluster_test_couple_models(features_tensors, models, test_p, test_n, clust_to_nodes)
        
        preds_all = tf.concat([preds_all, preds], 0)
        labels_all = np.hstack([labels_all, labels])

    elif(single_models_condition):
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

            print("single_features[clust].shape[0]", single_features[real_clust].shape[0])

            embs = single_models[real_clust](single_features[real_clust])

            preds, labels = get_preds(embs, test_p_clust, test_n_clust)

            if(labels_all is None):
                preds_all = preds
                labels_all = labels
            else:
                preds_all = tf.concat([preds_all, preds], 0)
                labels_all = np.hstack([labels_all, labels])
    else:
        raise Exception("I couldnt test over the edges inside the same cluster")

    name = f"{dataset}_only_couples" if couple_models_condition else f"{dataset}_couple_for_diff_clusts_single_for_same_clust"
    cms = plot_cf_matrix(labels_all, preds_all, name)

    roc_score = metrics.roc_auc_score(labels_all, preds_all)
    ap_score = metrics.average_precision_score(labels_all, preds_all)



    return roc_score, ap_score, cms
    

# compute Ã = D^{1/2}(A+I)D^{1/2}
def compute_adj_norm(adj):
    
    adj_I = adj + sp.eye(adj.shape[0])

    D = np.sum(adj_I, axis=1)
    D_power = sp.diags(np.asarray(np.power(D, -0.5)).reshape(-1))

    adj_norm = D_power.dot(adj_I).dot(D_power)

    return adj_norm

def complete_graph(node_to_clust):
    clust = "complete"
    adj_train, features, test_matrix, valid_matrix  = get_complete_data(DATASET_NAME, leave_intra_clust_edges=LEAVE_INTRA_CLUSTERS)

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

    test_ap, test_auc = 0,0
    # start training
    start_time = time.time()
    #model = train(features, adj_train, adj_train_norm, train_edges, valid_edges, valid_false_edges, clust, node_to_clust=node_to_clust)
    execution_time = time.time() - start_time
    #model.save_weights(f"weights/{DATASET_NAME}_{clust}")

    #test_ap, test_auc = test(features, model, test_edges, test_false_edges, DATASET_NAME, clust)
    


    test_ones = [1]*test_false_edges.shape[0]
    valid_ones = [1]*valid_false_edges.shape[0]

    test_false_matrix = csr_matrix((test_ones, (test_false_edges[:,0], test_false_edges[:,1])), adj_train.shape)
    valid_false_matrix = csr_matrix((valid_ones, (valid_false_edges[:,0], valid_false_edges[:,1])), adj_train.shape)

    return test_false_matrix, valid_false_matrix, test_ap, test_auc, execution_time


"""
returns a np array containing for each test edge:
    n_from_couple, n_to_couple, from_clust, to_clust \
    where n_from_couple and n_to_couple are respectively the clust of the first node and the clust of the second node
    from_clust are to_clust 
        - if the cluster are different -> the idx of the two nodes wrt the couple model
        - if the cluster are the same  -> idx of the nodes in the complete graph

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

            if(test_edges_clusts[-1][0] == 4960 or test_edges_clusts[-1][1] == 4960):
                print("from_clust != to_clust", test_edges_clusts[-1])

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

def couple_main(adjs, features_, tests, valids, clust_to_node, node_to_clust):

    test_false_matrix, valid_false_matrix, test_ap, test_auc, execution_time = complete_graph(None)

    n_test_edges = []
    n_valid_edges = []

    test_aps = [test_ap]
    test_aucs = [test_auc]
    execution_times = [execution_time]

    models = []

    for clust in range(len(adjs.keys())):
        print("\n")
        adj_train = adjs[clust]

        train_edges, _, _ = sparse_to_tuple(adj_train)
        valid_edges, _, _ = sparse_to_tuple(valids[clust])
        
        n_valid_edges.append(valid_edges.shape[0])

        test_false_matrix_c = test_false_matrix[clust_to_node[clust], :]
        test_false_matrix_c = test_false_matrix_c[:, clust_to_node[clust]]

        valid_false_matrix_c = valid_false_matrix[clust_to_node[clust], :]
        valid_false_matrix_c = valid_false_matrix_c[:, clust_to_node[clust]]
        
        test_false_edges, _, _ = sparse_to_tuple(test_false_matrix_c)
        valid_false_edges, _, _ = sparse_to_tuple(valid_false_matrix_c)


        # since get_test_edges returns a triu, we sum to its transpose 
        adj_train = adj_train + adj_train.T

        # get normalized adj
        adj_train_norm = compute_adj_norm(adj_train)

        features = features_[clust]
        
        print(f"valid_edges: {valid_edges.shape[0]}")
        print(f"valid_false_edges: {valid_false_edges.shape[0]}")

        # start training
        start_time = time.time()

        model = None
        if adj_train.shape[0] < 10_000 and MATRIX_OPERATIONS:
            model =  matrix_train(features, adj_train, adj_train_norm, train_edges, valid_edges, valid_false_edges, clust, node_to_clust = node_to_clust)
        else:
            model = train(features, adj_train, adj_train_norm, train_edges, valid_edges, valid_false_edges, clust)
        
        execution_times.append(time.time() - start_time)
        model.save_weights(f"weights/{DATASET_NAME}_{clust}")

        models.append(model)
    
    _, _, test_matrix, _  = get_complete_data(DATASET_NAME, leave_intra_clust_edges=LEAVE_INTRA_CLUSTERS)

    n_original_clusters = (1 + math.sqrt(1 + 8*(len(features_))))/2
    assert n_original_clusters == int(n_original_clusters)

    test_edges_p, _, _ = sparse_to_tuple(test_matrix)

    print(test_edges_p[(test_edges_p[:,0] == 4960) + (test_edges_p[:,1] == 4960)])

    test_edges_n, _, _ = sparse_to_tuple(test_false_matrix)
    test_edges_clusts_p = build_test_edges(test_edges_p, node_to_clust, clust_to_node, n_original_clusters)
    test_edges_clusts_n = build_test_edges(test_edges_n, node_to_clust, clust_to_node, n_original_clusters)


    return models, execution_times, test_edges_clusts_p, test_edges_clusts_n

def single_main(adjs, features_, tests, valids, clust_to_node, node_to_clust):

    test_false_matrix, valid_false_matrix, test_ap, test_auc, execution_time = complete_graph(None)#(node_to_clust)
    _, _, test_matrix_complete, _  = get_complete_data(DATASET_NAME, leave_intra_clust_edges=LEAVE_INTRA_CLUSTERS)

    n_test_edges = []
    n_valid_edges = []

    test_aps = [test_ap]
    test_aucs = [test_auc]
    execution_times = [execution_time]

    subset_lenghts = []
    
    models = []
    test_ps, test_ns = [], []
    test_p_intra, test_n_intra = 0, 0

    for clust in range(len(adjs.keys())):
        print("\n")
        #complete_adj = adjs[clust]

        adj_train = adjs[clust]

        train_edges, _, _ = sparse_to_tuple(adj_train)
        test_edges, _, _ = sparse_to_tuple(tests[clust])
        valid_edges, _, _ = sparse_to_tuple(valids[clust])
        
        n_test_edges.append(test_edges.shape[0])
        n_valid_edges.append(valid_edges.shape[0])

        test_false_matrix_c = test_false_matrix[clust_to_node[clust], :]
        test_false_matrix_c = test_false_matrix_c[:, clust_to_node[clust]]

        not_clust_idxs = []
        for not_clust in range(len(features_)):
            if not_clust != clust:
                not_clust_idxs += clust_to_node[not_clust]

        intra_f_clusts_1 = test_false_matrix[clust_to_node[clust], :][:, not_clust_idxs]
        intra_f_clusts_2 = test_false_matrix[not_clust_idxs, :][:, clust_to_node[clust]]
        
        intra_p_clusts_1 = test_matrix_complete[clust_to_node[clust], :][:, not_clust_idxs]
        intra_p_clusts_2 = test_matrix_complete[not_clust_idxs, :][:, clust_to_node[clust]]

        test_n_intra += (intra_f_clusts_1 != 0).sum()
        test_n_intra += (intra_f_clusts_2 != 0).sum()

        test_p_intra += (intra_p_clusts_1 !=0).sum()
        test_p_intra += (intra_p_clusts_2 !=0).sum()



        valid_false_matrix_c = valid_false_matrix[clust_to_node[clust], :]
        valid_false_matrix_c = valid_false_matrix_c[:, clust_to_node[clust]]
        
        test_false_edges, _, _ = sparse_to_tuple(test_false_matrix_c)
        valid_false_edges, _, _ = sparse_to_tuple(valid_false_matrix_c)

        subset_lenghts.append((len(valid_edges), len(valid_false_edges), len(test_edges), len(test_false_edges)))

        # since get_test_edges returns a triu, we sum to its transpose 
        adj_train = adj_train + adj_train.T

        # get normalized adj
        adj_train_norm = compute_adj_norm(adj_train)

        features = features_[clust]
        
        print(f"valid_edges: {valid_edges.shape[0]}")
        print(f"valid_false_edges: {valid_false_edges.shape[0]}")

        # start training
        start_time = time.time()
        
        model = None
        if adj_train.shape[0] < 10_000 and MATRIX_OPERATIONS:
            model = matrix_train(features, adj_train, adj_train_norm, train_edges, valid_edges, valid_false_edges, clust)
        else:
            model = train(features, adj_train, adj_train_norm, train_edges, valid_edges, valid_false_edges, clust)
        
        execution_times.append(time.time() - start_time)
        
        models.append(model)
        
        model.save_weights(f"weights/{DATASET_NAME}_{clust}")

        test_ap, test_auc, cms = test(features, model, test_edges, test_false_edges, DATASET_NAME, clust)
        test_ps.append(test_edges)
        test_ns.append(test_false_edges)

        test_aps.append(test_ap)
        test_aucs.append(test_auc)

    if not COUPLE_AND_SINGLE:
        test_with_intra_edges_to_zero(features_, models,  test_ps, test_ns, DATASET_NAME, "single_clusters_zero_intra",test_p_intra, test_n_intra) 

        print("intra test edges num: ", test_p_intra, test_n_intra)
        print(f"test ap: {test_aps}")
        print(f"test auc: {test_aucs}")
        print()
        print(f"n_test_edges: {n_test_edges}")
        print(f"n_valid_edges: {n_valid_edges}")

    return models, execution_times, test_edges, test_false_edges

if __name__ == "__main__":
    
    # load data
    single_data = load_data(DATASET_NAME, get_couples = False)
    couple_data = load_data(DATASET_NAME, get_couples = True)

    adjs_single, features_single, tests_single, valids_single, clust_to_node_single, node_to_clust_single, _ = single_data
    adjs_couple, features_couple, tests_couple, valids_couple, clust_to_node_couple, node_to_clust_couple, _ = couple_data


    if COUPLE_AND_SINGLE:
        print("train both")
        models_couple, execution_times_couple, test_p_couple, test_n_couple = couple_main(adjs_couple, features_couple, tests_couple, valids_couple, clust_to_node_couple, node_to_clust_couple)
        print("train couple end")
        models_single, execution_times_single, test_p_single, test_n_single = single_main(adjs_single, features_single, tests_single, valids_single, clust_to_node_single, node_to_clust_single)
        print("train single end")
        
        _, _, cms_both = couple_and_single_test(features_couple, models_couple, test_p_couple, test_n_couple, DATASET_NAME, single_models=models_single, single_features=features_single, single_clust_to_node=clust_to_node_single, single_node_to_clust=node_to_clust_single)
        print("test both end")
        _, _, cms_only_couples =couple_and_single_test(features_couple, models_couple, test_p_couple, test_n_couple, DATASET_NAME, clust_to_nodes = clust_to_node_couple)
        print("test only couples end")
        
        f1s, precs, recs = [], [], []
        for cm in cms_both:
            tp, fp, fn = cm[1][1], cm[0][1], cm[1][0]
            precs.append(tp/(tp+fp))
            recs.append(tp/(tp+fn))
            f1s.append(2*precs[-1]*recs[-1]/(precs[-1]+recs[-1])) 

        with open("results/{DATASET_NAME}_couple_for_diff_clusts_single_for_same_clust.txt", "a") as fout:
            fout.write(f"precs: {precs}\n")
            fout.write(f"recs: {recs}\n")
            fout.write(f"f1s: {f1s}\n")
            fout.write(f"time: {sum(execution_times_couple) + sum(execution_times_single)}\n")
            fout.write("-"*10)


        f1s, precs, recs = [], [], []
        for cm in cms_only_couples:
            tp, fp, fn = cm[1][1], cm[0][1], cm[1][0]
            precs.append(tp/(tp+fp))
            recs.append(tp/(tp+fn))
            f1s.append(2*precs[-1]*recs[-1]/(precs[-1]+recs[-1])) 

        with open("results/{DATASET_NAME}_only_couples.txt", "a") as fout:
            fout.write(f"precs: {precs}\n")
            fout.write(f"recs: {recs}\n")
            fout.write(f"f1s: {f1s}\n")
            fout.write(f"time: {sum(execution_times_couple)}\n")
            fout.write("-"*10)


        print(execution_times_couple)
        print(execution_times_single)


    elif COUPLES_TRAIN:
        models_couple, execution_times_couple, test_p_couple, test_n_couple = couple_main(adjs_couple, features_couple, tests_couple, valids_couple, clust_to_node_couple, node_to_clust_couple)
        _, _, cms =couple_and_single_test(features_couple, models_couple, test_p_couple, test_n_couple, DATASET_NAME, clust_to_node = clust_to_node_couple)
        print(execution_times_couple)
    else:
        raise Exception("No modality selected")
