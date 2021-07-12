import os  
import os.path
import math
import random

from scipy.sparse import csr_matrix
import scipy.sparse as sp
import seaborn as sn
import networkx as nx
import pandas as pd
import numpy as np

import tensorflow as tf

from sklearn import metrics
import matplotlib.pyplot as plt

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
    train_accs = []
    train_losses = []

    valid_accs = []
    valid_losses = []

    optimizer = tf.keras.optimizers.Adam(learning_rate=LR)
    
    n_nodes = adj_train.shape[0]

    # convert the normalized adj and the features to tensors
    adj_train_norm_tensor = convert_sparse_matrix_to_sparse_tensor(adj_train_norm)
    feature_tensor = convert_sparse_matrix_to_sparse_tensor(features)

    # get train ground truth 
    train_y = np.reshape(adj_train.toarray(), (n_nodes*n_nodes))
    train_y = tf.convert_to_tensor(train_y, dtype=tf.float32)
    model = MyModel(adj_train_norm_tensor)
    patience = 0

    train_n_edges = len(np.where(train_y > 0.51)[0])
    train_weight = float(n_nodes**2 - train_n_edges) / train_n_edges

    valid_edges_indeces = [x[0]*n_nodes + x[1] for x in valid_edges]
    valid_false_edges_indeces = [x[0]*n_nodes + x[1] for x in valid_false_edges]

    train_edges_indeces = [x[0]*n_nodes + x[1] for x in train_edges]

    top_loss_norm = n_nodes * n_nodes / float((n_nodes * (n_nodes-1)/2 - len(train_edges_indeces)) * 2)

    print(f"valid_edges_indeces: {len(valid_edges_indeces)}")
    print(f"valid_false_edges_indeces: {len(valid_false_edges_indeces)}")

    """ from_false_indeces = tf.random.uniform((len(train_edges)+5,), maxval = n_nodes, dtype=tf.dtypes.int32)
    to_false_indeces = tf.random.uniform((len(train_edges)+5,), maxval = n_nodes, dtype=tf.dtypes.int32)

    edges = from_false_indeces * n_nodes + to_false_indeces

    from_false_indeces = from_false_indeces[tf.gather(train_y, edges) == 0]
    to_false_indeces = to_false_indeces[tf.gather(train_y, edges) == 0] """

    #train_false_edges = tf.stack((from_false_indeces, to_false_indeces), 1).numpy()

    train_false_edges = get_false_edges(adj_train, len(train_edges), node_to_clust=node_to_clust)

    #train_false_edges_indeces =  edges[tf.gather(train_y, edges) == 0]
    print("train_false_edges", train_false_edges.shape[0])
    print("train_pos_edges", len(train_edges))
    train_y_pos_edges = tf.convert_to_tensor([1.0]*len(train_edges))
    train_y_neg_edges = tf.convert_to_tensor([0.0]*train_false_edges.shape[0])

    tmp_train_y = tf.concat([train_y_pos_edges, train_y_neg_edges], 0)

    batch_size = BATCH_SIZE
    for i in range(EPOCHS):
        
        print(f"epoch: {i}, clust: {clust_id}")

        epoch_losses = []
        epoch_accs = []

        for i in range(0, min(len(train_edges), train_false_edges.shape[0]), batch_size):
            
            if(i == 50):
                pass
                #optimizer.lr.assign(LR/10)
                #batch_size = batch_size/10 
            if(i == 100):
                pass
                #optimizer.lr.assign(LR/100)
            
            # shuffle data in order to have different batches in different epochs
            np.random.shuffle(train_edges)
            np.random.shuffle(train_false_edges)


            with tf.GradientTape() as tape:    
                # forward pass
                embs = model(feature_tensor, training=True)
                
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

        #valid_pred_p = tf.gather(valid_pred, valid_edges_indeces)
        #valid_pred_n = tf.gather(valid_pred, valid_false_edges_indeces)
        
        print("len(valid_edges)", len(valid_edges))
        valid_pred = tf.concat([valid_pred_p, valid_pred_n], 0)
        
        valid_y = [1]*len(valid_edges) + [0]*len(valid_false_edges)
        valid_y = tf.convert_to_tensor(valid_y, dtype=tf.float32)

        loss_weights = tf.convert_to_tensor([[-0.1]*len(valid_edges) + [1]*len(valid_false_edges)])

        valid_loss = topological_loss(valid_y, valid_pred, loss_weights)#total_loss(valid_y, valid_pred, logvar, mu, n_nodes, top_loss_norm)


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

    return model

def plot(train, valid, name, clust_id):
    plt.clf()
    plt.plot(train, label=f"train_{name}")
    plt.plot(valid, label=f"valid_{name}")
    plt.ylabel(f"{name}s")
    plt.xlabel("epochs")
    plt.legend([f"train_{name}", f"valid_{name}"])
    plt.savefig(f"plots/{name}_{clust_id}.png")


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

    for t in [0.5, 0.6, 0.7]:

        plt.clf()
        cm = metrics.confusion_matrix(labels_all, np.where(preds_all > t, 1, 0))#np.round(preds_all))
        df_cm = pd.DataFrame(cm, index = [i for i in "01"],
                    columns = [i for i in "01"])
        plt.figure(figsize = (10,7))
        sn.heatmap(df_cm, annot=True, fmt='g')
        plt.xlabel("predicted labels")
        plt.ylabel("true labels")
        plt.savefig(f"plots/conf_matrix_{dataset}_{clust}_{t}.png")

    print(f"roc_score: {roc_score}")
    print(f"ap_score: {ap_score}")

    roc_curve_plot(labels_all, preds_all, roc_score, dataset, clust)

    return roc_score, ap_score

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

    auc, ap = get_scores(embs, test_p, test_n, dataset, clust)

    return ap, auc

# compute Ã = D^{1/2}(A+I)D^{1/2}
def compute_adj_norm(adj):
    
    adj_I = adj + sp.eye(adj.shape[0])

    D = np.sum(adj_I, axis=1)
    D_power = sp.diags(np.asarray(np.power(D, -0.5)).reshape(-1))

    adj_norm = D_power.dot(adj_I).dot(D_power)

    return adj_norm

def complete_graph(node_to_clust):
    clust = "complete"
    adj_train, features, test_matrix, valid_matrix  = get_complete_data(DATASET_NAME)

    train_edges, _, _ = sparse_to_tuple(adj_train)
    test_edges, _, _ = sparse_to_tuple(test_matrix)
    valid_edges, _, _ = sparse_to_tuple(valid_matrix)
    
    data = [1]*(len(train_edges) + len(test_edges) + len(valid_edges))
    indexes = np.concatenate((train_edges,test_edges, valid_edges), 0)

    complete_adj = csr_matrix((data, (indexes[:,0], indexes[:,1])), shape = adj_train.shape)

    false_edges = get_false_edges(complete_adj, test_edges.shape[0] + valid_edges.shape[0], node_to_clust)
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

if __name__ == "__main__":
    

    # load data : adj, features, node labels and number of clusters
    data = load_data(DATASET_NAME)

    adjs = data[0]
    features_ = data[1] 
    tests = data[2]
    valids = data[3]
    clust_to_node = data[4]
    node_to_clust = data[5]

    test_false_matrix, valid_false_matrix, test_ap, test_auc, execution_time = complete_graph(None)#(node_to_clust)

    n_test_edges = []
    n_valid_edges = []

    test_aps = [test_ap]
    test_aucs = [test_auc]
    execution_times = [execution_time]

    subset_lenghts = []
    
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
        model = train(features, adj_train, adj_train_norm, train_edges, valid_edges, valid_false_edges, clust)
        execution_times.append(time.time() - start_time)
        model.save_weights(f"weights/{DATASET_NAME}_{clust}")

        test_ap, test_auc = test(features, model, test_edges, test_false_edges, DATASET_NAME, clust)
        
        test_aps.append(test_ap)
        test_aucs.append(test_auc)    

    print(f"test ap: {test_aps}")
    print(f"test auc: {test_aucs}")
    print(f"execution_times: {execution_times}")
    print()
    print(f"n_test_edges: {n_test_edges}")
    print(f"n_valid_edges: {n_valid_edges}")