import scipy.sparse as sp
import tensorflow as tf
import numpy as np

import seaborn as sn
import pandas as pd
from sklearn import metrics
import matplotlib.pyplot as plt

# convert sparse matrix to sparse tensor
def convert_sparse_matrix_to_sparse_tensor(sparse_mx):
    if not sp.isspmatrix_coo(sparse_mx):
        sparse_mx = sparse_mx.tocoo()
    indices_matrix = np.vstack((sparse_mx.row, sparse_mx.col)).transpose()
    values = sparse_mx.data
    shape = sparse_mx.shape
    sparse_tensor = tf.SparseTensor(indices=indices_matrix, values=values, dense_shape=shape)
    return tf.cast(sparse_tensor, dtype=tf.float32)

# compute Ã = D^{1/2}(A+I)D^{1/2}
def compute_adj_norm(adj):
    
    adj_I = adj + sp.eye(adj.shape[0])

    D = np.sum(adj_I, axis=1)
    D_power = sp.diags(np.asarray(np.power(D, -0.5)).reshape(-1))

    adj_norm = D_power.dot(adj_I).dot(D_power)

    return adj_norm

# plot train and valid loss or acc into the same plot wrt the epochs
def plot(train, valid, name, clust_id):
    plt.clf()
    plt.plot(train, label=f"train_{name}")
    plt.plot(valid, label=f"valid_{name}")
    plt.ylabel(f"{name}s")
    plt.xlabel("epochs")
    plt.legend([f"train_{name}", f"valid_{name}"])
    plt.savefig(f"plots/{name}_{clust_id}.png")

def plot_cf_matrix(labels, preds, name, test=False):
    to_remove_zero, to_remove_one = False, False

    if 0 not in labels:
        preds = np.append(preds, 0)
        labels = np.append(labels, 0)
        print(labels)
        print(preds)
        to_remove_zero = True
    if 1 not in labels:
        preds = np.append(preds, 1)
        labels = np.append(labels, 1)
        print(labels)
        print(preds)
        to_remove_one = True

    cms = []
    for t in [0.5, 0.6, 0.7]:

        plt.clf()
        cm = metrics.confusion_matrix(labels, np.where(preds > t, 1, 0))
        print(cm)
        if to_remove_one:
            cm[1][1] = cm[1][1] - 1
            assert cm[1][1] == 0 
        if to_remove_zero:
            cm[0][0] = cm[0][0] - 1
            assert cm[0][0] == 0 

        cms.append(cm)

        df_cm = pd.DataFrame(cm, index = [i for i in "01"],
                    columns = [i for i in "01"])
        plt.figure(figsize = (10,7))
        sn.heatmap(df_cm, annot=True, fmt='g')
        plt.xlabel("predicted labels")
        plt.ylabel("true labels")
        if not test:
            plt.savefig(f"plots/conf_matrix_{name}_{t}.png")

    return cms

def batch_prediction(edges, embs_from, embs_to, predictions):
    tmp_from = edges[:,0]
    tmp_to = edges[:,1]

    tmp_embs_from = tf.gather(embs_from, tmp_from)
    tmp_embs_to = tf.gather(embs_to, tmp_to)

    batch_logits = tf.linalg.diag_part(tf.matmul(tmp_embs_from, tmp_embs_to, transpose_b=True))
    
    if(predictions is None):
        predictions = batch_logits
    else:
        predictions = tf.concat((predictions, batch_logits), -1)

    return predictions

def get_predictions(embs_from, edges, batch_size, embs_to=None):
    if embs_to is None:
        embs_to = embs_from

    predictions = None

    for i in range(0, edges.shape[0], batch_size):
        tmp_edges = edges[i: i+batch_size]
        predictions = batch_prediction(tmp_edges, embs_from, embs_to, predictions)
    predictions = batch_prediction(edges[i+batch_size:], embs_from, embs_to, predictions)

    return predictions

def get_predictions_and_labels(embs_from, edges_pos, edges_neg, batch_size, embs_to=None):
    
    preds_pos = get_predictions(embs_from, edges_pos, batch_size, embs_to=embs_to)
    preds_neg = get_predictions(embs_from, edges_neg, batch_size, embs_to=embs_to)

    print(preds_pos.shape, edges_pos.shape)
    print(preds_pos.shape, edges_neg.shape)


    if (preds_pos is not None and preds_neg is not None):
        preds_all = tf.concat([preds_pos, preds_neg], 0)
    elif(preds_neg is not None):
        preds_all = preds_neg
    else: 
        preds_all = preds_pos

    labels_all = np.hstack([np.ones(len(edges_pos)), np.zeros(len(edges_neg))])

    return preds_all, labels_all