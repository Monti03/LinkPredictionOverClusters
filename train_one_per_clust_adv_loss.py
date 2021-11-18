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
        N_CLUSTERS = int(current_value)
        print("Num Clusters:", N_CLUSTERS)
    
    elif current_argument in ("--constlabel"):
        LABEL_OF_ALL_1 = True    
        print("CONST LABEL:", True)
    
    elif current_argument in ("--mse"):
        print("USING MSE LOSS WITH CONSTANT LABEL")
        LABEL_OF_ALL_1, MSE_LOSS = True, True    
        
    elif current_argument in ("--test"):
        EPOCHS = 5
        print("Testing train: n epochs:", EPOCHS)
    
    elif current_argument in ("--nonneggrad"):
        print("using non negative gradient")
        NON_NEG_GRAD = True
        LABEL_OF_ALL_1 = False    

if LABEL_OF_ALL_1 and NON_NEG_GRAD:
    raise Exception("ONLY ONE OF THE TWO CAN BE TRUE: LABEL_OF_ALL_1, NON_NEG_GRAD")

print("MSE_LOSS", MSE_LOSS, "LABEL_OF_ALL_1", LABEL_OF_ALL_1)


def batched_train(model, optimizer, feature_tensor, train_edges, train_false_edges, valid_edges, valid_false_edges, clust_id, classifier, node_to_clust=None, old_patience=0, last_loss=None):
    train_accs, train_losses = [], []
    valid_accs, valid_losses = [], [] 

    if last_loss is not None:
        valid_losses = [last_loss] 

    patience = old_patience

    n_nodes = feature_tensor.shape[0]

    batch_size = BATCH_SIZE
    n_epochs = 0
    # EACH NUM_EPOCHS_ADV_LOSS EPOCH COMPUTE THE ADV LOSS
    for epoch in range(NUM_EPOCHS_ADV_LOSS): 
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

                # get loss
                loss = topological_loss(tmp_train_y, train_pred, loss_weights)#total_loss(tmp_train_y, train_pred, logvar, mu, n_nodes, top_loss_norm)
                #print("calculated loss")
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

    #plot(train_losses, valid_losses, "loss", clust_id)
    #plot(train_accs, valid_accs, "acc", clust_id)

    # after 5 iterations adversarial loss
    if patience < PATIENCE:
        if LABEL_OF_ALL_1:
            single_label = [1/N_CLUSTERS]*N_CLUSTERS 
        else:
            single_label = [0]*N_CLUSTERS 
            single_label[clust_id] = 1  
        
        cluster_labels = [single_label] * n_nodes
        
        with tf.GradientTape() as tape: 
            embs = model(feature_tensor, training=True)
            predicted_clusters = classifier(embs)
                    
            print(f"="*15)
            print(f"ClassificationLoss: {loss}")
            print(f"="*15)


            if TRAIN_ALSO_CLASSIFIER:
                trainable_variables = model.trainable_variables + classifier.trainable_variables
            else:       
                trainable_variables = model.trainable_variables

            if MSE_LOSS:
                loss = tf.keras.losses.MeanSquaredError()(cluster_labels, predicted_clusters) 
                grad = tape.gradient(loss, trainable_variables)
            elif LABEL_OF_ALL_1 and not MSE_LOSS:
                loss = tf.keras.losses.CategoricalCrossentropy()(cluster_labels, predicted_clusters)
                grad = tape.gradient(loss, trainable_variables)
            elif NON_NEG_GRAD: 
                loss = tf.keras.losses.CategoricalCrossentropy()(cluster_labels, predicted_clusters)
                grad = tape.gradient(loss, trainable_variables)
            else: 
                loss = tf.keras.losses.CategoricalCrossentropy()(cluster_labels, predicted_clusters)
                grad = tape.gradient(-loss, trainable_variables)

            optimizer.apply_gradients(zip(grad, trainable_variables))

    return model, patience, min(valid_losses)

def plot(train, valid, name, clust_id):
    plt.clf()
    plt.plot(train, label=f"train_{name}")
    plt.plot(valid, label=f"valid_{name}")
    plt.ylabel(f"{name}s")
    plt.xlabel("epochs")
    plt.legend([f"train_{name}", f"valid_{name}"])
    plt.savefig(f"plots/{name}_{clust_id}.png")

def plot_cf_matrix(labels, preds, name):
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
        plt.savefig(f"plots/conf_matrix_{name}_{t}.png")

    return cms


def roc_curve_plot(testy, y_pred, roc_score, dataset, clust):
    
    lr_fpr, lr_tpr, _ = metrics.roc_curve(testy, y_pred)
    
    plt.clf()
    plt.plot(lr_fpr, lr_tpr, label='ROC AUC=%.3f' % (roc_score))

    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')

    plt.legend()
    plt.savefig(f"plots/roc_score_{dataset}_{clust}")


def get_false_test_valid_edges():
    adj_train, _, test_matrix, valid_matrix  = get_complete_data(DATASET_NAME, N_CLUSTERS, leave_intra_clust_edges=LEAVE_INTRA_CLUSTERS)

    train_edges, _, _ = sparse_to_tuple(adj_train)
    test_edges, _, _ = sparse_to_tuple(test_matrix)
    valid_edges, _, _ = sparse_to_tuple(valid_matrix)
    
    data = [1]*(len(train_edges) + len(test_edges) + len(valid_edges))
    indexes = np.concatenate((train_edges,test_edges, valid_edges), 0)

    complete_adj = csr_matrix((data, (indexes[:,0], indexes[:,1])), shape = adj_train.shape)


    false_edges = get_false_edges(complete_adj, test_edges.shape[0] + valid_edges.shape[0])

    valid_false_edges = false_edges[:valid_edges.shape[0]]
    test_false_edges = false_edges[valid_edges.shape[0]:]


    print(f"valid_edges: {valid_edges.shape[0]}")
    print(f"valid_false_edges: {valid_false_edges.shape[0]}")

    test_ones = [1]*test_false_edges.shape[0]
    valid_ones = [1]*valid_false_edges.shape[0]

    test_false_matrix = csr_matrix((test_ones, (test_false_edges[:,0], test_false_edges[:,1])), adj_train.shape)
    valid_false_matrix = csr_matrix((valid_ones, (valid_false_edges[:,0], valid_false_edges[:,1])), adj_train.shape)

    return test_false_matrix, valid_false_matrix


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

def initialize_train_data_and_model(features, adj_train, adj_train_norm, train_edges, valid_edges, valid_false_edges):
    optimizer = tf.keras.optimizers.Adam(learning_rate=LR)
    
    n_nodes = adj_train.shape[0]

    # convert the normalized adj and the features to tensors
    adj_train_norm_tensor = convert_sparse_matrix_to_sparse_tensor(adj_train_norm)
    feature_tensor = convert_sparse_matrix_to_sparse_tensor(features)

    # get train ground truth 
    train_y = np.reshape(adj_train.toarray(), (n_nodes*n_nodes))
    train_y = tf.convert_to_tensor(train_y, dtype=tf.float32)

    # define the model
    model = GAEModel(adj_train_norm_tensor)
    
    train_false_edges = get_false_edges(adj_train, len(train_edges))

    valid_edges_indeces = [x[0]*n_nodes + x[1] for x in valid_edges]
    valid_false_edges_indeces = [x[0]*n_nodes + x[1] for x in valid_false_edges]


    print(f"valid_edges_indeces: {len(valid_edges_indeces)}")
    print(f"valid_false_edges_indeces: {len(valid_false_edges_indeces)}")


    print("train_false_edges", train_false_edges.shape[0])
    print("train_pos_edges", len(train_edges))
    train_y_pos_edges = tf.convert_to_tensor([1.0]*len(train_edges))
    train_y_neg_edges = tf.convert_to_tensor([0.0]*train_false_edges.shape[0])

    return optimizer, model, train_false_edges, feature_tensor

def main(adjs, features_, tests, valids, clust_to_node, node_to_clust):

    test_false_matrix, valid_false_matrix = get_false_test_valid_edges()
    _, _, test_matrix_complete, _  = get_complete_data(DATASET_NAME, N_CLUSTERS, leave_intra_clust_edges=LEAVE_INTRA_CLUSTERS)

    n_test_edges = []
    n_valid_edges = []

    execution_times = []

    subset_lenghts = []
    
    models = []
    test_ps, test_ns = [], []
    test_p_intra, test_n_intra = 0, 0

    train_data = []

    for clust in range(len(adjs.keys())):
        print("\n")

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

        # initialize data and model
        opt, model, train_false_edges, feature_tensor = initialize_train_data_and_model(features, adj_train, adj_train_norm, train_edges, valid_edges, valid_false_edges)

        train_data.append({
            "time" : 0,
            "feature_tensor" : feature_tensor,
            "model" : model,
            "optimizer" : opt,
            "train_false_edges" : train_false_edges,
            "adj_train_norm" : adj_train_norm, 
            "train_edges" : train_edges, 
            "valid_edges" : valid_edges, 
            "valid_false_edges" : valid_false_edges, 
            "clust" : clust,
            "patience" : 0,
            "last_loss": None 
        })

    classifier = tf.keras.layers.Dense(N_CLUSTERS, activation=tf.nn.softmax)

    train_ended = [False] * N_CLUSTERS

    n_iterations = 0
    while sum(train_ended) < N_CLUSTERS:
        n_iterations += 1
        if n_iterations > EPOCHS: 
            break
        for clust in range(N_CLUSTERS):
            train_edges = train_data[clust]["train_edges"]
            valid_edges = train_data[clust]["valid_edges"]
            valid_false_edges = train_data[clust]["valid_false_edges"]

            patience = train_data[clust]["patience"]
            model = train_data[clust]["model"]
            optimizer = train_data[clust]["optimizer"]
            feature_tensor = train_data[clust]["feature_tensor"]
            train_false_edges = train_data[clust]["train_false_edges"]
            last_loss = train_data[clust]["last_loss"]

            if(patience < PATIENCE):
                # start training
                start_time = time.time()
                model, patience, last_loss = batched_train(model, optimizer, feature_tensor, train_edges, train_false_edges, valid_edges, valid_false_edges, clust, classifier, old_patience=patience, last_loss=last_loss)
                elapsed_time = time.time() - start_time

                train_data[clust]["model"] = model
                train_data[clust]["time"] = train_data[clust]["time"] + elapsed_time
                train_data[clust]["patience"] = patience
                train_data[clust]["last_loss"] = last_loss

                if patience >= PATIENCE:
                    train_ended[clust] = True

    models = [train_data[clust]["model"] for clust in range(N_CLUSTERS)]
    feature_tensors = [train_data[clust]["feature_tensor"] for clust in range(N_CLUSTERS)]
    execution_times = [train_data[clust]["time"] for clust in range(N_CLUSTERS)]

    test_matrix_complete = test_matrix_complete + test_matrix_complete.T
    test_false_matrix = test_false_matrix + test_false_matrix.T

    test_edges, test_false_edges = None, None
    for clust_1 in range(N_CLUSTERS):
        for clust_2 in range(clust_1, N_CLUSTERS):
            test_1_2 = test_matrix_complete[clust_to_node[clust_1],:][:, clust_to_node[clust_2]]
            test_1_2_false = test_false_matrix[clust_to_node[clust_1],:][:, clust_to_node[clust_2]]
            
            node_from, node_to = test_1_2.nonzero()
            node_from_false, node_to_false = test_1_2_false.nonzero()

            nodes_from_to = np.stack((node_from, node_to), 1)
            nodes_from_to_false = np.stack((node_from_false, node_to_false), 1)

            clusters = [[clust_1, clust_2]]*node_from.shape[0]
            clusters_false = [[clust_1, clust_2]]*node_from_false.shape[0]

            print(f"nodes_from_to: {nodes_from_to.shape},\t {nodes_from_to} ")
            print(f"nodes_from_to_false: {nodes_from_to_false.shape},\t {nodes_from_to_false}")

            if test_edges is None and nodes_from_to.shape[0] > 0:
                test_edges = np.concatenate((nodes_from_to, clusters), 1)
            elif nodes_from_to.shape[0] > 0:
                test_edges = np.concatenate((test_edges, np.concatenate((nodes_from_to, clusters), 1)), 0)

            if test_false_edges is None and nodes_from_to_false.shape[0] > 0:
                test_false_edges = np.concatenate((nodes_from_to_false, clusters_false), 1)
            elif nodes_from_to_false.shape[0] > 0:
                test_false_edges = np.concatenate((test_false_edges, np.concatenate((nodes_from_to_false, clusters_false), 1)), 0)
    
    model_name = "one_model_per_cluster"
    if MSE_LOSS:
        model_name += "_mse_loss"
    elif LABEL_OF_ALL_1 and not MSE_LOSS:
        model_name += "_const_labels_crossentropy_loss"
    else: 
        model_name += "_adv_loss_ohe"

    cms = test_adv_loss(models, test_edges, test_false_edges, feature_tensors, model_name)

    f1s, precs, recs = [], [], []

    ts = [0.5, 0.6, 0.7]
    for i in range(len(cms)):

        cm = cms[i]

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
        plt.savefig(f"plots/conf_matrix_{DATASET_NAME}_{model_name}_{ts[i]}.png")
        plt.close()
    
    dataset_folder_name = DATASET_NAME if DATASET_NAME != "amazon_electronics_computers" else "amazon_computers"
    with open(f"results/{dataset_folder_name}/{N_CLUSTERS}/{DATASET_NAME}_{model_name}.txt", "a") as fout:
        fout.write(f"precs: {precs}\n")
        fout.write(f"recs: {recs}\n")
        fout.write(f"f1s: {f1s}\n")
        fout.write(f"times: {sum(execution_times)}\n")
        fout.write("-"*10 + "\n")

    print("---execution times---")
    print(execution_times)

"""
    test_edges : [node_from, node_to, clust_from, clust_to]
        where node_from and node_to are wrt the cluster
"""
def test_adv_loss(models, test_edges, test_false_edges, feature_tensors, model_name):
    
    for i in range(len(models)):
        save_model(models[i], feature_tensors[i], DATASET_NAME, f"{model_name}_{i}", len(models))


    embs = [models[clust](feature_tensors[clust]) for clust in range(N_CLUSTERS)]

    preds, true_values = None, None

    for clust_1 in range(N_CLUSTERS):
        for clust_2 in range(clust_1, N_CLUSTERS):
            test_edges_1 = test_edges[:, 2] == clust_1
            test_edges_2 = test_edges[:, 3] == clust_2

            tmp_test_edges = test_edges[test_edges_1 * test_edges_2][:, :2]
            embs_from = tf.gather(embs[clust_1], tmp_test_edges[:,0])
            embs_to = tf.gather(embs[clust_2], tmp_test_edges[:,1])

            pos_preds = tf.sigmoid(tf.linalg.diag_part(tf.matmul(embs_from, embs_to, transpose_b=True)))

            test_edges_1 = test_false_edges[:, 2] == clust_1
            test_edges_2 = test_false_edges[:, 3] == clust_2

            tmp_test_edges = test_false_edges[test_edges_1 * test_edges_2][:, :2]
            embs_from = tf.gather(embs[clust_1], tmp_test_edges[:,0])
            embs_to = tf.gather(embs[clust_2], tmp_test_edges[:,1])

            neg_preds = tf.sigmoid(tf.linalg.diag_part(tf.matmul(embs_from, embs_to, transpose_b=True)))

            tmp_preds = tf.concat((pos_preds, neg_preds), 0)
            tmp_true_values = [1]*pos_preds.shape[0] + [0]*neg_preds.shape[0]

            if preds is not None:
                preds = tf.concat((preds, tmp_preds), 0)
                true_values += tmp_true_values
            else:
                preds, true_values = tmp_preds, tmp_true_values

    cms = plot_cf_matrix(np.array(true_values), preds.numpy(), model_name)

    for cm in cms: 
        tp, fp, fn = cm[1][1], cm[0][1], cm[1][0]
        prec, rec = tp/(tp + fp), tp/(tp+fn)
        f1 = 2*prec*rec/(prec+rec)
        
        print("--"*10)
        print(f"prec: {prec}")
        print(f"rec: {rec}")
        print(f"f1: {f1}")
        print("--"*10)
    
    return cms

if __name__ == "__main__":
    
    # load data
    single_data = load_data(DATASET_NAME, N_CLUSTERS, get_couples = False)

    adjs_single, features_single, tests_single, valids_single, clust_to_node_single, node_to_clust_single, _ = single_data

    main(adjs_single, features_single, tests_single, valids_single, clust_to_node_single, node_to_clust_single)

    

