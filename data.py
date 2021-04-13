import sqlite3
import numpy as np
import random 
from scipy import sparse as sp
import scipy.io as sio

# get n random edges that have 0 value in adj
def get_false_edges(adj, n):
    false_edges = []

    while len(false_edges) < n:
        r1 = random.randint(0, adj.shape[0]-1)
        r2 = random.randint(0, adj.shape[0]-1)
        
        # check that the edge is not present in the adj
        # and that is in the triu part
        if(adj[r1, r2] == 0 and r1<r2):
            false_edges.append([r1,r2])
            
    return np.array(false_edges)

# sparse matrix to (coords of the edges, values of the edges, shape of the matrix)
def sparse_to_tuple(sparse_mx):
    if not sp.isspmatrix_coo(sparse_mx):
        sparse_mx = sparse_mx.tocoo()
    coords = np.vstack((sparse_mx.row, sparse_mx.col)).transpose()
    values = sparse_mx.data
    shape = sparse_mx.shape
    return coords, values, shape

# get adj_train, train a and test edges
def get_test_edges(adj, test_size=0.1, train_size=0.15):
    adj_ = sp.triu(adj)
    coords, _, shape = sparse_to_tuple(adj_)
    
    all_edges = coords
    
    # define the number of edges for train and test
    num_train = int(train_size*all_edges.shape[0])
    num_test = int(test_size*all_edges.shape[0])

    # shuffle the edges
    np.random.shuffle(all_edges)

    # get the first num_test edges (after shuffling)
    # as the test_edges (the positive ones)
    test_edges = all_edges[:num_test]
    # get the first num_train after the first num_test edges (after shuffling)
    # as the train_edges (the positive ones)
    train_edges = all_edges[num_test:num_test+num_train]
    res_edges = all_edges[num_test+num_train:]
    

    n_nodes = adj_.shape[0]
    # with this method we keed the proportions the same in res, train and test
    #n_false_train_edges = int(((n_nodes**2 - len(res_edges))/len(res_edges))*len(train_edges))
    #n_false_test_edges = int(((n_nodes**2 - len(res_edges))/len(res_edges))*len(test_edges))
    
    n_false_train_edges = len(train_edges)
    n_false_test_edges = len(test_edges)
    
    print(f"train_edges: {len(train_edges)}")
    print(f"res_edges: {len(res_edges)}")
    print(f"train_false: {n_false_train_edges}, test_false: {n_false_test_edges}")
    print(f"false: {n_false_train_edges + n_false_test_edges}")
    print(f"total_false: {n_nodes**2 - len(res_edges) - len(train_edges) - len(test_edges)}")
    
    # get random false edges
    false_edges = get_false_edges(adj, n_false_train_edges + n_false_test_edges)
    # split them into train and test
    test_false_edges = false_edges[:n_false_test_edges]
    train_false_edges = false_edges[n_false_test_edges:]
    

    # turn the remaning edges into a sparse matrix
    adj_train = sp.csr_matrix((np.ones(res_edges.shape[0]), (res_edges[:, 0], res_edges[:, 1])), shape=adj.shape)

    return adj_train, res_edges, train_edges, np.array(train_false_edges), test_edges, np.array(test_false_edges) 

    
# load data: adj_complete, features, labels, num of classes
def load_data(dataset_name):

    if(dataset_name == "cora"):
        return load_cora_data()
    elif(dataset_name == "pubmed"):
        return load_pubmed_data()
    else:
        raise NotImplementedError

def load_edges(file_name):
    with open(file_name) as fin:
        edges = []
        for line in fin:
            edges.append(line.split(','))
    edges = np.array(edges)
    return edges

def load_pubmed_data():
    data = sio.loadmat('pubmed/pubmed.mat')
    adj_complete = data['W'].toarray()
    
    res_edges = load_edges("pubmed/res_edges.csv")
    adj_complete = sp.csr_matrix((np.ones(res_edges.shape[0]), (res_edges[:, 0], res_edges[:, 1])), shape=adj_complete.shape)
    
    test_edges = load_edges("pubmed/test_edges.csv")
    test_matrix = sp.csr_matrix((np.ones(test_edges.shape[0]), (test_edges[:, 0], test_edges[:, 1])), shape=adj_complete.shape)
    
    valid_edges = load_edges("pubmed/train_edges.csv")
    valid_matrix = sp.csr_matrix((np.ones(valid_edges.shape[0]), (valid_edges[:, 0], valid_edges[:, 1])), shape=adj_complete.shape)

    adj_complete = adj_complete.toarray()

    # some values have a self loop, I remove it
    for i in range(adj_complete.shape[0]):
        adj_complete[i,i] = 0

    clust_to_node = {}
    with open("pubmed/labels_pubmed.csv", "r") as fin:
        for i, line in enumerate(fin):
            clust = int(line.strip())
            if(clust_to_node.get(clust) == None):
                clust_to_node[clust] = []
            clust_to_node[clust].append(i)

    clust_to_adj = {}
    for key in clust_to_node.keys():
        tmp_adj = adj_complete[clust_to_node[key],:]
        tmp_adj = tmp_adj[:,clust_to_node[key]]
        clust_to_adj[key] = sp.csr_matrix(tmp_adj)
    
    adj_complete = None
    
    test_matrix = test_matrix.toarray()
    clust_to_test = {}
    for key in clust_to_node.keys():
        tmp_adj = test_matrix[clust_to_node[key],:]
        tmp_adj = tmp_adj[:,clust_to_node[key]]        
        clust_to_test[key] = sp.csr_matrix(tmp_adj)
    test_matrix = None

    valid_matrix = valid_matrix.toarray()
    clust_to_valid = {}
    for key in clust_to_node.keys():
        tmp_adj = valid_matrix[clust_to_node[key],:]
        tmp_adj = tmp_adj[:,clust_to_node[key]]
        clust_to_valid[key] = sp.csr_matrix(tmp_adj)
    valid_matrix = None

    all_features = data['fea']
    clust_to_features = {}
    for key in clust_to_node.keys():
        tmp_feat = all_features[clust_to_node[key],:]
        clust_to_features[key] = sp.csr_matrix(tmp_feat)
    

    return clust_to_adj, clust_to_features, clust_to_test, clust_to_valid
    

def load_pred_labels(dataset_name):
    labels = []
    with open("{}_pred_labels.csv".format(dataset_name), "r") as fin:
        for label in fin:
            labels.append(label.strip())
        
    return labels

def load_cora_data():
    pred_labels = load_pred_labels("CORA/cora2")
    
    adj_complete = np.loadtxt(open("CORA/W.csv", "rb"), delimiter=",")
    
    n_clusters = len(set(pred_labels))

    clust_to_node = {}
    for i, cluster in enumerate(pred_labels):
        if(clust_to_node.get(int(cluster)) == None):
            clust_to_node[int(cluster)] = []
        clust_to_node[int(cluster)].append(i)        

    clust_to_adj = {}
    for key in clust_to_node.keys():
        tmp_adj = adj_complete[clust_to_node[key],:]
        tmp_adj = tmp_adj[:,clust_to_node[key]]
        clust_to_adj[key] = sp.csr_matrix(tmp_adj)

    all_features = np.loadtxt(open("CORA/fea.csv", "rb"), delimiter=",")
    
    clust_to_features = {}
    for key in clust_to_node.keys():
        tmp_feat = all_features[clust_to_node[key],:]
        clust_to_features[key] = sp.csr_matrix(tmp_feat)
    
    return clust_to_adj, clust_to_features
