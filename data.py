import sqlite3
import numpy as np
import random 
from scipy import sparse as sp
import scipy.io as sio
from scipy.sparse import data

import json
from load_from_npz import from_flat_dict

# get n random edges that have 0 value in adj
def get_false_edges(adj, n, node_to_clust=None):
    false_edges = []

    while len(false_edges) < n:
        r1 = random.randint(0, adj.shape[0]-1)
        r2 = random.randint(0, adj.shape[0]-1)
        
        # check that the edge is not present in the adj
        # and that is in the triu part
        if((node_to_clust is None or node_to_clust[r1] == node_to_clust[r2]) and adj[r1, r2] == 0 and r1<r2 ):
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
def load_data(dataset_name, get_couples = False):

    if(dataset_name == "cora"):
        return load_cora_data()
    elif(dataset_name == "pubmed" or dataset_name == "pubmed_leave_intra_clust"):
        return load_pubmed_data(get_couples = get_couples)
    elif dataset_name == "amazon_electronics_computers":
        return load_amazon_electronics_computers_data(get_couples = get_couples)
    elif dataset_name == "amazon_electronics_photo":
        return load_amazon_electronics_computers_data(is_photos=True, get_couples = get_couples)
    elif dataset_name == "fb":
        return load_fb_data(get_couples = get_couples)
    else:
        raise NotImplementedError

def load_edges(file_name):
    with open(file_name) as fin:
        edges = []
        for line in fin:
            edges.append(line.split(','))
    edges = np.array(edges)
    return edges.astype(np.int)
    

def load_pubmed_data(sparsest_cut = False, get_couples = False):
    folder = "pubmed" if sparsest_cut == False else "pubmed_sparsest_cut"
    data = sio.loadmat(f'data/{folder}/pubmed.mat')
    adj_complete = data['W'].toarray()
    all_features = data['fea']
    
    res_edges = load_edges(f"data/{folder}/res_edges.csv")
    adj_complete = sp.csr_matrix((np.ones(res_edges.shape[0]), (res_edges[:, 0], res_edges[:, 1])), shape=adj_complete.shape)
    
    test_edges = load_edges(f"data/{folder}/test_edges.csv")
    test_matrix = sp.csr_matrix((np.ones(test_edges.shape[0]), (test_edges[:, 0], test_edges[:, 1])), shape=adj_complete.shape)
    
    valid_edges = load_edges(f"data/{folder}/train_edges.csv")
    valid_matrix = sp.csr_matrix((np.ones(valid_edges.shape[0]), (valid_edges[:, 0], valid_edges[:, 1])), shape=adj_complete.shape)

    adj_complete = adj_complete.toarray()


    # some values have a self loop, I remove it
    for i in range(adj_complete.shape[0]):
        adj_complete[i,i] = 0
    
    com_idx_to_clust_idx = {}
    node_to_clust = {}
    clust_to_node = {}
    with open(f"data/{folder}/labels_pubmed.csv", "r") as fin:
        for i, line in enumerate(fin):
            clust = int(line.strip())
            if(clust_to_node.get(clust) == None):
                clust_to_node[clust] = []
            
            com_idx_to_clust_idx[i] = len(clust_to_node[clust])
            
            clust_to_node[clust].append(i)


            node_to_clust[i] = clust

    if get_couples:
        return get_data_couples(all_features, adj_complete, test_matrix, valid_matrix, node_to_clust, clust_to_node, com_idx_to_clust_idx)

    clust_to_adj = {}
    for key in clust_to_node.keys():
        tmp_adj = adj_complete[clust_to_node[key],:]
        tmp_adj = tmp_adj[:,clust_to_node[key]]
        tmp_adj = tmp_adj + tmp_adj.T
        tmp_adj = np.triu(tmp_adj)
        print("TRIUUUU")
        clust_to_adj[key] = sp.csr_matrix(tmp_adj)
    
    adj_complete = None
    
    test_matrix = test_matrix.toarray()
    clust_to_test = {}
    for key in clust_to_node.keys():
        tmp_adj = test_matrix[clust_to_node[key],:]
        tmp_adj = tmp_adj[:,clust_to_node[key]]       
        tmp_adj = tmp_adj + tmp_adj.T
        tmp_adj = np.triu(tmp_adj) 
        clust_to_test[key] = sp.csr_matrix(tmp_adj)
    test_matrix = None

    valid_matrix = valid_matrix.toarray()
    clust_to_valid = {}
    for key in clust_to_node.keys():
        tmp_adj = valid_matrix[clust_to_node[key],:]
        tmp_adj = tmp_adj[:,clust_to_node[key]]
        tmp_adj = tmp_adj + tmp_adj.T
        tmp_adj = np.triu(tmp_adj)
        clust_to_valid[key] = sp.csr_matrix(tmp_adj)
    valid_matrix = None

    clust_to_features = {}
    for key in clust_to_node.keys():
        tmp_feat = all_features[clust_to_node[key],:]
        clust_to_features[key] = sp.csr_matrix(tmp_feat)
    

    return clust_to_adj, clust_to_features, clust_to_test, clust_to_valid, clust_to_node, node_to_clust, com_idx_to_clust_idx
    

def load_pred_labels(dataset_name):
    labels = []
    with open("{}_pred_labels.csv".format(dataset_name), "r") as fin:
        for label in fin:
            labels.append(label.strip())
        
    return labels


def load_cora_data():
    pred_labels = load_pred_labels("CORA/cora2")
    
    adj_complete = np.loadtxt(open("CORA/W.csv", "rb"), delimiter=",")
    adj_shape = adj_complete.shape

    res_edges = load_edges("CORA/cora_res_edges.csv")
    adj_complete = sp.csr_matrix((np.ones(res_edges.shape[0]), (res_edges[:, 0], res_edges[:, 1])), shape=adj_shape)
    
    test_edges = load_edges("CORA/cora_test_edges.csv")
    test_matrix = sp.csr_matrix((np.ones(test_edges.shape[0]), (test_edges[:, 0], test_edges[:, 1])), shape=adj_shape)
    
    valid_edges = load_edges("CORA/cora_train_edges.csv")
    valid_matrix = sp.csr_matrix((np.ones(valid_edges.shape[0]), (valid_edges[:, 0], valid_edges[:, 1])), shape=adj_shape)

    adj_complete = adj_complete.toarray()

    # some values have a self loop, I remove it
    for i in range(adj_complete.shape[0]):
        adj_complete[i,i] = 0

    clust_to_node = {}
    node_to_clust = {}
    with open("CORA/labels_cora_2.csv", "r") as fin:
        for i, line in enumerate(fin):
            clust = int(line.strip())
            if(clust_to_node.get(clust) == None):
                clust_to_node[clust] = []
            clust_to_node[clust].append(i)

            node_to_clust[i] = clust

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

    all_features = np.loadtxt(open("CORA/fea.csv", "rb"), delimiter=",")
    clust_to_features = {}
    for key in clust_to_node.keys():
        tmp_feat = all_features[clust_to_node[key],:]
        clust_to_features[key] = sp.csr_matrix(tmp_feat)

    return clust_to_adj, clust_to_features, clust_to_test, clust_to_valid, clust_to_node, node_to_clust

def get_complete_cora_data():
        
    adj_complete = np.loadtxt(open("CORA/W.csv", "rb"), delimiter=",")
    adj_shape = adj_complete.shape

    res_edges = load_edges("CORA/cora_res_edges.csv")
    
    test_edges = load_edges("CORA/cora_test_edges.csv")
    
    valid_edges = load_edges("CORA/cora_train_edges.csv")
    
    node_to_clust = {}
    with open("CORA/labels_cora.csv", "r") as fin:
        for i, line in enumerate(fin):
            clust = int(line.strip())
            node_to_clust[i] = clust

    
    adj_complete = sp.csr_matrix((np.ones(res_edges.shape[0]), (res_edges[:, 0], res_edges[:, 1])), shape=adj_shape)
    
    all_features = np.loadtxt(open("CORA/fea.csv", "rb"), delimiter=",")
    all_features = sp.csr_matrix(all_features)
    
    test_edges_indeces = [node_to_clust[int(x[0])] == node_to_clust[int(x[1])] for x in test_edges]
    test_edges = test_edges[test_edges_indeces]
    
    test_matrix = sp.csr_matrix((np.ones(test_edges.shape[0]), (test_edges[:, 0], test_edges[:, 1])), shape=adj_shape)

    valid_edges_indeces = [node_to_clust[int(x[0])] == node_to_clust[int(x[1])] for x in valid_edges]
    valid_edges = valid_edges[valid_edges_indeces]
    valid_matrix = sp.csr_matrix((np.ones(valid_edges.shape[0]), (valid_edges[:, 0], valid_edges[:, 1])), shape=adj_shape)

    return adj_complete, all_features, test_matrix, valid_matrix 


def get_complete_pubmed_data(sparsest_cut = False, leave_intra_clust_edges = False):
    folder = "pubmed" if sparsest_cut == False else "pubmed_sparsest_cut"
    
    data = sio.loadmat(f'data/{folder}/pubmed.mat')
    adj_complete = data['W'].toarray()

    adj_shape = adj_complete.shape

    res_edges = load_edges(f"data/{folder}/res_edges.csv")
    
    test_edges = load_edges(f"data/{folder}/test_edges.csv")
    
    valid_edges = load_edges(f"data/{folder}/train_edges.csv")
    
    node_to_clust = {}
    with open(f"data/{folder}/labels_pubmed.csv", "r") as fin:
        for i, line in enumerate(fin):
            clust = int(line.strip())
            node_to_clust[i] = clust

    
    adj_complete = sp.csr_matrix((np.ones(res_edges.shape[0]), (res_edges[:, 0], res_edges[:, 1])), shape=adj_shape)

    if(not leave_intra_clust_edges): 
        test_edges_indeces = [node_to_clust[int(x[0])] == node_to_clust[int(x[1])] for x in test_edges]
        test_edges = test_edges[test_edges_indeces]
    
    test_matrix = sp.csr_matrix((np.ones(test_edges.shape[0]), (test_edges[:, 0], test_edges[:, 1])), shape=adj_shape)


    if(not leave_intra_clust_edges):
        valid_edges_indeces = [node_to_clust[int(x[0])] == node_to_clust[int(x[1])] for x in valid_edges]
        valid_edges = valid_edges[valid_edges_indeces]
    valid_matrix = sp.csr_matrix((np.ones(valid_edges.shape[0]), (valid_edges[:, 0], valid_edges[:, 1])), shape=adj_shape)
    
    all_features = sp.csr_matrix(data['fea'])
    
    return adj_complete, all_features, test_matrix, valid_matrix 



def get_complete_data(dataset_name, leave_intra_clust_edges=False):
    if dataset_name == "pubmed":
        return get_complete_pubmed_data(leave_intra_clust_edges=leave_intra_clust_edges)
    elif dataset_name == "cora":
        return get_complete_cora_data()
    elif dataset_name == "amazon_electronics_computers":
        return get_complete_amazon_electronics_computers_data(leave_intra_clust_edges=leave_intra_clust_edges)
    elif dataset_name == "amazon_electronics_photo":
        return get_complete_amazon_electronics_computers_data(is_photos=True, leave_intra_clust_edges=leave_intra_clust_edges)
    elif dataset_name == "fb":
        return get_complete_fb_data(leave_intra_clust_edges=leave_intra_clust_edges)
    else:
        raise Exception("unknown dataset")


# is_photos = True: use photos dataset data
# else use the computer one
def get_complete_amazon_electronics_computers_data(is_photos = False, leave_intra_clust_edges = False):
    
    path = "amazon_electronics_photo" if is_photos else "amazon_electronics_computers"
    number_of_clusters = 2 if is_photos else 3

    data = np.load(f"data/{path}/{path}.npz")
    data = from_flat_dict(data)

    all_features = data["attr_matrix"]

    print("load edges")
    res_edges = load_edges(f"data/{path}/{path}.npz_res_edges.csv")
    
    test_edges = load_edges(f"data/{path}/{path}.npz_test_edges.csv")
    
    valid_edges = load_edges(f"data/{path}/{path}.npz_train_edges.csv")
    
    n_nodes = all_features.shape[0]#max(res_edges.max(), test_edges.max(), valid_edges.max()) + 1

    adj_shape = (n_nodes, n_nodes)
    print("complete shape", adj_shape)

    print("computing node to clust")
    node_to_clust = {}
    with open(f"data/{path}/{number_of_clusters}/labels_{path}.npz.csv", "r") as fin:
        for i, line in enumerate(fin):
            clust = int(line.strip())
            node_to_clust[i] = clust

    print("computing adj complete")
    adj_complete = sp.csr_matrix((np.ones(res_edges.shape[0]), (res_edges[:, 0], res_edges[:, 1])), shape=adj_shape)
    
    print("computing test matrix")

    if(not leave_intra_clust_edges): 
        test_edges_indeces = [node_to_clust[int(x[0])] == node_to_clust[int(x[1])] for x in test_edges]
        test_edges = test_edges[test_edges_indeces]
    
    test_matrix = sp.csr_matrix((np.ones(test_edges.shape[0]), (test_edges[:, 0], test_edges[:, 1])), shape=adj_shape)
    
    print("computing valid matrix")
    if(not leave_intra_clust_edges): 
        valid_edges_indeces = [node_to_clust[int(x[0])] == node_to_clust[int(x[1])] for x in valid_edges]
        valid_edges = valid_edges[valid_edges_indeces]
    
    valid_matrix = sp.csr_matrix((np.ones(valid_edges.shape[0]), (valid_edges[:, 0], valid_edges[:, 1])), shape=adj_shape)


    return adj_complete, all_features, test_matrix, valid_matrix

def load_amazon_electronics_computers_data(is_photos = False, get_couples = False):

    path = "amazon_electronics_photo" if is_photos else "amazon_electronics_computers"
    number_of_clusters = 2 if is_photos else 3

    data = np.load(f"data/{path}/{path}.npz")
    data = from_flat_dict(data)
    
    adj_complete = data["adj_matrix"]
    all_features = data['attr_matrix']

    res_edges = load_edges(f"data/{path}/{path}.npz_res_edges.csv")
    adj_complete = sp.csr_matrix((np.ones(res_edges.shape[0]), (res_edges[:, 0], res_edges[:, 1])), shape=adj_complete.shape)
    
    test_edges = load_edges(f"data/{path}/{path}.npz_test_edges.csv")
    test_matrix = sp.csr_matrix((np.ones(test_edges.shape[0]), (test_edges[:, 0], test_edges[:, 1])), shape=adj_complete.shape)
    
    valid_edges = load_edges(f"data/{path}/{path}.npz_train_edges.csv")
    valid_matrix = sp.csr_matrix((np.ones(valid_edges.shape[0]), (valid_edges[:, 0], valid_edges[:, 1])), shape=adj_complete.shape)

    adj_complete = adj_complete.toarray()

    # some values have a self loop, I remove it
    for i in range(adj_complete.shape[0]):
        adj_complete[i,i] = 0
    
    com_idx_to_clust_idx = {}
    node_to_clust = {}
    clust_to_node = {}
    with open(f"data/{path}/{number_of_clusters}/labels_{path}.npz.csv", "r") as fin:
        for i, line in enumerate(fin):
            clust = int(line.strip())
            if(clust_to_node.get(clust) == None):
                clust_to_node[clust] = []

            com_idx_to_clust_idx[i] = len(clust_to_node[clust])
            
            clust_to_node[clust].append(i)

            node_to_clust[i] = clust

    if get_couples:
        return get_data_couples(all_features, adj_complete, test_matrix, valid_matrix, node_to_clust, clust_to_node, com_idx_to_clust_idx)

    clust_to_adj = {}
    for key in clust_to_node.keys():
        tmp_adj = adj_complete[clust_to_node[key],:]
        tmp_adj = tmp_adj[:,clust_to_node[key]]
        tmp_adj = tmp_adj + tmp_adj.T
        tmp_adj = np.triu(tmp_adj)
        clust_to_adj[key] = sp.csr_matrix(tmp_adj)
    
    adj_complete = None
    
    test_matrix = test_matrix.toarray()
    clust_to_test = {}
    for key in clust_to_node.keys():
        tmp_adj = test_matrix[clust_to_node[key],:]
        tmp_adj = tmp_adj[:,clust_to_node[key]]  
        tmp_adj = tmp_adj + tmp_adj.T
        tmp_adj = np.triu(tmp_adj)      
        clust_to_test[key] = sp.csr_matrix(tmp_adj)
    test_matrix = None

    valid_matrix = valid_matrix.toarray()
    clust_to_valid = {}
    for key in clust_to_node.keys():
        tmp_adj = valid_matrix[clust_to_node[key],:]
        tmp_adj = tmp_adj[:,clust_to_node[key]]
        tmp_adj = tmp_adj + tmp_adj.T
        tmp_adj = np.triu(tmp_adj)
        clust_to_valid[key] = sp.csr_matrix(tmp_adj)
    valid_matrix = None

    all_features = data['attr_matrix']
    clust_to_features = {}
    for key in clust_to_node.keys():
        tmp_feat = all_features[clust_to_node[key],:]
        clust_to_features[key] = sp.csr_matrix(tmp_feat)
    

    return clust_to_adj, clust_to_features, clust_to_test, clust_to_valid, clust_to_node, node_to_clust, com_idx_to_clust_idx


def load_musae_features(path):
    with open(path) as fin:
        features_dict = json.load(fin)

    features, lenghts  = [], []
    max_ = 0
    for i in range(len(features_dict)):
        feat = features_dict[str(i)]
        feat_len = len(feat)

        lenghts.append(feat_len)

        max_ = max(features_dict[str(i)] + [max_])
        
        assert len(features_dict[str(i)]) == feat_len

        features.append(feat)

    from_nodes = [i for i in range(len(lenghts)) for _ in range(lenghts[i])]
    to_nodes = [feat for i in range(len(features)) for feat in features[i]]

    feat = sp.csr_matrix(([1]*len(from_nodes),(from_nodes, to_nodes)), shape= (len(features), max_+1))

    return feat

def get_complete_fb_data(leave_intra_clust_edges = False):
    
    print("load edges")
    res_edges = load_edges("data/fb/fb_res_edges.csv")
    
    test_edges = load_edges("data/fb/fb_test_edges.csv")
    
    valid_edges = load_edges("data/fb/fb_train_edges.csv")
    
    n_nodes = max(res_edges.max(), test_edges.max(), valid_edges.max()) + 1

    adj_shape = (n_nodes, n_nodes)
    print("complete shape", adj_shape)

    print("computing node to clust")
    node_to_clust = {}
    with open("data/fb/3/labels_fb.csv", "r") as fin:
        for i, line in enumerate(fin):
            clust = int(line.strip())
            node_to_clust[i] = clust

    print("computing adj complete")
    adj_complete = sp.csr_matrix((np.ones(res_edges.shape[0]), (res_edges[:, 0], res_edges[:, 1])), shape=adj_shape)
    
    print("computing test matrix")
    if(not leave_intra_clust_edges): 
        test_edges_indeces = [node_to_clust[int(x[0])] == node_to_clust[int(x[1])] for x in test_edges]
        test_edges = test_edges[test_edges_indeces]
    
    test_matrix = sp.csr_matrix((np.ones(test_edges.shape[0]), (test_edges[:, 0], test_edges[:, 1])), shape=adj_shape)
    
    print("computing valid matrix")
    if(not leave_intra_clust_edges): 
        valid_edges_indeces = [node_to_clust[int(x[0])] == node_to_clust[int(x[1])] for x in valid_edges]
        valid_edges = valid_edges[valid_edges_indeces]
    valid_matrix = sp.csr_matrix((np.ones(valid_edges.shape[0]), (valid_edges[:, 0], valid_edges[:, 1])), shape=adj_shape)
    
    feat_file = "data/fb/musae_facebook_features.json"

    feat = load_musae_features(feat_file)
    
    return adj_complete, feat, test_matrix, valid_matrix

def load_fb_data(get_couples=False):

    feat_file = "data/fb/musae_facebook_features.json"

    all_features = load_musae_features(feat_file)

    n_nodes = all_features.shape[0]

    res_edges = load_edges(f"data/fb/fb_res_edges.csv")
    adj_complete = sp.csr_matrix((np.ones(res_edges.shape[0]), (res_edges[:, 0], res_edges[:, 1])), shape=(n_nodes, n_nodes))
    
    test_edges = load_edges(f"data/fb/fb_test_edges.csv")
    test_matrix = sp.csr_matrix((np.ones(test_edges.shape[0]), (test_edges[:, 0], test_edges[:, 1])), shape=(n_nodes, n_nodes))
    
    valid_edges = load_edges(f"data/fb/fb_train_edges.csv")
    valid_matrix = sp.csr_matrix((np.ones(valid_edges.shape[0]), (valid_edges[:, 0], valid_edges[:, 1])), shape=(n_nodes, n_nodes))

    adj_complete = adj_complete.toarray()

    # some values have a self loop, I remove it
    for i in range(n_nodes):
        adj_complete[i,i] = 0
    
    com_idx_to_clust_idx = {}

    node_to_clust = {}
    clust_to_node = {}
    with open(f"data/fb/3/labels_fb.csv", "r") as fin:
        for i, line in enumerate(fin):
            clust = int(line.strip())
            if(clust_to_node.get(clust) == None):
                clust_to_node[clust] = []

            com_idx_to_clust_idx[i] = len(clust_to_node[clust])
                
            clust_to_node[clust].append(i)

            node_to_clust[i] = clust

    if get_couples:
        return get_data_couples(all_features, adj_complete, test_matrix, valid_matrix, node_to_clust, clust_to_node, com_idx_to_clust_idx)

    clust_to_adj = {}
    for key in clust_to_node.keys():
        tmp_adj = adj_complete[clust_to_node[key],:]
        tmp_adj = tmp_adj[:,clust_to_node[key]]
        tmp_adj = tmp_adj + tmp_adj.T
        tmp_adj = np.triu(tmp_adj)
        clust_to_adj[key] = sp.csr_matrix(tmp_adj)
    
    adj_complete = None
    
    test_matrix = test_matrix.toarray()
    clust_to_test = {}
    for key in clust_to_node.keys():
        tmp_adj = test_matrix[clust_to_node[key],:]
        tmp_adj = tmp_adj[:,clust_to_node[key]]        
        tmp_adj = tmp_adj + tmp_adj.T
        tmp_adj = np.triu(tmp_adj)
        clust_to_test[key] = sp.csr_matrix(tmp_adj)
    test_matrix = None

    valid_matrix = valid_matrix.toarray()
    clust_to_valid = {}
    for key in clust_to_node.keys():
        tmp_adj = valid_matrix[clust_to_node[key],:]
        tmp_adj = tmp_adj[:,clust_to_node[key]]
        tmp_adj = tmp_adj + tmp_adj.T
        tmp_adj = np.triu(tmp_adj)
        clust_to_valid[key] = sp.csr_matrix(tmp_adj)
    valid_matrix = None


    clust_to_features = {}
    for key in clust_to_node.keys():
        tmp_feat = all_features[clust_to_node[key],:]
        clust_to_features[key] = sp.csr_matrix(tmp_feat)
    

    return clust_to_adj, clust_to_features, clust_to_test, clust_to_valid, clust_to_node, node_to_clust, com_idx_to_clust_idx

def get_data_couples(all_features, train_matrix, test_matrix, valid_matrix, node_to_clust, clust_to_node, com_idx_to_clust_idx):
    couple_clust_to_node = {}
    clust_to_adj = {}
    clust_to_features = {}
    clust_to_test = {}
    clust_to_valid = {}

    # map the clusters so that the smaller ones are at the beginning
    # so that the couples will be smaller
    sizes = np.array([len(clust_to_node[i]) for i in range(len(clust_to_node))])
    sorted = np.argsort(sizes).tolist()

    new_node_to_clust = {}
    for node in node_to_clust.keys():
        old_clust = node_to_clust[node]
        new_clust = sorted.index(old_clust)

        new_node_to_clust[node] = new_clust

    node_to_clust = new_node_to_clust

    new_clust_to_node = {}
    for old_clust in clust_to_node.keys():
        new_clust = sorted.index(old_clust)
        new_clust_to_node[new_clust] = clust_to_node[old_clust]

    clust_to_node = new_clust_to_node

    counter = 0

    for clust_1 in range(len(clust_to_node.keys())):
        for clust_2 in range(clust_1+1, len(clust_to_node.keys())):
            nodes = clust_to_node[clust_1] + clust_to_node[clust_2] 
            
            couple_features = sp.csr_matrix(all_features[nodes, :])
            
            couple_train = train_matrix[nodes, :][:, nodes]
            couple_train = np.triu(couple_train + couple_train.T)
            couple_train = sp.csr_matrix(couple_train)

            couple_test  = test_matrix.toarray()[nodes, :][:, nodes]
            couple_test = np.triu(couple_test + couple_test.T)
            couple_test = sp.csr_matrix(couple_test)


            couple_valid = valid_matrix.toarray()[nodes, :][:, nodes]
            couple_valid = np.triu(couple_valid + couple_valid.T)
            couple_valid = sp.csr_matrix(couple_valid)

            couple_clust_to_node[counter] = nodes
            
            clust_to_adj[counter] = couple_train
            clust_to_test[counter] = couple_test 
            clust_to_valid[counter] = couple_valid 
            clust_to_features[counter] = couple_features

            counter += 1

    return clust_to_adj, clust_to_features, clust_to_test, clust_to_valid, couple_clust_to_node, node_to_clust, com_idx_to_clust_idx
