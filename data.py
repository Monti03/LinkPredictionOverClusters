import sqlite3
import numpy as np
import random 
from scipy import sparse as sp
import scipy.io as sio

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
def load_data(dataset_name):

    if(dataset_name == "cora"):
        return load_cora_data()
    elif(dataset_name == "pubmed"):
        return load_pubmed_data()
    elif(dataset_name == "citeseer"):
        return load_citeseer_data()
    elif dataset_name == "ppi":
        return load_ppi_data()
    elif dataset_name == "cnv8":
        return load_cnv8_data()
    elif dataset_name == "deezer":
        return load_deezer_data()
    elif dataset_name == "pubmed_sparsest_cut":
        return load_pubmed_data(sparsest_cut=True)
    else:
        raise NotImplementedError

def load_edges(file_name):
    with open(file_name) as fin:
        edges = []
        for line in fin:
            edges.append(line.split(','))
    edges = np.array(edges)
    return edges.astype(np.int)
    

def load_pubmed_data(sparsest_cut = False):
    folder = "pubmed" if sparsest_cut == False else "pubmed_sparsest_cut"
    data = sio.loadmat(f'{folder}/pubmed.mat')
    adj_complete = data['W'].toarray()
    
    res_edges = load_edges(f"{folder}/res_edges.csv")
    adj_complete = sp.csr_matrix((np.ones(res_edges.shape[0]), (res_edges[:, 0], res_edges[:, 1])), shape=adj_complete.shape)
    
    test_edges = load_edges(f"{folder}/test_edges.csv")
    test_matrix = sp.csr_matrix((np.ones(test_edges.shape[0]), (test_edges[:, 0], test_edges[:, 1])), shape=adj_complete.shape)
    
    valid_edges = load_edges(f"{folder}/train_edges.csv")
    valid_matrix = sp.csr_matrix((np.ones(valid_edges.shape[0]), (valid_edges[:, 0], valid_edges[:, 1])), shape=adj_complete.shape)

    adj_complete = adj_complete.toarray()

    # some values have a self loop, I remove it
    for i in range(adj_complete.shape[0]):
        adj_complete[i,i] = 0
    
    node_to_clust = {}
    clust_to_node = {}
    with open(f"{folder}/labels_pubmed.csv", "r") as fin:
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

    all_features = data['fea']
    clust_to_features = {}
    for key in clust_to_node.keys():
        tmp_feat = all_features[clust_to_node[key],:]
        clust_to_features[key] = sp.csr_matrix(tmp_feat)
    

    return clust_to_adj, clust_to_features, clust_to_test, clust_to_valid, clust_to_node, node_to_clust
    

def load_pred_labels(dataset_name):
    labels = []
    with open("{}_pred_labels.csv".format(dataset_name), "r") as fin:
        for label in fin:
            labels.append(label.strip())
        
    return labels

def load_citeseer_data():
    adj_shape = sio.loadmat("citeseer/citeseer.mat")["W"].shape

    res_edges = load_edges("citeseer/citeseer_res_edges.csv")
    adj_complete = sp.csr_matrix((np.ones(res_edges.shape[0]), (res_edges[:, 0], res_edges[:, 1])), shape=adj_shape)
    
    test_edges = load_edges("citeseer/citeseer_test_edges.csv")
    test_matrix = sp.csr_matrix((np.ones(test_edges.shape[0]), (test_edges[:, 0], test_edges[:, 1])), shape=adj_shape)
    
    valid_edges = load_edges("citeseer/citeseer_train_edges.csv")
    valid_matrix = sp.csr_matrix((np.ones(valid_edges.shape[0]), (valid_edges[:, 0], valid_edges[:, 1])), shape=adj_shape)

    adj_complete = adj_complete.toarray()

    # some values have a self loop, I remove it
    for i in range(adj_complete.shape[0]):
        adj_complete[i,i] = 0

    clust_to_node = {}
    node_to_clust = {}
    with open("citeseer/2/labels_citeseer.csv", "r") as fin:
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

    all_features = sio.loadmat("citeseer/citeseer.mat")["fea"]
    clust_to_features = {}
    for key in clust_to_node.keys():
        tmp_feat = all_features[clust_to_node[key],:]
        clust_to_features[key] = sp.csr_matrix(tmp_feat)

    return clust_to_adj, clust_to_features, clust_to_test, clust_to_valid, clust_to_node, node_to_clust


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

def get_complete_citeseer_data():
    
    data = sio.loadmat('citeseer/citeseer.mat')
    adj_complete = data['W']

    adj_shape = adj_complete.shape

    res_edges = load_edges("citeseer/citeseer_res_edges.csv")
    
    test_edges = load_edges("citeseer/citeseer_test_edges.csv")
    
    valid_edges = load_edges("citeseer/citeseer_train_edges.csv")
    
    node_to_clust = {}
    with open("citeseer/2/labels_citeseer.csv", "r") as fin:
        for i, line in enumerate(fin):
            clust = int(line.strip())
            node_to_clust[i] = clust

    
    adj_complete = sp.csr_matrix((np.ones(res_edges.shape[0]), (res_edges[:, 0], res_edges[:, 1])), shape=adj_shape)
        
    test_edges_indeces = [node_to_clust[int(x[0])] == node_to_clust[int(x[1])] for x in test_edges]
    test_edges = test_edges[test_edges_indeces]
    
    test_matrix = sp.csr_matrix((np.ones(test_edges.shape[0]), (test_edges[:, 0], test_edges[:, 1])), shape=adj_shape)

    valid_edges_indeces = [node_to_clust[int(x[0])] == node_to_clust[int(x[1])] for x in valid_edges]
    valid_edges = valid_edges[valid_edges_indeces]
    valid_matrix = sp.csr_matrix((np.ones(valid_edges.shape[0]), (valid_edges[:, 0], valid_edges[:, 1])), shape=adj_shape)
    
    all_features = sp.csr_matrix(data['fea'])
    
    return adj_complete, all_features, test_matrix, valid_matrix 

def get_complete_pubmed_data(sparsest_cut = False):
    folder = "pubmed" if sparsest_cut == False else "pubmed_sparsest_cut"
    
    data = sio.loadmat(f'{folder}/pubmed.mat')
    adj_complete = data['W'].toarray()

    adj_shape = adj_complete.shape

    res_edges = load_edges(f"{folder}/res_edges.csv")
    
    test_edges = load_edges(f"{folder}/test_edges.csv")
    
    valid_edges = load_edges(f"{folder}/train_edges.csv")
    
    node_to_clust = {}
    with open(f"{folder}/labels_pubmed.csv", "r") as fin:
        for i, line in enumerate(fin):
            clust = int(line.strip())
            node_to_clust[i] = clust

    
    adj_complete = sp.csr_matrix((np.ones(res_edges.shape[0]), (res_edges[:, 0], res_edges[:, 1])), shape=adj_shape)
        
    test_edges_indeces = [node_to_clust[int(x[0])] == node_to_clust[int(x[1])] for x in test_edges]
    test_edges = test_edges[test_edges_indeces]
    
    test_matrix = sp.csr_matrix((np.ones(test_edges.shape[0]), (test_edges[:, 0], test_edges[:, 1])), shape=adj_shape)

    valid_edges_indeces = [node_to_clust[int(x[0])] == node_to_clust[int(x[1])] for x in valid_edges]
    valid_edges = valid_edges[valid_edges_indeces]
    valid_matrix = sp.csr_matrix((np.ones(valid_edges.shape[0]), (valid_edges[:, 0], valid_edges[:, 1])), shape=adj_shape)
    
    all_features = sp.csr_matrix(data['fea'])
    
    return adj_complete, all_features, test_matrix, valid_matrix 

def load_ppi_data():
    #adj_complete = np.loadtxt(open("ppi/W.csv", "rb"), delimiter=",")
    features = np.load("ppi/ppi-feats.npy")
    adj_shape = (features.shape[0], features.shape[0])
    features = None
    
    print("loaded features")

    res_edges = load_edges("ppi/ppi_res_edges.csv")
    adj_complete = sp.csr_matrix((np.ones(res_edges.shape[0]), (res_edges[:, 0], res_edges[:, 1])), shape=adj_shape)
    
    test_edges = load_edges("ppi/ppi_test_edges.csv")
    test_matrix = sp.csr_matrix((np.ones(test_edges.shape[0]), (test_edges[:, 0], test_edges[:, 1])), shape=adj_shape)
    
    valid_edges = load_edges("ppi/ppi_train_edges.csv")
    valid_matrix = sp.csr_matrix((np.ones(valid_edges.shape[0]), (valid_edges[:, 0], valid_edges[:, 1])), shape=adj_shape)

    # some values have a self loop, I remove it
    for i in range(adj_complete.shape[0]):
        adj_complete[i,i] = 0

    clust_to_node = {}
    node_to_clust = {}
    with open("ppi/2/labels_ppi.csv", "r") as fin:
        for i, line in enumerate(fin):
            clust = int(line.strip())
            if(clust_to_node.get(clust) == None):
                clust_to_node[clust] = []
            clust_to_node[clust].append(i)

            node_to_clust[i] = clust
    
    for key in clust_to_node.keys():
        print(f"{key} shape: {len(clust_to_node[key])}")

    clust_to_adj = {}
    for key in clust_to_node.keys():
        tmp_adj = adj_complete[clust_to_node[key],:]
        tmp_adj = tmp_adj[:,clust_to_node[key]]
        clust_to_adj[key] = sp.csr_matrix(tmp_adj)
    
    adj_complete = None
    
    #test_matrix = test_matrix.toarray()
    clust_to_test = {}
    for key in clust_to_node.keys():
        tmp_adj = test_matrix[clust_to_node[key],:]
        tmp_adj = tmp_adj[:,clust_to_node[key]]        
        clust_to_test[key] = sp.csr_matrix(tmp_adj)
    test_matrix = None

    #valid_matrix = valid_matrix.toarray()
    clust_to_valid = {}
    for key in clust_to_node.keys():
        tmp_adj = valid_matrix[clust_to_node[key],:]
        tmp_adj = tmp_adj[:,clust_to_node[key]]
        clust_to_valid[key] = sp.csr_matrix(tmp_adj)
    valid_matrix = None
    
    all_features = np.load("ppi/ppi-feats.npy")

    clust_to_features = {}
    for key in clust_to_node.keys():
        tmp_feat = all_features[clust_to_node[key],:]
        clust_to_features[key] = sp.csr_matrix(tmp_feat)

    return clust_to_adj, clust_to_features, clust_to_test, clust_to_valid, clust_to_node, node_to_clust


def get_complete_ppi_data():
    all_features = np.load("ppi/ppi-feats.npy")
    adj_shape = (all_features.shape[0], all_features.shape[0])
    
    print("complete shape", adj_shape)

    all_features = sp.csr_matrix(all_features) 

    res_edges = load_edges("ppi/ppi_res_edges.csv")
    
    test_edges = load_edges("ppi/ppi_test_edges.csv")
    
    valid_edges = load_edges("ppi/ppi_train_edges.csv")
    
    node_to_clust = {}
    with open("ppi/2/labels_ppi.csv", "r") as fin:
        for i, line in enumerate(fin):
            clust = int(line.strip())
            node_to_clust[i] = clust

    
    adj_complete = sp.csr_matrix((np.ones(res_edges.shape[0]), (res_edges[:, 0], res_edges[:, 1])), shape=adj_shape)
        
    test_edges_indeces = [node_to_clust[int(x[0])] == node_to_clust[int(x[1])] for x in test_edges]
    test_edges = test_edges[test_edges_indeces]
    
    test_matrix = sp.csr_matrix((np.ones(test_edges.shape[0]), (test_edges[:, 0], test_edges[:, 1])), shape=adj_shape)

    valid_edges_indeces = [node_to_clust[int(x[0])] == node_to_clust[int(x[1])] for x in valid_edges]
    valid_edges = valid_edges[valid_edges_indeces]
    valid_matrix = sp.csr_matrix((np.ones(valid_edges.shape[0]), (valid_edges[:, 0], valid_edges[:, 1])), shape=adj_shape)
    
    return adj_complete, all_features, test_matrix, valid_matrix 

def load_cnv8_data():
    features = np.genfromtxt("cnv8/features.csv", delimiter=",")

    adj_shape = (features.shape[0], features.shape[0])
    features = None
    
    print("loaded features")

    res_edges = load_edges("cnv8/cnv8_res_edges.csv")
    adj_complete = sp.csr_matrix((np.ones(res_edges.shape[0]), (res_edges[:, 0], res_edges[:, 1])), shape=adj_shape)
    
    test_edges = load_edges("cnv8/cnv8_test_edges.csv")
    test_matrix = sp.csr_matrix((np.ones(test_edges.shape[0]), (test_edges[:, 0], test_edges[:, 1])), shape=adj_shape)
    
    valid_edges = load_edges("cnv8/cnv8_train_edges.csv")
    valid_matrix = sp.csr_matrix((np.ones(valid_edges.shape[0]), (valid_edges[:, 0], valid_edges[:, 1])), shape=adj_shape)

    print("setting to zero the main diagonal")

    # some values have a self loop, I remove it
    #for i in range(adj_complete.shape[0]):
    #adj_complete[i,i] = 0

    adj_complete.setdiag([0]*adj_shape[0])

    print("computing node to clust")

    clust_to_node = {}
    node_to_clust = {}
    with open("cnv8/6/labels_cnv8.csv", "r") as fin:
        for i, line in enumerate(fin):
            clust = int(line.strip())
            if(clust_to_node.get(clust) == None):
                clust_to_node[clust] = []
            clust_to_node[clust].append(i)

            node_to_clust[i] = clust
    
    for key in clust_to_node.keys():
        print(f"{key} shape: {len(clust_to_node[key])}")


    print("computing clust to adj")

    clust_to_adj = {}
    for key in clust_to_node.keys():
        tmp_adj = adj_complete[clust_to_node[key],:]
        tmp_adj = tmp_adj[:,clust_to_node[key]]
        clust_to_adj[key] = sp.csr_matrix(tmp_adj)
    
    adj_complete = None

    print("computing clust to test")
    #test_matrix = test_matrix.toarray()
    clust_to_test = {}
    for key in clust_to_node.keys():
        tmp_adj = test_matrix[clust_to_node[key],:]
        tmp_adj = tmp_adj[:,clust_to_node[key]]        
        clust_to_test[key] = sp.csr_matrix(tmp_adj)
    test_matrix = None

    #valid_matrix = valid_matrix.toarray()
    print("computing clust to valid")
    clust_to_valid = {}
    for key in clust_to_node.keys():
        tmp_adj = valid_matrix[clust_to_node[key],:]
        tmp_adj = tmp_adj[:,clust_to_node[key]]
        clust_to_valid[key] = sp.csr_matrix(tmp_adj)
    valid_matrix = None
    
    all_features = np.load("ppi/ppi-feats.npy")

    print("computing clust to features")
    clust_to_features = {}
    for key in clust_to_node.keys():
        tmp_feat = all_features[clust_to_node[key],:]
        clust_to_features[key] = sp.csr_matrix(tmp_feat)

    return clust_to_adj, clust_to_features, clust_to_test, clust_to_valid, clust_to_node, node_to_clust


def get_complete_cnv8():
    all_features = np.genfromtxt("cnv8/features.csv", delimiter=",")
    print("got features")
    all_features = sp.csr_matrix(all_features)
    
    adj_shape = (all_features.shape[0], all_features.shape[0])
    print("complete shape", adj_shape)

    print("load edges")
    res_edges = load_edges("cnv8/cnv8_res_edges.csv")
    
    test_edges = load_edges("cnv8/cnv8_test_edges.csv")
    
    valid_edges = load_edges("cnv8/cnv8_train_edges.csv")
    
    print("computing node to clust")
    node_to_clust = {}
    with open("cnv8/6/labels_cnv8.csv", "r") as fin:
        for i, line in enumerate(fin):
            clust = int(line.strip())
            node_to_clust[i] = clust

    print("computing adj complete")
    adj_complete = sp.csr_matrix((np.ones(res_edges.shape[0]), (res_edges[:, 0], res_edges[:, 1])), shape=adj_shape)
    
    print("computing test matrix")
    test_edges_indeces = [node_to_clust[int(x[0])] == node_to_clust[int(x[1])] for x in test_edges]
    test_edges = test_edges[test_edges_indeces]
    
    test_matrix = sp.csr_matrix((np.ones(test_edges.shape[0]), (test_edges[:, 0], test_edges[:, 1])), shape=adj_shape)
    
    print("computing valid matrix")
    valid_edges_indeces = [node_to_clust[int(x[0])] == node_to_clust[int(x[1])] for x in valid_edges]
    valid_edges = valid_edges[valid_edges_indeces]
    valid_matrix = sp.csr_matrix((np.ones(valid_edges.shape[0]), (valid_edges[:, 0], valid_edges[:, 1])), shape=adj_shape)
    
    return adj_complete, all_features, test_matrix, valid_matrix 

def load_deezer_data():

    res_edges = load_edges("deezer/deezer_res_edges.csv")
    test_edges = load_edges("deezer/deezer_test_edges.csv")
    valid_edges = load_edges("deezer/deezer_train_edges.csv")
    
    print(res_edges.__class__)
    print(res_edges[0][0].__class__)

    n_nodes = max(res_edges.max(), test_edges.max(), valid_edges.max()) + 1
    adj_shape = (n_nodes, n_nodes)
    
    test_matrix = sp.csr_matrix((np.ones(test_edges.shape[0]), (test_edges[:, 0], test_edges[:, 1])), shape=adj_shape)
    adj_complete = sp.csr_matrix((np.ones(res_edges.shape[0]), (res_edges[:, 0], res_edges[:, 1])), shape=adj_shape)
    valid_matrix = sp.csr_matrix((np.ones(valid_edges.shape[0]), (valid_edges[:, 0], valid_edges[:, 1])), shape=adj_shape)

    # some values have a self loop, I remove it
    for i in range(adj_complete.shape[0]):
        adj_complete[i,i] = 0

    clust_to_node = {}
    node_to_clust = {}
    with open("deezer/3/labels_deezer.csv", "r") as fin:
        for i, line in enumerate(fin):
            clust = int(line.strip())
            if(clust_to_node.get(clust) == None):
                clust_to_node[clust] = []
            clust_to_node[clust].append(i)

            node_to_clust[i] = clust
    
    for key in clust_to_node.keys():
        print(f"{key} shape: {len(clust_to_node[key])}")

    clust_to_adj = {}
    for key in clust_to_node.keys():
        tmp_adj = adj_complete[clust_to_node[key],:]
        tmp_adj = tmp_adj[:,clust_to_node[key]]
        clust_to_adj[key] = sp.csr_matrix(tmp_adj)
    
    adj_complete = None
    
    #test_matrix = test_matrix.toarray()
    clust_to_test = {}
    for key in clust_to_node.keys():
        tmp_adj = test_matrix[clust_to_node[key],:]
        tmp_adj = tmp_adj[:,clust_to_node[key]]        
        clust_to_test[key] = sp.csr_matrix(tmp_adj)
    test_matrix = None

    #valid_matrix = valid_matrix.toarray()
    clust_to_valid = {}
    for key in clust_to_node.keys():
        tmp_adj = valid_matrix[clust_to_node[key],:]
        tmp_adj = tmp_adj[:,clust_to_node[key]]
        clust_to_valid[key] = sp.csr_matrix(tmp_adj)
    valid_matrix = None
    
    #all_features = sp.diags([1]*n_nodes)

    clust_to_features = {}
    for key in clust_to_node.keys():
        #tmp_feat = #all_features[clust_to_node[key],:]
        clust_to_features[key] = sp.diags([1]*len(clust_to_node[key]))#sp.csr_matrix(tmp_feat)

    return clust_to_adj, clust_to_features, clust_to_test, clust_to_valid, clust_to_node, node_to_clust


def get_complete_deezer_data():
    
    print("load edges")
    res_edges = load_edges("deezer/deezer_res_edges.csv")
    
    test_edges = load_edges("deezer/deezer_test_edges.csv")
    
    valid_edges = load_edges("deezer/deezer_train_edges.csv")
    
    n_nodes = max(res_edges.max(), test_edges.max(), valid_edges.max()) + 1

    adj_shape = (n_nodes, n_nodes)
    print("complete shape", adj_shape)

    print("computing node to clust")
    node_to_clust = {}
    with open("deezer/3/labels_deezer.csv", "r") as fin:
        for i, line in enumerate(fin):
            clust = int(line.strip())
            node_to_clust[i] = clust

    print("computing adj complete")
    adj_complete = sp.csr_matrix((np.ones(res_edges.shape[0]), (res_edges[:, 0], res_edges[:, 1])), shape=adj_shape)
    
    print("computing test matrix")
    test_edges_indeces = [node_to_clust[int(x[0])] == node_to_clust[int(x[1])] for x in test_edges]
    test_edges = test_edges[test_edges_indeces]
    
    test_matrix = sp.csr_matrix((np.ones(test_edges.shape[0]), (test_edges[:, 0], test_edges[:, 1])), shape=adj_shape)
    
    print("computing valid matrix")
    valid_edges_indeces = [node_to_clust[int(x[0])] == node_to_clust[int(x[1])] for x in valid_edges]
    valid_edges = valid_edges[valid_edges_indeces]
    valid_matrix = sp.csr_matrix((np.ones(valid_edges.shape[0]), (valid_edges[:, 0], valid_edges[:, 1])), shape=adj_shape)
    
    all_features = sp.diags([1]*n_nodes)

    return adj_complete, all_features, test_matrix, valid_matrix 


def get_complete_data(dataset):
    if dataset == "pubmed":
        return get_complete_pubmed_data()
    elif dataset == "cora":
        return get_complete_cora_data()
    elif dataset == "citeseer":
        return get_complete_citeseer_data()
    elif dataset == "ppi":
        return get_complete_ppi_data()
    elif dataset == "cnv8":
        return get_complete_cnv8()
    elif dataset == "deezer":
        return get_complete_deezer_data()
    elif dataset == "pubmed_sparsest_cut":
        return get_complete_pubmed_data(sparsest_cut=True)
    else:
        raise Exception("unknown dataset")