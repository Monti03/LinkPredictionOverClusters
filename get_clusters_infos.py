
import scipy.sparse as sp
from data import *

def get_num_nodes_with_only_edges_inside_the_cluster(adj_complete, clusters_dict):
    clusters = list(clusters_dict.keys())

    n_zeros = []

    for cluster in clusters:
        #not_clust = [i for i in range(adj_complete.shape[0]) if i not in clusters_dict[cluster]]
        
        not_clust_idxs = []
        for not_clust in clusters:
            if not_clust != cluster:
                not_clust_idxs += clusters_dict[cluster]

        clust_adj = adj_complete[clusters[cluster], :][:, not_clust_idxs]
        zero_values = (clust_adj.sum(axis=1) == 0).sum()

        n_zeros.append(zero_values)
    
    return n_zeros



def get_num_ignored_edges(adj_complete, clusters_dict):

    clusters = list(clusters_dict.keys())

    n_clusters = len(clusters)

    n_edges = []

    for i in range(n_clusters):
        for j in range(i+1, n_clusters):
            
            i_j_adj = adj_complete[clusters_dict[i], : ][:, clusters_dict[j]]

            n_edges.append((i_j_adj != 0).sum())
            

    return n_edges

if __name__ == "__main__":
    for dataset in ["pubmed", "amazon_electronics_computers", "amazon_electronics_photo", "fb"]:
        print("\n==============================")
        print(f"dataset: {dataset}")
        print()
        train_matrix, _, test_matrix, valid_matrix = get_complete_data(dataset, leave_intra_clust_edges=True)
        
        print()

        train_matrix = train_matrix + train_matrix.T
        test_matrix = test_matrix + test_matrix.T
        valid_matrix = valid_matrix + valid_matrix.T

        adj_complete =  train_matrix + test_matrix + valid_matrix

        _, _, _, _, clust_to_node, _ = load_data(dataset)

        num_ignored_edges_comp = get_num_ignored_edges(adj_complete, clust_to_node) 
        num_ignored_edges_train = get_num_ignored_edges(train_matrix, clust_to_node) 
        num_ignored_edges_test = get_num_ignored_edges(test_matrix, clust_to_node) 
        num_ignored_edges_dev = get_num_ignored_edges(valid_matrix, clust_to_node) 
        
        num_nodes_with_only_edges_inside_the_cluster_comp = get_num_nodes_with_only_edges_inside_the_cluster(adj_complete, clust_to_node)
        num_nodes_with_only_edges_inside_the_cluster_train = get_num_nodes_with_only_edges_inside_the_cluster(train_matrix, clust_to_node)
        num_nodes_with_only_edges_inside_the_cluster_test = get_num_nodes_with_only_edges_inside_the_cluster(test_matrix, clust_to_node)
        num_nodes_with_only_edges_inside_the_cluster_dev = get_num_nodes_with_only_edges_inside_the_cluster(valid_matrix, clust_to_node)

        edges_comp = sp.csr_matrix.count_nonzero(sp.triu(adj_complete))
        edges_train = sp.csr_matrix.count_nonzero(sp.triu(train_matrix))
        edges_test = sp.csr_matrix.count_nonzero(sp.triu(test_matrix))
        edges_dev = sp.csr_matrix.count_nonzero(sp.triu(valid_matrix))
        
        tot_nodes = adj_complete.shape[0]


        print(f"\tnum_edges_comp: {edges_comp}")
        print(f"\tnum_edges_train: {edges_train}")
        print(f"\tnum_edges_test: {edges_test}")
        print(f"\tnum_edges_dev:  {edges_dev}")

        print()

        print(f"\tnum_ignored_edges_comp: {num_ignored_edges_comp}, {sum(num_ignored_edges_comp)}")
        print(f"\tnum_ignored_edges_train:{num_ignored_edges_train}, {sum(num_ignored_edges_train)}")
        print(f"\tnum_ignored_edges_test: {num_ignored_edges_test}, {sum(num_ignored_edges_test)}")
        print(f"\tnum_ignored_edges_dev:  {num_ignored_edges_dev}, {sum(num_ignored_edges_dev)}")

        print()

        
        print(f"\tnum_nodes_with_only_edges_inside_the_cluster_comp: {num_nodes_with_only_edges_inside_the_cluster_comp}, {sum(num_nodes_with_only_edges_inside_the_cluster_comp)}")
        #print(f"\tnum_nodes_with_only_edges_inside_the_cluster_train: {num_nodes_with_only_edges_inside_the_cluster_train},  {sum(num_nodes_with_only_edges_inside_the_cluster_train)}")
        #print(f"\tnum_nodes_with_only_edges_inside_the_cluster_test: {num_nodes_with_only_edges_inside_the_cluster_test}, {sum(num_nodes_with_only_edges_inside_the_cluster_test)}")
        #print(f"\tnum_nodes_with_only_edges_inside_the_cluster_dev:  {num_nodes_with_only_edges_inside_the_cluster_dev}, {sum(num_nodes_with_only_edges_inside_the_cluster_dev)}")
