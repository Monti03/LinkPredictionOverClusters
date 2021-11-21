import numpy as np
import tensorflow as tf
import time

from utils.utils import *

from networks.multiple_models import MultipleModels
from networks.between_cluster_fc import BetweenClusterFC

from trainers.MultiClustTrainer import MultiClustTrainer

from loss import topological_loss

class OnePerClustTrainer(MultiClustTrainer):

    def __init__(self, use_fc:bool, model_name:str, batch_size:int, train_patience:int, max_epochs:int, lr:float, n_clusters:int, dataset:str):
        super(OnePerClustTrainer, self).__init__(model_name, batch_size, train_patience, max_epochs, lr, n_clusters, dataset)
        self.use_fc = use_fc

    def initialize_model(self):
        self.model = MultipleModels(self.adj_train_norm_tensor)
        
        if self.use_fc:
            self.between_cluster_fc = BetweenClusterFC(self.n_clusters)
        self.name = f"{self.dataset}_one_per_cluster_with_fc" if self.use_fc else f"{self.dataset}_one_per_cluster_with_scalar"

    def batch_loop(self, embs_clust_from:tf.Tensor, embs_clust_to:tf.Tensor, batch_edges:np.ndarray, 
        batch_false_edges:np.ndarray, clust_from:int, clust_to:int): 

        if self.use_fc and clust_from!=clust_to:
            pos_preds = self.between_cluster_fc(embs_clust_from, embs_clust_to, batch_edges,       clust_from, clust_to)
            neg_preds = self.between_cluster_fc(embs_clust_from, embs_clust_to, batch_false_edges, clust_from, clust_to)
    
        else:
            node_from = batch_edges[:,0]
            node_to = batch_edges[:,1]
            
            node_from_false = batch_false_edges[:,0]
            node_to_false = batch_false_edges[:,1]

            embs_from = tf.gather(embs_clust_from, node_from)
            embs_to = tf.gather(embs_clust_to, node_to)

            embs_from_false = tf.gather(embs_clust_from, node_from_false)
            embs_to_false = tf.gather(embs_clust_to, node_to_false)

            pos_preds = tf.linalg.diag_part(tf.matmul(embs_from, embs_to, transpose_b=True))
            neg_preds = tf.linalg.diag_part(tf.matmul(embs_from_false, embs_to_false, transpose_b=True))

    
        # concatenate the predictions and build the gt
        predictions = tf.concat((pos_preds, neg_preds), 0)        
        batch_gt = tf.concat([tf.ones(pos_preds.shape[0]),tf.zeros(neg_preds.shape[0])], 0)

        return predictions, batch_gt
    


