import numpy as np
import tensorflow as tf
import time

from utils.utils import *

from networks.multiple_models import MultipleModels

from loss import topological_loss

class MultiClustTrainer():

    def __init__(self, model_name:str, batch_size:int, train_patience:int, max_epochs:int, lr:float, n_clusters:int, dataset:str):
        self.model_name = model_name
        self.batch_size = batch_size
        self.train_patience = train_patience
        self.max_epochs = max_epochs
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
        self.n_clusters = n_clusters
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
        raise Exception("Use one of the implementations of this class")        

    #clust_from:int, clust_to:int are used into the OnePerCLusterTrainer batch loop
    def batch_loop(self, embs_clust_from:tf.Tensor, embs_clust_to:tf.Tensor, batch_edges:np.ndarray, 
        batch_false_edges:np.ndarray, clust_from:int, clust_to:int):
        
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

        return self.batch_loop(embs_1, embs_2, batch_edges_c1_c2, batch_false_edges_c1_c2, clust_1, clust_2)

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

    def test(self, test_edges, test_false_edges, test):
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

        cms = plot_cf_matrix(test_gt, test_predictions, self.name, test)

        roc_score = metrics.roc_auc_score(test_gt, test_predictions)
        ap_score = metrics.average_precision_score(test_gt, test_predictions)

        return cms, roc_score, ap_score