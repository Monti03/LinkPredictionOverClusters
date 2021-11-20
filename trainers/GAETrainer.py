import numpy as np
import tensorflow as tf
import time

from utils.utils import *
from utils.data import sparse_to_tuple, get_false_edges

from networks.gae_model import GAEModel

from loss import topological_loss

class Trainer():
    def __init__(self, model_name:str, batch_size:int, train_patience:int, max_epochs:int, lr:float):
        self.model_name = model_name
        self.batch_size = batch_size
        self.train_patience = train_patience
        self.max_epochs = max_epochs
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=lr)

    def prepare_data(self, adj_train:sp.csr_matrix, features:sp.csr_matrix, valid_edges:np.ndarray, valid_false_edges:np.ndarray):
        self.train_accs, self.train_losses = [], []

        self.valid_accs, self.valid_losses = [], [] 

        # get the normalized adj matrix
        adj_train_norm = compute_adj_norm(adj_train)

        # convert the normalized adj and the features to tensors
        self.adj_train_norm_tensor = convert_sparse_matrix_to_sparse_tensor(adj_train_norm)
        self.feature_tensor = convert_sparse_matrix_to_sparse_tensor(features)

        # get the positive train edges
        self.train_edges, _, _ = sparse_to_tuple(adj_train)
        
        # train false edges are chosen here because we can also considere false 
        # some edges that are in valid or test set as true edges
        # get the train false edges
        self.train_false_edges = get_false_edges(adj_train, len(self.train_edges))

        self.valid_edges = valid_edges
        self.valid_false_edges = valid_false_edges

        self.initialize_model()

    def initialize_model(self):
        self.model = GAEModel(self.adj_train_norm_tensor)

            

    def __batch_loop(self, training:bool, batch_edges:np.ndarray, batch_false_edges:np.ndarray):
        # forward pass
        embs = self.model(self.feature_tensor, training=training)
        
        node_from = batch_edges[:,0]
        node_to = batch_edges[:,1]

        # get the embedding of the nodes inside the batch_edges
        embs_from = tf.gather(embs, node_from)
        embs_to = tf.gather(embs, node_to)

        # get the predictions of batch_edges
        pos_preds = tf.linalg.diag_part(tf.matmul(embs_from, embs_to, transpose_b=True))
        
        # same for the false edges
        node_from = batch_false_edges[:,0]
        node_to = batch_false_edges[:,1]

        embs_from = tf.gather(embs, node_from)
        embs_to = tf.gather(embs, node_to)

        neg_preds = tf.linalg.diag_part(tf.matmul(embs_from, embs_to, transpose_b=True))
    
        # concatenate the predictions and build the gt
        predictions = tf.concat((pos_preds, neg_preds), 0)
        batch_gt = tf.concat([tf.ones(pos_preds.shape[0]),tf.zeros(neg_preds.shape[0])], 0)

        # get loss
        loss = topological_loss(batch_gt, predictions)

        # get acc
        ta = tf.keras.metrics.Accuracy()
        ta.update_state(batch_gt, tf.round(tf.nn.sigmoid(predictions)))
        acc = ta.result().numpy()

        return acc, loss


    def train_loop(self):

        # shuffle data in order to have different batches in different epochs
        np.random.shuffle(self.train_edges)      
        np.random.shuffle(self.train_false_edges)
        
        batch_losses, batch_accs = [], []

        for i in range(0, min(len(self.train_edges), self.train_false_edges.shape[0]), self.batch_size):
            batch_edges = self.train_edges[i: i+self.batch_size]    
            batch_false_edges = self.train_false_edges[i: i+self.batch_size]            
        
            with tf.GradientTape() as tape:  
                acc, loss = self.__batch_loop(True, batch_edges, batch_false_edges)
        
                grad = tape.gradient(loss, self.model.trainable_variables)
            
                # optimize the weights
                self.optimizer.apply_gradients(zip(grad, self.model.trainable_variables))

                batch_losses.append(loss.numpy())
                batch_accs.append(acc)

        self.train_losses.append(sum(batch_losses)/len(batch_accs))
        self.train_accs.append(sum(batch_accs)/len(batch_accs))


    def valid_loop(self):        
        acc, loss = self.__batch_loop(False, self.valid_edges, self.valid_false_edges)

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