import numpy as np
import tensorflow as tf
import time

from utils.utils import *

from networks.gae_model import GAEModel
from networks.shared_model import LastSharedWithAdversarialLoss

from trainers.GAETrainer import Trainer
from trainers.SharedLayerTrainer import SharedTrainer

from loss import topological_loss

class TrainerWithAdvLoss(Trainer):
    

    def __init__(self, model_name:str, batch_size:int, train_patience:int, max_epochs:int, lr:float, n_clusters:int, clust_id:int,
        adv_train_also_classifier:bool = False, adv_const_labesl:bool=True, n_adv_epochs:int=5):
        super(TrainerWithAdvLoss, self).__init__(model_name, batch_size, train_patience, max_epochs, lr)

        self.adv_train_also_classifier = adv_train_also_classifier
        self.adv_const_labesl = adv_const_labesl
        self.n_adv_epochs = n_adv_epochs
        self.n_clusters = n_clusters
        self.clust_id = clust_id


    def initialize_model(self):
        self.model = GAEModel(self.adj_train_norm_tensor)
        self.classifier = tf.keras.layers.Dense(self.n_clusters, activation=tf.nn.softmax)

        self.cluster_labels = []
        if self.adv_const_labesl:
            single_label = [1/self.n_clusters]*self.n_clusters 
        else:
            single_label = [0]*self.n_clusters
            single_label[self.clust_id] = 1  
        
        # self.feature_tensor.shape[0] = n_nodes inside the cluster
        self.cluster_labels = [single_label] * self.feature_tensor.shape[0]

    def __adv_backprop(self):
        with tf.GradientTape() as tape:  
            # get the embedding of each node
            embs = self.model(self.feature_tensor, training=True)
            predicted_clusters = self.classifier(embs)
            
            if self.adv_train_also_classifier:
                trainable_variables = self.model.trainable_variables + self.classifier.trainable_variables
            else: 
                trainable_variables = self.model.trainable_variables

            if self.adv_const_labesl:
                loss = tf.keras.losses.MeanSquaredError()(self.cluster_labels, predicted_clusters)
                grad = tape.gradient(loss, trainable_variables)
            else:
                loss = tf.keras.losses.CategoricalCrossentropy()(self.cluster_labels, predicted_clusters)
                grad = tape.gradient(-loss, trainable_variables)

            self.optimizer.apply_gradients(zip(grad, trainable_variables))
        
    def train(self):
        start_time = time.time()
        patience, n_epoch = 0, 0
        while patience <= self.train_patience and n_epoch < self.max_epochs:
            if n_epoch % self.n_adv_epochs != 0 or n_epoch == 0:
                self.train_loop()
                loss = self.valid_loop()
                if (loss > min(self.valid_losses)):
                    patience += 1
                else:
                    patience = 0
            else:
                self.__adv_backprop()

            n_epoch += 1    

            print(f"Epoch: {n_epoch}") 
            print(f"TRAIN_RESULTS:\n\ttrain_acc: {self.train_accs[-1]}\t train_loss: {self.train_losses[-1]}")
            print(f"VALID_RESULTS:\n\tvalid_acc: {self.valid_accs[-1]}\t valid_loss: {self.valid_losses[-1]}")
            print(f"------------------")
        self.execution_time = time.time() - start_time

class SharedTrainerWithAdvLoss(SharedTrainer):

    def __init__(self, share_last:bool, model_name:str, batch_size:int, train_patience:int, 
        max_epochs:int, lr:float, n_clusters:int, dataset:str, adv_train_also_classifier:bool = False, 
        adv_const_labesl:bool=True, n_adv_epochs:int=5):

        super(SharedTrainerWithAdvLoss, self).__init__(share_last, model_name, batch_size, train_patience, max_epochs, lr, n_clusters, dataset)  

        self.adv_train_also_classifier = adv_train_also_classifier
        self.adv_const_labesl = adv_const_labesl
        self.n_adv_epochs = n_adv_epochs

    def initialize_model(self):
        self.model = LastSharedWithAdversarialLoss(self.adj_train_norm_tensor)   
        n_nodes = [adj.shape[0] for adj in self.adj_train_norm_tensor]

        self.cluster_labels = []
        for i in range(self.n_clusters):
            if self.adv_const_labesl:
                ith_label = [1/self.n_clusters]*self.n_clusters
            else:
                ith_label = [0]*self.n_clusters
                ith_label[i] = 1
            self.cluster_labels += [ith_label] * n_nodes[i]   

        self.name = f"{self.dataset}_share_last_with_adv"
        if self.adv_const_labesl:
            self.name += "_const_labels"

    def __adv_backprop(self):
        with tf.GradientTape() as tape:  
            predicted_classes = []
            # predict the classes for each node
            for clust in range(self.n_clusters):
                predicted_classes.append(self.model(self.features_tensor[clust], cluster=clust, training=True, predict_cluster=True))

            predicted_clusters = tf.concat(predicted_classes, 0)
            
            if self.adv_train_also_classifier:
                trainable_variables = self.model.trainable_variables
            else: 
                trainable_variables = self.model.conv_1.trainable_variables


            if self.adv_const_labesl:
                loss = tf.keras.losses.MeanSquaredError()(self.cluster_labels, predicted_clusters)
                grad = tape.gradient(loss, trainable_variables)
            else:
                loss = tf.keras.losses.CategoricalCrossentropy()(self.cluster_labels, predicted_clusters)
                grad = tape.gradient(-loss, trainable_variables)

            self.optimizer.apply_gradients(zip(grad, trainable_variables))
            

    def train(self):
        start_time = time.time()
        patience, n_epoch = 0, 0
        while patience <= self.train_patience and n_epoch < self.max_epochs:
            if n_epoch % self.n_adv_epochs != 0 or n_epoch == 0:
                self.train_loop()
                loss = self.valid_loop()
                if (loss > min(self.valid_losses)):
                    patience += 1
                else:
                    patience = 0    

                print(f"Epoch: {n_epoch}") 
                print(f"TRAIN_RESULTS:\n\ttrain_acc: {self.train_accs[-1]}\t train_loss: {self.train_losses[-1]}")
                print(f"VALID_RESULTS:\n\tvalid_acc: {self.valid_accs[-1]}\t valid_loss: {self.valid_losses[-1]}")
                print(f"------------------")
            else:
                print(f"ADV EPOCH: {n_epoch}") 
                self.__adv_backprop()
            
            n_epoch += 1

        self.execution_time = time.time() - start_time