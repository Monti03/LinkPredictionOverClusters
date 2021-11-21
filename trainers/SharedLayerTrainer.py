import numpy as np
import tensorflow as tf
import time

from utils.utils import *

from networks.shared_model import LastShared, FirstShared

from trainers.MultiClustTrainer import MultiClustTrainer

from loss import topological_loss

class SharedTrainer(MultiClustTrainer):

    def __init__(self, share_last:bool, model_name:str, batch_size:int, train_patience:int, max_epochs:int, lr:float, n_clusters:int, dataset:str):
        super(SharedTrainer, self).__init__(model_name, batch_size, train_patience, max_epochs, lr, n_clusters, dataset)
        self.share_last = share_last

    def initialize_model(self):
        if self.share_last:
            self.model = LastShared(self.adj_train_norm_tensor)
        else:
            self.model = FirstShared(self.adj_train_norm_tensor)

        self.name = f"{self.dataset}_share_last" if self.share_last else f"{self.dataset}_share_first"

    


