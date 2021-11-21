# LinkPredictionOverClusters
## Approach
In this repo I have developed different techniques that try to exploit the clusters obtained from a graph in order to improve link prediction results. The proposed models are the following ones:
- couple models: for each couple of clusters train a GAE model and predict the test edges between nodes inside a single cluster as avg of the predictions of the models trained over such cluster
- single models with fc: for each cluster I train a GAE model and for each couple of clusters I train a FC that is used to map the embeddings of each node over a common dimension with the nodes of the other cluster
- shared model: a model where one of the two convolutiona layers of the GAE model is shared among the different clusters
- shared model with adversary loss: in order to improve the precedent model we tried to use an adversary loss to let the embeddings in output from the shared layer be independent from the cluster from which they come
- single models with adversary loss: in this case we train one GAE model for each cluster and use an adversary loss to let the embeddings in output be independent from the cluster from which they come

## Instruction
- Shared Model: ``python3 train_shared_model.py``
- Shared Model with Adversary Loss: ``python3 train_shared_model.py --adv``
- Single Models with Adversary Loss: ``python3 train_one_per_clust_adv_loss.py``
- Single Models with FC: ``python3 train_single_models_and_fc_between.py --use_fc``
- Couples : ``python3 train_couples.py``

Other usefull parameters are:
- ``--test``: to run a quick test with a few epochs
- ``--dataset=`` to chose the dataset (from puubmed, [amazon electronics](https://github.com/spindro/AP-GCN/tree/master/data) and [facebook](http://snap.stanford.edu/data/facebook-large-page-page-network.html))
- ``--n_clusters=`` to chose the number of cluster to consider
