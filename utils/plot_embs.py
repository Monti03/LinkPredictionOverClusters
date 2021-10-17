
from sklearn.manifold import TSNE
import seaborn as sns
import numpy as np
from umap import UMAP

from matplotlib import pyplot as plt

N_CLUSTERS = 3


def plot_embeddings_tsne_umap(embs_list, dataset, model_name):

	n_nodes = sum([emb.shape[0] for emb in embs_list])

	# TSNE
	embs_idx, last_index = [], 0
	for i in range(len(embs_list)):
		tmp_embs__idx = [False] * n_nodes 
		tmp_embs__idx[last_index : last_index + embs_list[i].shape[0]] = [True] *embs_list[i].shape[0]
		last_index = last_index + embs_list[i].shape[0]
		embs_idx.append(tmp_embs__idx)


	tsne = TSNE()
	embs = np.concatenate(embs_list)
	tsne_embs = tsne.fit_transform(embs)
	
	colors = ["red", "blue", "green", "black", "brown"]
	markers = [".", "o", "v", "p", "s"]
	for i in range(len(embs_list)):	
		plt.scatter(tsne_embs[embs_idx[i]][:,0],tsne_embs[embs_idx[i]][:,1], c=colors[i], marker=markers[i])
	
	plt.savefig(f"embs_plots/{N_CLUSTERS}/tsne_{dataset}_{model_name}.png")
	plt.clf()
	#UMAP
	umap = UMAP()
	umap_embs = umap.fit_transform(embs)
	colors = ["red", "blue", "green", "black", "brown"]
	markers = [".", "o", "v", "p", "s"]
	for i in range(len(embs_list)):	
		plt.scatter(umap_embs[embs_idx[i]][:,0],umap_embs[embs_idx[i]][:,1], c=colors[i], marker=markers[i])
	
	plt.savefig(f"embs_plots/{N_CLUSTERS}/umap_{dataset}_{model_name}.png")
	plt.clf()



if __name__ == "__main__":
	
	models = ["shared_with_all_labels_of_1", "shared_with_adversarial_loss", "shared", "shared_with_MSE_loss"]
	datasets = ["amazon_electronics_computers", "pubmed", "fb"]
	
	for model in models:
		for dataset in datasets:
			embs = []
			for clust in range(N_CLUSTERS):
				clust_embs = np.loadtxt(f"first_layer_out/{N_CLUSTERS}/{dataset}_{model}_{N_CLUSTERS}_{clust}.csv", delimiter=',')

				embs.append(clust_embs)

			plot_embeddings_tsne_umap(embs, dataset, model)			