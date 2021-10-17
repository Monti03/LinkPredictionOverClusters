import numpy as np

def save_shared_model_embs(model, features, dataset, model_name):
	print(features.__class__)
	print(features[0].__class__)
	print(features[0].shape)

	for clust in range(len(features)):
		embs = model(features[clust], clust, training=False).numpy()

		np.savetxt(f"embeddings/{dataset}/{len(features)}/{model_name}_{clust}.csv", embs, delimiter=",")

def save_gae_model_embs(model, features, dataset, model_name, n_clusters):
	embs = model(features)
	np.savetxt(f"embeddings/{dataset}/{n_clusters}/{model_name}.csv", embs, delimiter=",")