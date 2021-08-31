from data import get_complete_data
import random

if __name__ == "__main__":
    datasets = ["fb", "amazon_electronics_computers.npz", "pubmed"]

    for dataset in datasets:
        dataset_name = dataset if ".npz" not in dataset else dataset[:-4]
        print(dataset_name)
        data = get_complete_data(dataset_name)
        for n_clusters in range(3,6):
            n_nodes = data[0].shape[0]
            
            with open(f"data/{dataset_name}_random/{n_clusters}/labels_{dataset}.csv", "w") as fout:
                for i in range(n_nodes):
                    fout.write(str(random.randint(0, n_clusters-1)) + "\n")