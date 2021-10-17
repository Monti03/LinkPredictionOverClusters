import matplotlib.pyplot as plt

import numpy as np
import sys
from os import listdir

def plot(values, names, dataset, measure, n_cluster):
    N = len(values)
    
    ind = np.arange(1) 
    width = 0.2
    
    for i in range(N):
        plt.bar(ind + width*i, values[i], width, label=names[i])

    plt.ylabel(f"{measure}")
    plt.title(dataset)
    print(ind + width/2)

    plt.tick_params(
    axis='x',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom=False,      # ticks along the bottom edge are off
    top=False,         # ticks along the top edge are off
    labelbottom=False) # labels along the bottom edge are off

    if measure is not "times":
        plt.ylim((0.7, 1))

    if measure != "times":
        plt.legend(loc='lower right')
    else:
        plt.legend(loc='upper right')

    #plt.show()
    plt.savefig(f"bar_plots/{dataset}_{n_cluster}_{measure}.png")
    plt.clf()

if __name__ == "__main__":

    datasets = ["pubmed", "fb", "amazon_computers"]
    n_clusters = [3,4,5]

    times = {
        "fb":     {3:385.4075713157654, 4:21.95679759979248, 5:23.27451252937317},
        "pubmed": {3:55.76260590553284, 4:3.301196813583374, 5:3.3920040130615234},
        "amazon_computers": {3:66.74398279190063, 4:4.18024754524231, 5:4.268364667892456},
    }

    model_names = {
        "complete": "complete_model",
        #"couple_and_single":"couple_for_diff_clusts_single_for_same_clust",
        #"only_couples": "only_couples",
        #"fc": "multiple_models_between_with_fc_all_test",
        #"scalar product": "multiple_models_between_as_product_all_test",
        #"share_first": "share_first_all_test",
        "share_last": "share_last_all_test",
        "share_last_adv_loss_ohe" : "share_last_with_adversarial_loss_all_test",
        "share_last_const_label" : "shared_with_all_labels_of_1"
        
    }

    model_names_keys = list(model_names.keys())

    data = {}

    for dataset in datasets:
        data[dataset] = {}
        for n_cluster in n_clusters:
            data[dataset][n_cluster] = {
                "precs": [],
                "recs":[],
                "f1s":[],
                "times":[]
            }
            for model_name in model_names_keys:
                dataset_complete = dataset if dataset != "amazon_computers" else "amazon_electronics_computers"
                
                if model_name == "complete":
                    file_name = f"results/{dataset}/3/{dataset_complete}_{model_names[model_name]}.txt"
                else: 
                    file_name = f"results/{dataset}/{n_cluster}/{dataset_complete}_{model_names[model_name]}.txt"
                
                with open(file_name) as fin:
                    for line in fin:
                        if(not line.strip() == "----------"):
                            measure, values = line.split(":")
                            if measure == "time" or measure == "times":
                                measure = "times"
                                value = eval(values.strip())

                                if model_name != "complete":
                                    value += times[dataset][n_cluster]

                            else:
                                print(measure, values)
                                value = eval(values.strip())[-1]
                            data[dataset][n_cluster][measure].append(value)

            for measure in data[dataset][n_cluster]:
                values = data[dataset][n_cluster][measure]
                plot(values, model_names_keys, dataset, measure, n_cluster)
                        
        
