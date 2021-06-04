# learning rate
LR = 0.001 # pubmed ->0.001

# dataset to be used (cora or citeseer)
DATASET_NAME = "pubmed_leave_intra_clust"

# dropout rate
DROPOUT = 0.3
BATCH_SIZE = 5000
# number of epochs
EPOCHS = 200

# output size of the first conv layer
CONV1_OUT_SIZE = 1024 # pubmed -> 256
CONV2_OUT_SIZE = 512 # pubmed -> 64


# embedding size
CONV_MU_OUT_SIZE = 64 # pubmed -> 64
CONV_VAR_OUT_SIZE = 64 # pubmed -> 64

PATIENCE = 10 #pubmed -> 10

SEED = 93

POS_WIGHT = 0