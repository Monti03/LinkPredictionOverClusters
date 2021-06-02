# learning rate
LR = 0.001 # pubmed ->0.001

# dataset to be used (cora or citeseer)
DATASET_NAME = "pubmed_sparsest_cut"

# dropout rate
DROPOUT = 0.3
BATCH_SIZE = 5000
# number of epochs
EPOCHS = 100_000

# output size of the first conv layer
CONV1_OUT_SIZE = 256 # pubmed -> 256
CONV2_OUT_SIZE = 64 # pubmed -> 64


# embedding size
CONV_MU_OUT_SIZE = 16 # pubmed -> 64
CONV_VAR_OUT_SIZE = 16 # pubmed -> 64

PATIENCE = 20 #pubmed -> 10

SEED = 93

POS_WIGHT = 1