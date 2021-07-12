# learning rate
LR = 0.000_1 # pubmed ->0.001

# dataset to be used
DATASET_NAME = "amazon_electronics_photo"

# dropout rate
DROPOUT = 0.3
BATCH_SIZE = 5000
# number of epochs
EPOCHS = 5_000

# output size of the first conv layer
CONV1_OUT_SIZE = 1024 # pubmed -> 1024
CONV2_OUT_SIZE = 512 # pubmed -> 512


# embedding size
CONV_MU_OUT_SIZE = 64 # pubmed -> 64
CONV_VAR_OUT_SIZE = 64 # pubmed -> 64

PATIENCE = 10 #pubmed -> 10

SEED = 93

POS_WIGHT = 0

SHARE_FIRST = False