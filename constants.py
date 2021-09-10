# learning rate
LR = 0.000_1 # pubmed ->0.001

# dataset to be used
DATASET_NAME = "amazon_electronics_computers_random"

# dropout rate
DROPOUT = 0.3
BATCH_SIZE = 10_000
# number of epochs
EPOCHS = 1

# output size of the first conv layer
CONV1_OUT_SIZE = 256 # pubmed -> 1024
CONV2_OUT_SIZE = 128 # pubmed -> 512


# embedding size
CONV_MU_OUT_SIZE = 64 # pubmed -> 64
CONV_VAR_OUT_SIZE = 64 # pubmed -> 64

PATIENCE = 10 #pubmed -> 10

SEED = 93

POS_WIGHT = 0

SHARE_FIRST = False

LEAVE_INTRA_CLUSTERS = True
COUPLES_TRAIN = False
MATRIX_OPERATIONS = True

SINGLE_MODELS = True

COUPLE_AND_SINGLE = True

FC_OUTPUT_DIMENSION = 64

USE_FCS = True

N_CLUSTERS = 4