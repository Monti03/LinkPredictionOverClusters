# learning rate
LR = 0.000_1 

# dataset to be used
DATASET_NAME = "pubmed"

# dropout rate
DROPOUT = 0.3
BATCH_SIZE = 10_000
# number of epochs
EPOCHS = 50_000

# output size of the first conv layer
CONV1_OUT_SIZE = 256 
CONV2_OUT_SIZE = 128 


# embedding size
CONV_MU_OUT_SIZE = 64 
CONV_VAR_OUT_SIZE = 64 

PATIENCE = 10 

SEED = 93

POS_WIGHT = 0

SHARE_FIRST = False

LEAVE_INTRA_CLUSTERS = True
COUPLES_TRAIN = False
MATRIX_OPERATIONS = False

SINGLE_MODELS = True

COUPLE_AND_SINGLE = True

FC_OUTPUT_DIMENSION = 64

USE_FCS = True

N_CLUSTERS = 3


# if true, than the adv loss is calculated considering 
# as lable [1/N_CLUSTERS]*N_CLUSTERS in order to give to all the nodes
# the same sort of class
LABEL_OF_ALL_1 = False

TRAIN_ALSO_CLASSIFIER = False

NUM_EPOCHS_ADV_LOSS = 7