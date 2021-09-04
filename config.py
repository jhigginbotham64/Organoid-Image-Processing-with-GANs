# config.py
# configuration parameters and environment variables

import tensorflow as tf

# map colorizations to cell types
grey = "PH" # phase
red = "mOr" # modified orange
green = "GFP" # green fluorescent protein
blue = "CFP" # cyan fluorescent protein
colors = [grey, red, green, blue] # for convenience

# helper variables for when we don't want to train all 3 models
targets = [red, green, blue]
input_and_targets = [grey] + targets

# deterministic dataset shuffles are important, because the plates need
# to be mixed up but images of the same specimen must stay together
# https://www.tensorflow.org/api_docs/python/tf/random/set_seed
tf.random.set_seed(seed=0xDEADBEEF)

# fix image dimensions
img_scale = 8 # for easy adjusting
img_size = round(829 * (1 / img_scale)) # all images in data set are 829 x 829

# hyperparameters

batch_size = 64 # size of training batches
eta = 0.001 # learning rate
# number of training iterations per model
epochs = {
    red: 20,
    green: 20,
    blue: 20,
}
# epoch settings used to create the srp batch of models, with my notes
# epochs = {
#     red: 100, # might be able to handle more, improvement remains steady the whole time
#     green: 30, # loss shoots up a few epochs after this, probably overfitting. also seems to be more affected by randomness than others, takes a few training attempts to get the best loss.
#     blue: 30, # improves very little before this and practically flatlines after, probably not learning at all
# }
