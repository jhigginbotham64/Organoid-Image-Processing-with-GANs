# -*- coding: utf-8 -*-
"""Correct Copy of Organoid Image Colorization with GANs.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1rgtLfv1CnWKR3Pcttt41n0QqwuH1qu6L
"""

# external modules
from PIL import Image
from google.colab import drive # replace with other driver
import matplotlib.pyplot as plt # requires additional configuration
import tensorflow as tf
import numpy as np
import tensorflow.experimental.numpy as tnp

# native modules
import timeit 
from collections import defaultdict 
import datetime
import itertools
import uuid
import io
import os
import time

# Commented out IPython magic to ensure Python compatibility.
# base routes to help with further paths
cwd = Path(os.getcwd())
cwd = Path('/content')
mnt = cwd / "drive"

# path to shared data folder in google drive.
# accessing a shared folder from your personal
# drive requires right-clicking the folder and
# selecting "Add Shortcut to Drive" from the context
# menu. i added the 2021 srp folder to my root ("MyDrive")- DONE
# directory, so that's why it shows up this way.
d16 = mnt / "MyDrive" / "AI Summer 2021 - group view access" / "2021 Projects" / "Cardiac Organoid" / "Full dataset of D16"

# OIP AM group shared folder
am = mnt / "MyDrive" / "OIP" / "AM"

# path to directory where i saved the models i 
# trained near the end of srp.
mods = am / "models"

# gets appended to a "model batch" id, the {} gets filled 
# with the color. don't worry, you'll see later. :)
modsuffix = "_{}_colorizer" 

# mount google drive in colab session
#drive.mount(str(mnt))- Maybe can uncomment if this doesn't work
drive.mount('/content/drive')

# path to trace logs and other data generated by this notebook
logs = am / "logs"

def now():
    return datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

# log and summary writer for this particular run
log = logs / now()
w = tf.summary.create_file_writer(str(log))
w.set_as_default()

# callbacks for fit and evaluate
cbs =[
      tf.keras.callbacks.TensorBoard(
        log_dir=str(log), 
        histogram_freq=5, 
        write_graph=True, 
        write_images=True, 
        write_steps_per_second=True, 
        update_freq='epoch',),
      ]

log_dir="logs\\fit\\" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

# tensorboard
# %load_ext tensorboard
# %tensorboard --logdir $log
# %reload_ext tensorboard 
#Suggested by colab when I got error (line right above)

# resources to get you started on optimizing UNets, GANs, and CNNs:
# https://www.tensorflow.org/tutorials/generative/dcgan
# https://www.tensorflow.org/tutorials/generative/pix2pix
# https://medium.datadriveninvestor.com/an-overview-on-u-net-architecture-d6caabf7caa4
# https://stackoverflow.com/questions/56227915/how-to-correctly-use-batch-normalization-with-u-net-in-keras
# https://www.kaggle.com/dingdiego/u-net-batchnorm-augmentation-stratification
# https://towardsdatascience.com/colorizing-black-white-images-with-u-net-and-conditional-gan-a-tutorial-81b2df111cd8

# apparently the unet architecture was developed specifically
# for biomedical image segmentation with limited data sets.
# https://en.wikipedia.org/wiki/U-Net
def UNet(latent_dim, name):
    inputs = tf.keras.layers.Input(shape=(latent_dim, latent_dim, 1))
    downscaled = tf.keras.layers.experimental.preprocessing.Rescaling(scale=1./255)(inputs)

    conv1 = tf.keras.layers.Conv2D(16, kernel_size=(5, 5), strides=1)(downscaled)
    conv1 = tf.keras.layers.LeakyReLU()(conv1)
    conv1 = tf.keras.layers.Conv2D(32, kernel_size=(3, 3), strides=1)(conv1)
    conv1 = tf.keras.layers.LeakyReLU()(conv1)
    conv1 = tf.keras.layers.Conv2D(32, kernel_size=(3, 3), strides=1)(conv1)
    conv1 = tf.keras.layers.LeakyReLU()(conv1)

    conv2 = tf.keras.layers.Conv2D(32, kernel_size=(5, 5), strides=1)(conv1)
    conv2 = tf.keras.layers.LeakyReLU()(conv2)
    conv2 = tf.keras.layers.Conv2D(64, kernel_size=(3, 3), strides=1)(conv2)
    conv2 = tf.keras.layers.LeakyReLU()(conv2)
    conv2 = tf.keras.layers.Conv2D(64, kernel_size=(3, 3), strides=1)(conv2)
    conv2 = tf.keras.layers.LeakyReLU()(conv2)

    conv3 = tf.keras.layers.Conv2D(64, kernel_size=(5, 5), strides=1)(conv2)
    conv3 = tf.keras.layers.LeakyReLU()(conv3)
    conv3 = tf.keras.layers.Conv2D(128, kernel_size=(3, 3), strides=1)(conv3)
    conv3 = tf.keras.layers.LeakyReLU()(conv3)
    conv3 = tf.keras.layers.Conv2D(128, kernel_size=(3, 3), strides=1)(conv3)
    conv3 = tf.keras.layers.LeakyReLU()(conv3)
    
    bottleneck = tf.keras.layers.Conv2D(128, kernel_size=(3, 3), strides=1, activation='relu', padding='same')(conv3)

    cat1 = tf.keras.layers.Concatenate()([bottleneck, conv3])

    cup3 = tf.keras.layers.Conv2DTranspose(128, kernel_size=(3, 3), strides=1, activation='relu')(cat1)
    cup3 = tf.keras.layers.Conv2DTranspose(128, kernel_size=(3, 3), strides=1, activation='relu')(cup3)
    cup3 = tf.keras.layers.Conv2DTranspose(64, kernel_size=(5, 5), strides=1, activation='relu')(cup3)

    cat2 = tf.keras.layers.Concatenate()([cup3, conv2])

    cup2 = tf.keras.layers.Conv2DTranspose(64, kernel_size=(3, 3), strides=1, activation='relu')(cat2)
    cup2 = tf.keras.layers.Conv2DTranspose(64, kernel_size=(3, 3), strides=1, activation='relu')(cup2)
    cup2 = tf.keras.layers.Conv2DTranspose(32, kernel_size=(5, 5), strides=1, activation='relu')(cup2)

    cat3 = tf.keras.layers.Concatenate()([cup2, conv1])

    cup1 = tf.keras.layers.Conv2DTranspose(32, kernel_size=(3, 3), strides=1, activation='relu')(cat3)
    cup1 = tf.keras.layers.Conv2DTranspose(32, kernel_size=(3, 3), strides=1, activation='relu')(cup1)
    cup1 = tf.keras.layers.Conv2DTranspose(3, kernel_size=(5, 5), strides=1, activation='relu')(cup1)

    outputs = tf.keras.layers.experimental.preprocessing.Rescaling(scale=255.0)(cup1)

    return tf.keras.models.Model(inputs=inputs, outputs=outputs, name=name)

# pretty standard image classification stuff
# https://en.wikipedia.org/wiki/Convolutional_neural_network
def CNN(latent_dim, name):
    return tf.keras.models.Sequential(layers=[
        tf.keras.layers.InputLayer(input_shape=(latent_dim, latent_dim, 3)),
        tf.keras.layers.experimental.preprocessing.Rescaling(scale=1./255),
        tf.keras.layers.Conv2D(32, kernel_size=(7, 7), strides=1, activation='relu'),
        tf.keras.layers.Conv2D(32, kernel_size=(7, 7), strides=1, activation='relu'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(64, kernel_size=(5, 5), strides=1, activation='relu'),
        tf.keras.layers.Conv2D(64, kernel_size=(5, 5), strides=1, activation='relu'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(128, kernel_size=(3, 3), strides=1, activation='relu'),
        tf.keras.layers.Conv2D(128, kernel_size=(3, 3), strides=1, activation='relu'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(256, kernel_size=(3, 3), strides=1, activation='relu'),
        tf.keras.layers.Conv2D(256, kernel_size=(3, 3), strides=1, activation='relu'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(16, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ], name=name)

# blending tutorials for fun and profit (and SCIENCE!) :)
# original (also source for UNet and CNN layer code):
# https://colab.research.google.com/drive/1kMbrig8CMM5f2-hDTISSQmzaslG0LGlJ?usp=sharing
# GAN with model sublcassing:
# https://www.tensorflow.org/guide/keras/customizing_what_happens_in_fit#wrapping_up_an_end-to-end_gan_example
class GAN(tf.keras.Model):
    def __init__(self, name, latent_dim, generator, discriminator):
        super(GAN, self).__init__(name=name)
        self.indim = latent_dim
        self.gen = generator
        self.disc = discriminator
    
    def compile(self, g_err_fn, d_err_fn, g_optimizer, d_optimizer):
        super(GAN, self).compile()
        self.gerr = g_err_fn
        self.derr = d_err_fn
        self.gopt = g_optimizer
        self.dopt = d_optimizer
    
    def call(self, x, training=None, mask=None):
        return tf.cast(
            tf.clip_by_value(
                self.gen(
                    tf.cast(
                        tf.clip_by_value(x, 0, 255), 'uint8'),
                        training, mask), 0, 255), 'uint8')

    def generator_loss(self, fake_output, real_y):
        # known to be defined for loss = mean squared error
        fake_output = tf.cast(fake_output, 'float32')
        real_y = tf.cast(real_y, 'float32')
        fake_output = tf.keras.layers.experimental.preprocessing.Rescaling(scale=1./255)(fake_output)
        real_y = tf.keras.layers.experimental.preprocessing.Rescaling(scale=1./255)(real_y)
        return self.gerr(fake_output, real_y)

    def discriminator_loss(self, real_output, fake_output):
        # ditto as generator loss, but for binary cross entropy
        real_loss = self.derr(tf.ones_like(real_output) - tf.random.uniform(shape=(tf.shape(real_output)[0], 1), maxval=0.1), real_output)
        fake_loss = self.derr(tf.zeros_like(fake_output) + tf.random.uniform(shape=(tf.shape(fake_output)[0], 1), maxval=0.1), fake_output)
        total_loss = real_loss + fake_loss
        return total_loss

    def train_step(self, data):
        x, y = data
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            # generate an image -> G(x)
            generated_images = self.gen(x, training=True)
            # probability that the given image is real -> D(x)
            real_output = self.disc(y, training=True)
            # probability that the given image is the one generated -> D(G(x))
            generated_output = self.disc(generated_images, training=True)
            # L2 Loss -> || y - G(x) ||^2
            gen_loss = self.generator_loss(generated_images, y)
            # log loss for the discriminator
            disc_loss = self.discriminator_loss(real_output, generated_output)
        # compute gradients
        gradients_of_generator = gen_tape.gradient(gen_loss, self.gen.trainable_variables)
        gradients_of_discriminator = disc_tape.gradient(disc_loss, self.disc.trainable_variables)
        # run optimizers
        self.gopt.apply_gradients(zip(gradients_of_generator, self.gen.trainable_variables))
        self.dopt.apply_gradients(zip(gradients_of_discriminator, self.disc.trainable_variables))
        return {"gen_loss": gen_loss, "disc_loss": disc_loss}

# data pipeline, split into several stages (variables)
# for readability and ease of debugging and reuse.
# overall strategy is to meet project requirements while
# maximizing performance by leveraging tensorflow as
# much as freaking possible.
# base idea:
# https://www.tensorflow.org/guide/data#consuming_sets_of_files
# implemented several performance tips from docs:
# https://www.tensorflow.org/guide/data_performance
# also here for unexplained example of cache going before prefetch:
# https://www.tensorflow.org/tutorials/load_data/images#configure_the_dataset_for_performance

# map colorizations to cell types
grey = "PH" # phase
red = "mOr" # modified orange
green = "GFP" # green fluorescent protein
blue = "CFP" # cyan fluorescent protein
colors = [grey, red, green, blue] # for convenience

# helper variables for when we don't want to train all 3 models
targets = [red, green, blue]
input_and_targets = [grey] + targets

# fix image dimensions
img_scale = 8 # for easy adjusting
img_size = round(829 * (1 / img_scale)) # all images in data set are 829 x 829

# get file names
# deterministic dataset shuffles are important, because the plates need
# to be mixed up but images of the same specimen must stay together
# https://www.tensorflow.org/api_docs/python/tf/random/set_seed
tf.random.set_seed(seed=0xDEADBEEF)
# number of unique specimens in dataset

d16 = mnt / "MyDrive" / "AI Summer 2021 - group view access" / "2021 Projects" / "Cardiac Organoid" / "Full dataset of D16"
n = len(tf.data.Dataset.list_files(str(d16 / "*" / grey / "JPG" / "*.jpg"))) 

fnames = {c: tf.data.Dataset.list_files(
              str(d16 / "*" / c / "JPG" / "*.jpg"), shuffle=False
            ).shuffle(
              buffer_size=n, seed=0xDEADBEEF, reshuffle_each_iteration=False
            ) for c in input_and_targets}

# load images
def load_img(p):
    img = tf.io.read_file(p)
    if tf.strings.split(p, os.sep)[-3] == grey:
      img = tf.image.decode_image(img, channels=1, expand_animations=False)
    else:
      img = tf.image.decode_image(img, channels=3, expand_animations=False)
    img = tf.image.resize(img, size=(img_size, img_size), method=tf.image.ResizeMethod.LANCZOS5, antialias=True)
    img = tf.cast(tf.clip_by_value(img, 0, 255), 'uint8')
    return img

imgs = {c: fnames[c].map(
            load_img, num_parallel_calls=tf.data.AUTOTUNE, deterministic=True
          ) for c in input_and_targets}

# augment images
# your best bet for getting started with augmentation is this:
# https://www.tensorflow.org/tutorials/images/data_augmentation
def augmented_img_set(img):
  img_hflip = tf.image.flip_left_right(img)
  return tf.data.Dataset.from_tensor_slices([
      img,
      tf.image.rot90(img, k=1),
      tf.image.rot90(img, k=2),
      tf.image.rot90(img, k=3),
      img_hflip,
      tf.image.rot90(img_hflip, k=1),
      tf.image.rot90(img_hflip, k=2),
      tf.image.rot90(img_hflip, k=3),                       
  ])

aug_imgs = {c: imgs[c].flat_map(
            augmented_img_set
          ) for c in input_and_targets}

# batch, cache, prefetch
batch_size = 64 # size of training batches (technically a hyperparameter)

data = {c: aug_imgs[c].batch(
            batch_size=batch_size, num_parallel_calls=tf.data.AUTOTUNE, deterministic=True
          ).cache().prefetch(
            buffer_size=tf.data.AUTOTUNE
          ) for c in input_and_targets}

# data splits
# percent of data used for training, validation, and testing
train_size = 0.8
val_size = test_size = round((1 - train_size) / 2, ndigits=10)

# number of batches after augmentation, with ceil to account for
# batch sizes that don't evenly divide the number of instances
n_batches = int(np.ceil(n * 8 / batch_size))
# number of batches for training, validation, and testing,
# arranged slightly differently for different batch sizes,
# bearing in mind how batching works in tensorflow, and also
# recalling that visual quality is more important than any
# numerical metric, making it acceptable to bias towards training
n_val_batches = 1 if batch_size != 1 else int(np.floor(n_batches * val_size))
# round test batches up (or just add 1) due to possibly incomplete 
# final batch. it also gets a little more than val because the test 
# measurements are a little more important than validation measurements.
n_tst_batches = 2 if batch_size != 1 else int(np.ceil(n_batches * test_size))
n_trn_batches = n_batches - n_val_batches - n_tst_batches

# dataset navigation helpers
def get(d, i=0, n=1):
  return d.skip(i).take(n)
def get_unbatched(d, i=0, n=1):
  return get(d.unbatch(), i, n)
def trn_batches(c):
  return get(data[c], n=n_trn_batches)
def val_batches(c):
  return get(data[c], i=n_trn_batches, n=n_val_batches)
def tst_batches(c):
  return get(data[c], i=(n_trn_batches + n_val_batches), n=n_tst_batches)
def trn(c):
  return tf.data.Dataset.zip((trn_batches(grey), trn_batches(c)))
def val(c):
  return tf.data.Dataset.zip((val_batches(grey), val_batches(c)))
def tst(c):
  return tf.data.Dataset.zip((tst_batches(grey), tst_batches(c)))

# visualizations
pltimgsize = round(img_size * 0.05) # basic unit for subplot grid size calculations

# helper for formatting images stored as numpy arrays
def format_numpy_img(img):
  if img.shape[2] == 1: # grayscale
    return img.reshape(img_size, img_size)
  elif img.shape[2] == 3: # rgb
    return img

# quick-and-dirty helper for displaying single images
def show_numpy_img(img):
  image = format_numpy_img(img)
  if img.shape[2] == 1:
    plt.imshow(image, cmap='gray')
  elif img.shape[2] == 3:
    plt.imshow(image)
  plt.show()

# convert matplotlib figure to png for saving or else dumping to tensorboard:
# https://www.tensorflow.org/tensorboard/image_summaries
def plot_to_image(figure):
    """Converts the matplotlib plot specified by 'figure' to a PNG image and returns it."""
    buf = io.BytesIO()
    plt.savefig(buf, format='jpg', transparent=True, bbox_inches='tight', pad_inches=0.05)
    buf.seek(0)
    image = tf.image.decode_image(buf.getvalue(), channels=3)
    image = tf.expand_dims(image, 0)
    return image

# use this function to display grayscale inputs side-by-side with
# predictions and targets. y should be a dictionary containing predictions
# by color, and batch_fn should be one of the *_batches functions from earlier.
# scans (skips) to index i in the data and displays the next n image sets.
# show_predictions and show_labels turn their respective sets of color images
# on and off. the grayscale images are always shown.
def show_img_sets(y=None, batch_fn=val_batches, i=0, n=batch_size, 
                  show_inputs=True, show_predictions=True, show_labels=True,
                  show_titles=False, show_axes=False, tboard_name="prediction_run"):
    pltcols = 1 + len(targets) * (int(show_predictions) + int(show_labels))
    pltrows = n
    fig = plt.figure(figsize=(
        pltimgsize * pltcols, 
        pltimgsize * pltrows))
    rgbs = []
    if show_predictions and not show_labels:
      rgbs = [y[c] for c in targets]
    elif not show_predictions and show_labels:
      rgbs = [batch_fn(c).unbatch() for c in targets]
    elif show_predictions and show_labels:
      rgbs = itertools.chain.from_iterable(
                [(y[c], batch_fn(c).unbatch()) for c in targets])
    for j, r in get(tf.data.Dataset.zip(
        (batch_fn(grey).unbatch(),) + tuple(rgbs)
        ), i, n).enumerate().as_numpy_iterator():
        for k, img in enumerate(r):
            nimg = format_numpy_img(img)
            ax = fig.add_subplot(pltrows, pltcols, 1 + k + j * pltcols, aspect='equal')
            if not show_axes:
                ax.axis('off')
            if k == 0:
                ax.imshow(nimg, cmap='gray')
            else:
                ax.imshow(nimg)
    
    tf.summary.image(f"{tboard_name}_{now()}", plot_to_image(fig), step=0)

# hyperparameters
# learning rate
eta = 0.001
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
