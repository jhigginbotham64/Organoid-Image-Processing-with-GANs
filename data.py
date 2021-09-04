# data.py
# data loading and batching

import tensorflow as tf
import numpy as np
from pathlib import Path
import os
import datetime
import itertools

from config import (
  img_size,
  red,
  green,
  blue,
  grey,
  targets,
  batch_size
)

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
# drive.mount('/content/drive')

# path to trace logs and other data generated by this notebook
logs = am / "logs"

def now():
    return datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

# log and summary writer for this particular run
# log = logs / now()
# w = tf.summary.create_file_writer(str(log))
# w.set_as_default()

# callbacks for fit and evaluate
# cbs =[
#       tf.keras.callbacks.TensorBoard(
#         log_dir=str(log), 
#         histogram_freq=5, 
#         write_graph=True, 
#         write_images=True, 
#         write_steps_per_second=True, 
#         update_freq='epoch',),
#       ]

# tensorboard
# %load_ext tensorboard
# %tensorboard --logdir $log
# %reload_ext tensorboard 
#Suggested by colab when I got error (line right above)

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

# get file names

# number of unique specimens in dataset
# d16 = mnt / "MyDrive" / "AI Summer 2021 - group view access" / "2021 Projects" / "Cardiac Organoid" / "Full dataset of D16"
# n = len(tf.data.Dataset.list_files(str(d16 / "*" / grey / "JPG" / "*.jpg"))) 

# fnames = {c: tf.data.Dataset.list_files(
#               str(d16 / "*" / c / "JPG" / "*.jpg"), shuffle=False
#             ).shuffle(
#               buffer_size=n, seed=0xDEADBEEF, reshuffle_each_iteration=False
#             ) for c in input_and_targets}

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

# imgs = {c: fnames[c].map(
#             load_img, num_parallel_calls=tf.data.AUTOTUNE, deterministic=True
#           ) for c in input_and_targets}

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

# aug_imgs = {c: imgs[c].flat_map(
#             augmented_img_set
#           ) for c in input_and_targets}


# batch, cache, prefetch

# data = {c: aug_imgs[c].batch(
#             batch_size=batch_size, num_parallel_calls=tf.data.AUTOTUNE, deterministic=True
#           ).cache().prefetch(
#             buffer_size=tf.data.AUTOTUNE
#           ) for c in input_and_targets}

# data splits
# percent of data used for training, validation, and testing
train_size = 0.8
val_size = test_size = round((1 - train_size) / 2, ndigits=10)

# number of batches after augmentation, with ceil to account for
# batch sizes that don't evenly divide the number of instances
# n_batches = int(np.ceil(n * 8 / batch_size))
# number of batches for training, validation, and testing,
# arranged slightly differently for different batch sizes,
# bearing in mind how batching works in tensorflow, and also
# recalling that visual quality is more important than any
# numerical metric, making it acceptable to bias towards training
# n_val_batches = 1 if batch_size != 1 else int(np.floor(n_batches * val_size))
# round test batches up (or just add 1) due to possibly incomplete 
# final batch. it also gets a little more than val because the test 
# measurements are a little more important than validation measurements.
# n_tst_batches = 2 if batch_size != 1 else int(np.ceil(n_batches * test_size))
# n_trn_batches = n_batches - n_val_batches - n_tst_batches

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