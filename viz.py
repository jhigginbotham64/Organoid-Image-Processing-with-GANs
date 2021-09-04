# viz.py
# visualization

from config import img_size, batch_size
from data import val_batches

import matplotlib.pyplot as plt
import io

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
