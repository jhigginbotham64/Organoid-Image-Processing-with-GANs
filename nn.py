# nn.py
# neural network classes

import tensorflow as tf

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
