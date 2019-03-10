import keras
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tqdm import tqdm
from keras.datasets import mnist
from keras.layers import Dense, Dropout, Input
from keras.models import Model, Sequential
from keras.layers.advanced_activations import LeakyReLU
from keras.optimizers import Adam



def load_data():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    x_train = ((x_train.astype(np.float32) - 127.5)/127.5).reshape(60000, 784)
    
    
    return (x_train, y_train, x_test, y_test)

def init_Generator():
    generator = Sequential()
    generator.add(Dense(units=256, input_dim=100))
    generator.add(LeakyReLU(0.2))

    generator.add(Dense(units=512))
    generator.add(LeakyReLU(0.2))

    generator.add(Dense(units=784, activation='tanh'))

    generator.compile(loss='binary_crossentropy', optimizer=Adam(lr=0.0002, beta_1=0.5))

    return generator

def init_Discriminator():
    discriminator = Sequential()
    discriminator.add(Dense(units=1024, input_dim=784))
    discriminator.add(LeakyReLU(0.2))
    discriminator.add(Dropout(0.3))

    discriminator.add(Dense(units=512))
    discriminator.add(LeakyReLU(0.2))
    discriminator.add(Dropout(0.3))

    discriminator.add(Dense(units=256))
    discriminator.add(LeakyReLU(0.2))

    discriminator.add(Dense(units=1, activation='sigmoid'))

    discriminator.compile(loss='binary_crossentropy', optimizer=Adam(lr=0.0002, beta_1=0.5))

    return discriminator


def init_GAN():
    generator = init_Generator()
    discriminator = init_Discriminator()

    discriminator.trainable = False

    gan_input = Input(shape=(100,))
    x = generator(gan_input)

    gan_output = discriminator(x)

    gan = Model(inputs=gan_input, outputs=gan_output)
    gan.compile(loss='binary_crossentropy', optimizer='adam')

    return gan, generator, discriminator

def train(epochs=1, batch_size=128):
    (x_train, _, _, _) = load_data()
    batch_count = x_train.shape[0] / batch_size

    gan, generator, discriminator = init_GAN()

    for e in range(1, epochs+1):
        print("Epoch {}".format(e))

        for _ in tqdm(range(batch_size)):
            noise = np.random.normal(0,1, [batch_size, 100])

            generated_images = generator.predict(noise)

            image_batch = x_train[np.random.randint(low=0, high=x_train.shape[0], size=batch_size)]


            X = np.concatenate([image_batch, generated_images])

            y_dis = np.zeros(2*batch_size)
            y_dis[:batch_size]=0.9

            discriminator.trainable = True

            discriminator.train_on_batch(X, y_dis)
            noise = np.random.normal(0,1, [batch_size, 100])
            y_gen = np.ones(batch_size)

            discriminator.trainable=False

            gan.train_on_batch(noise, y_gen)

            if e==1 or e % 20 == 0:
                plot_images(e, generator)


def plot_images(epoch, generator, examples=100, dim=(10,10), figsize=(10,10)):
    noise = np.random.normal(loc=0, scale=1, size=[examples, 100])
    generated_images = generator.predict(noise)
    generated_images = generated_images.reshape(100,28,28)
    plt.figure(figsize=figsize)

    for i in range(generated_images.shape[0]):
        plt.subplot(dim[0], dim[1], i+1)
        plt.imshow(generated_images[i], interpolation='nearest')
        plt.axis('off')

    plt.tight_layout()
    plt.savefig('generated_image_{}.png'.format(epoch))
    plt.close()



if __name__ == "__main__":
    train(400, 128)
