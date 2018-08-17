import keras
from keras.datasets import mnist 
from keras.optimizers import Adam
from keras.layers import Dense, Input
from keras.models import Model, Sequential
from keras import initializers
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.core import Dropout

from tqdm import tqdm
import matplotlib.pyplot as plt 
import numpy as np 
from random import shuffle
import shutil,os

(X_train, Y_train), (X_test, Y_test) = mnist.load_data()


X_train = (X_train.astype(np.float32)-127.5)/127.5
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1]*X_train.shape[2])

randomDim = 10
adam = Adam(lr=1e-4)

generator = Sequential()
generator.add(Dense(256, input_dim = randomDim, kernel_initializer = initializers.glorot_normal()))
generator.add(LeakyReLU(0.2))
generator.add(Dense(512))
generator.add(LeakyReLU(0.2))
generator.add(Dense(1024))
generator.add(LeakyReLU(0.2))
generator.add(Dense(784, activation='tanh'))
generator.compile(loss='binary_crossentropy', optimizer = adam)

discriminator = Sequential()
discriminator.add(Dense(1024, input_dim=784, kernel_initializer=initializers.glorot_normal()))
discriminator.add(LeakyReLU(0.2))
discriminator.add(Dropout(0.3))
discriminator.add(Dense(512))
discriminator.add(LeakyReLU(0.2))
discriminator.add(Dropout(0.3))
discriminator.add(Dense(256))
discriminator.add(LeakyReLU(0.2))
discriminator.add(Dropout(0.3))
discriminator.add(Dense(1, activation='sigmoid'))
discriminator.compile(loss='binary_crossentropy', optimizer=adam)

discriminator.trainable = False
ganInput = Input(shape=(randomDim,))
x = generator(ganInput)
ganOutput = discriminator(x)
gan = Model(inputs=ganInput, outputs=ganOutput)
gan.compile(loss='binary_crossentropy', optimizer=adam)

dLosses = []
gLosses = []

def plotLoss(epoch):
	plt.figure(figsize=(10, 8))
	plt.plot(dLosses, label='Discriminitive loss')
	plt.plot(gLosses, label='Generative loss')
	plt.xlabel('Epoch')
	plt.ylabel('Loss')
	plt.legend()
	plt.savefig('plots/gan_loss_epoch_%d.png' % epoch)

def plotGeneratedImages(epoch, examples=100, dim=(10, 10), figsize=(10, 10)):
	noise = np.random.normal(0, 1, size=[examples, randomDim])
	generatedImages = generator.predict(noise)
	generatedImages = generatedImages.reshape(examples, 28, 28)

	plt.figure(figsize=figsize)
	for i in range(generatedImages.shape[0]):
		plt.subplot(dim[0], dim[1], i+1)
		plt.imshow(generatedImages[i], interpolation='nearest', cmap='gray_r')
		plt.axis('off')
	plt.tight_layout()
	plt.savefig('generated_images/gan_generated_image_epoch_%d.png' % epoch)

def saveModels(epoch):
	generator.save('models/gan_generator_epoch_%d.h5' % epoch)
	discriminator.save('models/gan_discriminator_epoch_%d.h5' % epoch)

def train(epoch=1, batchSize=200):
	batchCount = int(X_train.shape[0] / batchSize)
	print ('Epochs:', epoch)
	print ('Batch size:', batchSize)
	print ('Batches per epoch:', batchCount)

	for i in range(epoch):
		print ('-'*15, 'Epoch %d' % i, '-'*15)
		for _ in tqdm(range(batchCount)):
			noise = np.random.normal(0,1,size=[batchSize,randomDim])
			imageBatch = X_train[_*batchSize:_*batchSize + batchSize]


			generatedImages = generator.predict(noise)
			X = np.concatenate([imageBatch, generatedImages])
			yDis = np.zeros(2*batchSize)
			yDis[:batchSize] = 0.9

			discriminator.trainable = True
			dloss = discriminator.train_on_batch(X, yDis)

			noise = np.random.normal(0,1,size=[batchSize, randomDim])
			yGen = np.ones(batchSize)
			discriminator.trainable = False
			gloss = gan.train_on_batch(noise, yGen)

		print("Discriminator loss = ",dloss)
		print("Generator loss = ",gloss)

		dLosses.append(dloss)
		gLosses.append(gloss)
		if i == 1 or i % 20 == 0 :
			plotGeneratedImages(i)
			saveModels(i)

	plotLoss(i)
def delete_prev_metadata():
	if os.path.exists('generated_images'):
		shutil.rmtree('generated_images')
	if os.path.exists('models'):
		shutil.rmtree('models')
	if os.path.exists('plots'):
		shutil.rmtree('plots')

	if not os.path.exists('generated_images'):
		os.mkdir('generated_images')
	if not os.path.exists('models'):
		os.mkdir('models')
	if not os.path.exists('plots'):
		os.mkdir('plots')

delete_prev_metadata()
train(epoch = 300)




