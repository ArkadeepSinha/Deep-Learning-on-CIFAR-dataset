import tensorflow as tf
from keras.datasets import cifar10
import numpy as np
import numpy.random as nr
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten
from sklearn.metrics import accuracy_score
from keras.optimizers import SGD
nc = 10 # Number of classes


(Xtrain, ytrain), (Xtest, ytest) = cifar10.load_data()
#Show sample images
plt.figure(1)
imgplot1 = plt.imshow(Xtrain[nr.randint(50000)])
plt.show()

plt.figure(2)
imgplot2 = plt.imshow(Xtest[nr.randint(10000)])
plt.show()

Xtrain = Xtrain.astype('float32')
Xtrain = Xtrain[0:20000,:] / 255.0
Xtest = Xtest.astype('float32')
Xtest = Xtest / 255.0
ytrainEnc = tf.one_hot(ytrain[0:20000:,0], depth=nc) 
ytestEnc = tf.one_hot(ytest[:,0], depth=nc)

model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same', input_shape=(32, 32, 3)))
model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
model.add(MaxPooling2D(2, 2))
model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
model.add(MaxPooling2D(2, 2))
model.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
model.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
model.add(MaxPooling2D(2, 2))
model.add(Flatten())
model.add(Dense(128, activation='relu', kernel_initializer='he_uniform'))
model.add(Dense(10, activation='softmax'))
opt = SGD(learning_rate=0.001, momentum=0.9)          
model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
history = model.fit(Xtrain, ytrainEnc, epochs=1, batch_size=200, validation_data=(Xtest, ytestEnc))

ypred = model.predict(Xtest)
ypred = np.argmax(ypred,axis=1)
score = accuracy_score(ypred,ytest)
print('Accuracy score is',100*score,'%')