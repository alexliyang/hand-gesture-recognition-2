from keras.models import Sequential
from keras.layers import Dense, Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.layers import Dropout, SpatialDropout2D, Activation, Flatten
from keras.optimizers import SGD
from keras.utils.np_utils import to_categorical

class CNN:
    def __init__(self, n_classes):
        self.model = Sequential()
        self.model.add(Convolution2D(32, 3, 3, input_shape=(1, 100, 100)))
        self.model.add(Activation('relu'))

        self.model.add(ZeroPadding2D())
        self.model.add(Convolution2D(32, 3, 3))
        self.model.add(Activation('relu'))
        self.model.add(MaxPooling2D(pool_size=(2,2)))
                
        self.model.add(Convolution2D(64, 3, 3))
        self.model.add(Activation('relu'))

        self.model.add(ZeroPadding2D())
        self.model.add(Convolution2D(64, 3, 3))
        self.model.add(Activation('relu'))
        self.model.add(MaxPooling2D(pool_size=(2,2)))

        self.model.add(Convolution2D(128, 3, 3))
        self.model.add(Activation('relu'))

        self.model.add(ZeroPadding2D())
        self.model.add(Convolution2D(128, 3, 3))
        self.model.add(Activation('relu'))
        self.model.add(MaxPooling2D(pool_size=(2,2)))

        self.model.add(Convolution2D(256, 3, 3))
        self.model.add(Activation('relu'))

        self.model.add(ZeroPadding2D())
        self.model.add(Convolution2D(256, 3, 3))
        self.model.add(Activation('relu'))
        self.model.add(MaxPooling2D(pool_size=(2,2)))

        self.model.add(Flatten())
        self.model.add(Dense(1024))
        self.model.add(Activation('relu'))
        self.model.add(Dropout(0.5))
        
        self.model.add(Dense(1024))
        self.model.add(Activation('relu'))
        self.model.add(Dropout(0.5))

        self.model.add(Dense(n_classes))
        self.model.add(Activation('softmax'))

        sgd = SGD(lr=1e-2, decay=5e-4, momentum=0.9, nesterov=True)
        self.model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
        self.model.summary()

    def train(self, X_train, Y_train, X_test, Y_test, batchSize=1, epoch=10):
        self.model.fit(X_train, Y_train, batch_size=batchSize, nb_epoch=epoch, validation_data=(X_test, Y_test))

    def train_gen(self, imgGen, m, Xv, yv, epoch=5):
        self.model.fit_generator(imgGen, samples_per_epoch=m, nb_epoch=epoch, validation_data=(Xv,yv))

    def test(self, X_test, Y_test):
        score = self.model.evaluate(X_test, Y_test)
        return 100 * score[1]

    def test_gen(self, data, n):
        score = self.model.evaluate_generator(data, n)
        return 100 * score[1]
