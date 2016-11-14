from keras.models import Sequential
from keras.layers import Dense, Convolution2D, MaxPooling2D
from keras.layers import Dropout, SpatialDropout2D, Activation, Flatten
from keras.optimizers import SGD
from keras.utils.np_utils import to_categorical

class CNN:
    def __init__(self, n_classes):
#        self.model.add(Dropout(0.25))
#        self.model.add(SpatialDropout2D(0.25))

        self.model = Sequential()
        self.model.add(Convolution2D(20, 7, 7, subsample=(2, 2), input_shape=(1, 100, 100)))
        self.model.add(Activation('relu'))

        self.model.add(Convolution2D(35, 5, 5, subsample=(2, 2)))
        self.model.add(Activation('relu'))
        self.model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
        
        self.model.add(Convolution2D(40, 3, 3))
        self.model.add(Activation('relu'))

        self.model.add(Convolution2D(40, 3, 3))
        self.model.add(Activation('relu'))
        
        self.model.add(Convolution2D(35, 3, 3))
        self.model.add(Activation('relu'))
#        self.model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))

        self.model.add(Flatten())
        self.model.add(Dense(256))
        self.model.add(Activation('relu'))
        
        self.model.add(Dense(256))
        self.model.add(Activation('relu'))

        self.model.add(Dense(n_classes))
        self.model.add(Activation('softmax'))

        self.model.summary()

        sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
        self.model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

    def train(self, X_train, Y_train, validationRatio=0.0):
        self.model.fit(X_train, Y_train, batch_size=8, nb_epoch=1, validation_split=validationRatio)

    def test(self, X_test, Y_test):
        score = self.model.evaluate(X_test, Y_test)
        print "Test accuracy:", 100 * score[1], "%"
