from keras.callbacks import TensorBoard
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.initializers import RandomUniform
import numpy as np
import time


def get_model():
    model = Sequential([
        Dense(752, input_shape=(25,), kernel_initializer=RandomUniform(minval=-1, maxval=1),
              bias_initializer=RandomUniform(minval=-1, maxval=1)),
        Activation('relu'),
        Dense(500, kernel_initializer=RandomUniform(minval=-1, maxval=1),
              bias_initializer=RandomUniform(minval=-1, maxval=1)),
        Activation('relu'),
        Dense(250, kernel_initializer=RandomUniform(minval=-1, maxval=1),
              bias_initializer=RandomUniform(minval=-1, maxval=1)),
        Activation('relu'),
        Dense(100, kernel_initializer=RandomUniform(minval=-1, maxval=1),
              bias_initializer=RandomUniform(minval=-1, maxval=1)),
        Activation('relu'),
        Dense(50, kernel_initializer=RandomUniform(minval=-1, maxval=1),
              bias_initializer=RandomUniform(minval=-1, maxval=1)),
        Activation('relu'),
        Dense(10, kernel_initializer=RandomUniform(minval=-1, maxval=1),
              bias_initializer=RandomUniform(minval=-1, maxval=1)),
        Activation('relu')
    ])
    model.compile(optimizer='adam', loss='mse')
    return model


def network_output():
    x_test = np.random.rand(10000, 25)
    model = get_model()
    pred = model.predict(x_test)
    return x_test, pred


if __name__ == "__main__":
    x_train, y_train = network_output()
    model2 = get_model()
    tensorboard_callback = TensorBoard(histogram_freq=1, log_dir=f'./logs/log_{time.time()}')
    model2.fit(x_train, y_train, validation_split=0.1, epochs=10000, callbacks=[tensorboard_callback])
