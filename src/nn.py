from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D
from keras.models import load_model
import numpy as np

class Q(object):
    """Een model met de juiste architectuur dat we kunnen trainen."""

    def __init__(self, stacked_frames=4, name=None):
        """maak het model"""

        if name is None:
            model = Sequential()
            model.add(Conv2D(filters=16, kernel_size=(7, 4), strides=(7, 4), padding='same',
                             data_format='channels_last', activation='relu', input_shape=(203, 144, stacked_frames)))
            model.add(Conv2D(filters=32, kernel_size=(8, 8), strides=4, padding='same',
                             data_format='channels_last', activation='relu'))
            model.add(Conv2D(filters=32, kernel_size=(2, 2), strides=2, padding='same',
                             data_format='channels_last', activation='relu'))
            model.add(Flatten())
            model.add(Dense(units=200, activation='relu'))
            model.add(Dense(units=2, activation='linear'))

            model.compile(optimizer='rmsprop', loss='mse')

            self.model = model
        else:
            self.model = load_model(name)

        self.GAMMA = 0.9

    def predict_action(self, state):
        """Return the balance remaining after withdrawing *amount*
        dollars."""
        prediction = self.model.predict(x=state[np.newaxis, ...])[0]
        if prediction[1] > prediction[0]:
            return 1
        else:
            return 0

    def omschrijving(self):
        """Print the summary of the model"""
        print(self.model.summary())

    def train(self, previous_state, action, reward, current_state):
        y = self.model.predict(x=previous_state[np.newaxis, ...])[0]
        y[action] = reward + self.GAMMA * np.max(self.model.predict(x=current_state[np.newaxis, ...]))
        self.model.fit(x=previous_state[np.newaxis, ...], y=np.array([y]), verbose=0)

    def save(self, name = 'flappy_model.h5'):
        self.model.save(name)