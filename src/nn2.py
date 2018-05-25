from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D
from keras.models import load_model

import numpy as np
import pandas as pd
import time

pd.options.mode.chained_assignment = None

class Q(object):
    """Een model met de juiste architectuur dat we kunnen trainen."""

    def __init__(self, reset_Q=False):
        """maak de twee modellen, en definieer Gamma"""

        self.model = [0, 1]

        if reset_Q:
            model_0 = Sequential()
            model_0.add(Dense(units=10, activation='relu', input_shape=[2]))
            model_0.add(Dense(units=10, activation='relu'))
            #model_0.add(Dense(units=10, activation='relu'))
            model_0.add(Dense(units=1, activation='linear'))
            #maybe add decay in the optimizer
            model_0.compile(optimizer='rmsprop', loss='mse')

            self.model[0] = model_0

            model_1 = Sequential()
            model_1.add(Dense(units=10, activation='relu', input_shape=[2]))
            model_1.add(Dense(units=10, activation='relu'))
            #model_1.add(Dense(units=10, activation='relu'))
            model_1.add(Dense(units=1, activation='linear'))
            #maybe add decay in the optimizer
            model_1.compile(optimizer='rmsprop', loss='mse')

            self.model[1] = model_1

            print('New models initialized.')
        else:
            self.model[0] = load_model('flappy_model_0.h5')
            self.model[1] = load_model('flappy_model_1.h5')
            print('Loaded models from drive.')

        self.GAMMA = 0.96
        # self.omschrijving()

    def predict(self, state, action):
        return self.model[action].predict(x=state[np.newaxis, ...])[0][0]

    def fit(self, state, y, action):
        self.model[action].fit(x=state[np.newaxis, ...], y=np.array([y]), verbose=0)

    def predict_action(self, state):
        """[np.newaxis, ...] staat erbij omdat .predict een verzameling aan inputs verwacht. """

        #print('1: ', self.predict(state, 1), '    0: ', self.predict(state, 0))

        if self.predict(state, 1) > self.predict(state, 0):
            return 1
        else:
            return 0

    def omschrijving(self):
        """Print the summary of the model"""
        print(self.model[0].summary())

    def train(self, previous_state, action, reward, current_state):
        #print('y was: ', y, ' action: ', action, ' reward: ', reward)
        y = reward + self.GAMMA * max(self.predict(current_state, 0), self.predict(current_state, 1))
        #print('y is: ', y)
        self.fit(previous_state, y, action)
        #y[action] = self.predict(previous_state, action)
        #print('y wordt: ', y, ' action: ', action, ' reward: ', reward)

    def train_with_memory(self, memories, batch_size=10000, mini_batch_size=500, iterations=10):
        start = time.time()
        print('memories.shape: ', memories.shape)
        print('procent 100 in memories: ', memories[memories.reward == 100].shape[0] / memories.shape[0])

        for i in range(iterations):

            batch = memories.sample(min(batch_size, memories.shape[0]), replace=True, weights=memories['reward'].apply(lambda r: min(abs(r), 12)))
            if i == 0:
                print('batch.shape: ', batch.shape)
                print('procent 100 in batch: ', batch[batch.reward == 100].shape[0] / batch.shape[0])

            for action in [0, 1]:
                sub_batch = batch.loc[batch['action'] == action]

                for k, g in sub_batch.groupby(np.arange(len(sub_batch)) // mini_batch_size):
                    x = pd.DataFrame(g.current_state.values.tolist()).values
                    g['y'] = g.as_matrix(columns=['reward']) + self.GAMMA * np.maximum(self.model[0].predict(x), self.model[1].predict(x))

                    x = pd.DataFrame(g.previous_state.values.tolist()).values
                    y = g.as_matrix(columns=['y'])

                    if i == 0 and k == 0:
                        score = self.model[action].evaluate(x, y, verbose=0)
                        print("score: ", score)
                    else:
                        self.model[action].fit(x, y, verbose=0)

        print("Trainen duurde ", time.time() - start, ' sec.')

    def save(self):
        self.model[0].save('flappy_model_0.h5')
        self.model[1].save('flappy_model_1.h5')
