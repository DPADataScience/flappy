import numpy as np
import pandas as pd
import time


pd.options.mode.chained_assignment = None

class Q(object):
    """Een model met de juiste architectuur dat we kunnen trainen."""

    def __init__(self, reset_Q=False):
        """maak de twee modellen, en definieer Gamma"""

        if reset_Q:
            self.table = np.random.rand(100, 100, 2, 2)

            print('New table initialized.')
        else:
            try:
                self.table = np.load('table.npy')
                print('table loaded from disk (tably.npy).')
            except:
                self.table = np.random.rand(100, 100, 2, 2)
                print('Kon table niet laden van schijf. Nieuwe table aangemaakt.')
        self.table[:,:,0,0] = 500
        self.table[:,:,0,1] = -499
        self.GAMMA = 0.99
        self.ALPHA = 0.7

    def predict(self, state, action):
        return self.table[state[0], state[1] + 50, state[2], action]

    def fit(self, previous_state, action, reward, current_state):
        y_old = self.table[previous_state[0], previous_state[1] + 50, previous_state[2], action]
        y_new = y_old + self.ALPHA * (reward + self.GAMMA * max(self.table[current_state[0], current_state[1] + 50, current_state[2],]) - y_old)
        self.table[previous_state[0], previous_state[1] + 50, previous_state[2], action] = y_new

        #if previous_state[0] == 4 and previous_state[1] == 13:
        #    print(previous_state, action, reward, current_state, y_old, y_new)
        return abs(y_old - y_new)
        #
        #print(previous_state, action, reward, current_state, 'y was: ', y_old, ' en is: ', y_new, ' verschil: ', y_old - y_new)

    def predict_action(self, state):
        """[np.newaxis, ...] staat erbij omdat .predict een verzameling aan inputs verwacht. """

        #print('1: ', self.predict(state, 1), '    0: ', self.predict(state, 0))

        if self.predict(state, 1) > self.predict(state, 0):
            return 1
        else:
            return 0

    def train_with_memory(self, memories, batch_size=10000, iterations=100):
        start = time.time()
        print('memories.shape: ', memories.shape)
        print('procent 100 in memories: ', memories[memories.reward == 100].shape[0] / memories.shape[0])

        for i in range(iterations):
            batch = memories.sample(min(batch_size, memories.shape[0]), replace=True, weights=memories['weight'].apply(lambda x: np.log(x)))
            if i == 0:
                print('batch.shape: ', batch.shape)
                print('procent 100 in batch: ', batch[batch.reward == 100].shape[0] / batch.shape[0])

            difference_sum = 0
            for index, row in batch.iterrows():
                difference_sum += self.fit(row['previous_state'], row['action'], row['reward'], row['current_state'])
            #print('gemiddelde verschil is: ', difference_sum / batch.shape[0], 'bij alpha is: ', self.ALPHA)

        print("Trainen duurde ", time.time() - start, ' sec.')

    def save(self):
        np.save('table.npy', self.table)