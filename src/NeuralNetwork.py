from keras import Sequential
from keras.layers import Dense, Conv2D
from numpy import random

class NeuralNetwork():
    def __init__(self, params = None):
        self.accuracy = 0
        self.params = params
        self.network = {} #Returns a dict with params to create random network

    def create_random(self):
        for key in self.params:
            self.network[key] = random.choice(self.params[key])

    