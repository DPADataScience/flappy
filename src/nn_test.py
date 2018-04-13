from src import NeuralNetwork
import random

def create_population(self, count):
    population = []
    for _ in range(0, count):
        # Create randon network
        network = NeuralNetwork(self.nn_params_choices)
        network.create_random()

        population.append(network)

    return population


def breed(self, mother, father):
    """Make two children as parts of their parents.
    Args:
        mother (dict): Network parameters
        father (dict): Network parameters
    """
    children = []
    for _ in range(2):
        child = {}

        #Loop through the parameters and pick params for the kid
        for param in self.nn_params_choices:
            child[param] = random.choice([mother.network[param], father.network[param]])


        #create a new network object
        network = NeuralNetwork(self.nn_params_choices)
        network.create_set(child)
        children.append(child)

    return children


def mutate(self, network):
