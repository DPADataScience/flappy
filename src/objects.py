import gym
from mss import mss
import numpy as np
from PIL import Image
from pynput.keyboard import Key, Controller
import time
from src import nn


class Agent:
    def __init__(self, environment,  disount_rate=0.9, epsilon=1.0, max_epsilon=1.0, min_epsilon=0.01, decay_rate=0.01):
        """"
        Creates an instance of Agent.

        :param environment: Environment the agent is created for
        :param disount_rate: float The gamma value of Reinnfocement learning - i.e. the cumulative discoutned reward value
        :param epsilon: float A value between 0 and 1 that is used in the training stages of the agent. The higher the value
        the likelier the agent will take random actions
        :param max_epsilon: float The max value of epsilon
        :param min_epsilon: float the min value of epsilon
        :param decay_rate: float The decay rate to be used by the agent. The higher the rate the quicker the agent will
        reduce the value of epsilon
        """
        self.env = environment
        self.gamma = disount_rate
        self.epsilon = epsilon
        self.max_epsilon = max_epsilon
        self.min_epsilon = min_epsilon
        self.decay_rate = decay_rate
        self.model = nn.create_model() #TODO create network:  _create_network()

    def _create_network(self):
        raise NotImplementedError("yet to be implemented")

    def train(self, total_episodes):
        """"
        Trains the agent to play the game assigned to this agent using Reinforcement Learning
        """
        # raise NotImplementedError
        rewards = []
        for episode in range(total_episodes):
            state = self.env.reset()
            total_rewards = 0
            while True:
                # here we implement the exploration possibility in the beginning we want to take random actions
                # so the agent can explore the environment, later once it has learned about the environment it can
                # start to take actions it knows
                predicted = self.model.predict(state)
                action = np.argmax(predicted[0])
                if np.random.uniform(low=0, high=1) < self.epsilon:
                    action = np.random.choice(self.env.action_space)

                # take the action and observe the outcome state and reward
                next_state, reward, done, info = self.env.step(action)

                # retrain the neural net
                predicted_next_state = self.model.predict(next_state)
                Qmax = np.max(predicted_next_state)
                y = predicted

                y[0][action] = reward + self.gamma * Qmax
                self.model.fit(x=state, y=y, verbose=0)

                state = next_state
                total_rewards += reward

                if done:
                    break

                # adjust epsilon by an exponential function with lower bound min_epsilon
                self.epsilon = self.min_epsilon + (self.max_epsilon - self.min_epsilon) * np.exp(-self.decay_rate * episode)
                rewards.append(total_rewards)

        print("Score over time: " + str(sum(rewards) / total_episodes))


    def play(self, episodes=1, repeat_play=False):
        """"
        The agent plays the game.

        :param episodes: int number of rounds the agent should play the game
        :param repeat_play: boolean whether the agent should repeat playing after dying or not
        """
        for episode in range(episodes):
            state = self.env.reset()
            print("****************************************************")
            print("EPISODE ", episode)
            while True:
                action = np.argmax(self.model.predict(state))
                next_state, reward, done, info = self.env.step(action)

                if done:
                    if repeat_play:
                        self.env.reset()
                    else:
                        print("Flappy died")
                state = next_state


class Environment:
    def __init__(self, app, FPS=30, stacked_frames=4):
        """"
        Creates an instance of Environment
        """
        self.app = app
        self.fps = FPS
        self.stacked_frames = stacked_frames
        self.framerate = 1/FPS
        self.coordinates = self._get_window_coordinates(app)
        self.action_space = (0, 1) #the possibe actions in this environment


    def _get_window_coordinates(self, app):
        """"
        Returns the coordinates of an app as x_start, y_start, x_end, y_end

        :param app: an instance of pywinauto.Application
        :return coordinates: dict The coordinates in the for of x_start, y_start, x_end, y_end
        """
        window = app.window_()
        hwnd = window.wrapper_object()
        rect = hwnd.client_area_rect() #Dit stukje code heeft me echt veels te veel moeite gekost om het te achterhalen
        # return rect.left, rect.top, rect.right, rect.bottom

        coordinates = {'top': rect.top,
                       'left': rect.left,
                       'height': rect.bottom - rect.top - 106,
                       'width': rect.right - rect.left
                       }

        return coordinates

    def _grab_frame(self, coordinates):
        """"
        grabs screen within coordinates, and apply process_image

        :param coordinates: dict the coordinates of the screen
        :return image: numpy.array representation of processed images
        """
        sct = mss()
        screen_image = sct.grab(coordinates)
        processed_image = self._process_image(screen_image)
        return processed_image

    def _process_image(self, image):
        """"
        Processes an image by converting it to a numpy array, scaling without losing detail and selecting only edges that matter.

        :param image: an instance of mss.screenshot.Screenshot
        :return np.array: The processed image as numpy array (black/white and scaled)
        """

        arr = np.array(image)
        arr = np.delete(arr, np.s_[::2], 0)
        arr = np.delete(arr, np.s_[::2], 1)
        arr = np.sum(arr, axis=2)
        arr = (np.multiply(200 < arr, arr < 227) * 255).astype('uint8')
        return arr

    def _convert_image_to_array(self, image):
        """"
        Converts an image to a numpy array

        :param image: an instane of PIL.Image
        :return np.array: The image as numpy array
        """
        return np.array(image)

    def convert_array_to_image(self, arr):
        return Image.fromarray(arr)

    def _calculate_reward(self, state):
        """"
        calculate the reward of the state. Flappy is dead when the screen has stopped moving, so when two consecutive frames
        are equal. A point is scored when an obstacle is above flappy, and before it wasn't. An object is above Flappy when
        there are two white pixels in the first 50 pixels on the first row.

        :param state: four consecutive processed frames
        :return reward: int representing the reward if a point is scored or if flappy has died.
        """
        if np.sum(state[30:32, :, 3]) == 8415:
            print("in menu")
            return -100
        elif np.sum(state[:, 0:50, 3]) / 255 == 50:
            print("buiten scherm")
            return -100
        elif np.sum((state[:,:,3] - state[:,:,2])) == 0 and np.sum((state[:,:,2] - state[:,:,1])) == 0:
            print("flappy is dood")
            return -1000
        elif sum(state[0,:50,3]) == 510 and sum(state[0,:50,2]) == 510 and sum(state[0,:50,1]) != 510 and sum(state[0,:50,0]) != 510:
            print("punt gescoord!")
            return 1000
        else:
            return 0

    def _press_space(self):
        """"
        Presses space if a pywinauto application is active
        """
        keyboard = Controller()
        keyboard.press(Key.space)
        keyboard.release(Key.space)

    def reset(self):
        """"
        Resets the state of the environment and returns an initial observation.

        :return observation: np.array Returns an observation of the environmet after starting the game. The dimensions
        of the array are #TODO(, self.stacked_Frames)
        """
        self._press_space()
        observation = self._stream()
        return observation

    def _stream(self):
        # stack = deque(maxlen=self.stacked_frames)
        stack = []
        for i in range(self.stacked_frames):
            frame = self._grab_frame(self.coordinates)
            stack.extend([frame])
            time.sleep(self.framerate)

        state = np.dstack(stack)
        return state[np.newaxis, ...]

    def step(self, action):
        """"
        Run one timestep of the environment's dynamics. When end of
        episode is reached, you are responsible for calling `reset()`
        to reset this environment's state.
        Accepts an action and returns a tuple (observation, reward, done, info).

        :param action: int 0 for do nothing 1 for press space
        :return observation: np.array
        :return reward: float  amount of reward returned after previous action
        :return done: boolean whether the episode has ended, in which case further step() calls will return undefined results
        :return info: ontains auxiliary diagnostic information (helpful for debugging, and sometimes learning)

        """
        if action == 1:
            self._press_space()

        observation = self._stream()
        reward = self._calculate_reward(observation)
        done = False
        info = {}
        if reward == -1000:
            done = True

        return observation, reward, done, info

    def close(self):
        self._kill_app()

    def _kill_app(self):
        """"
        Kills a pywintauto.Application

        :return: void
        """
        self.app.kill()