from PIL import ImageGrab, Image
import time
import numpy as np
#import cv2
import pandas as pd
import os
import subprocess
import random
import pyautogui
import win32gui
from pywinauto.application import  Application
from pywinauto.keyboard import SendKeys
from pywinauto.controls.hwndwrapper import HwndWrapper
from mss import mss
from collections import deque
from src.nn import create_model
from keras.models import load_model
import random

def launch_flappy(folder='../FlappyBirdClone/', filename = 'flappy.py', timeout=2):
    """"
    Launches the flappy bird game as a python subprocess

    :param folder: string the folder to launch from
    :param filename: string the filename to launch
    :param timeout: int number of seconds to sleep after launching process
    :return: void
    """

    py = 'python '
    command = py + folder + filename
    p = subprocess.Popen(command, cwd=folder)
    time.sleep(timeout)
    return p


def press_space():
    """"
    Presses space if a pywinauto application is active

    :return: void
    """
    SendKeys('{VK_SPACE}')



def get_application(title='Flappy Bird'):
    """"
    Connects to a window and returns this as an pywintauto.Application

    :param title: sting the name of the window
    :return pyWinAuto.application: returns the app
    """
    app = Application()
    app.connect(title=title)
    return app


def kill_app(app):
    """"
    Kills a pywintauto.Application

    :return: void
    """
    app.kill()


def get_window_coordinates(app):
    """"
    Returns the coordinates of an app as x_start, y_start, x_end, y_end

    :param app: an instance of pywinauto.Application
    :return rect.left, rect.top, rect.right, rect.bottom: The coordinates in the for of x_start, y_start, x_end, y_end
    """
    window = app.window_()
    hwnd = window.wrapper_object()
    rect = hwnd.client_area_rect() #Dit stukje code heeft me echt veels te veel moeite gekost om het te achterhalen
    return rect.left, rect.top, rect.right, rect.bottom


def convert_image_to_array(image):
    """"
    Converts an image to a numpy array

    :param image: an instane of PIL.Image
    :return np.array: The image as numpy array
    """
    return np.array(image)


def convert_array_to_image(arr):
    return Image.fromarray(arr)


def process_image(image):
    """"
    Processes an image by converting it to a numpy array, scaling without losing detail and selecting only edges that matter.

    :param image: an instance of mss.screenshot.Screenshot
    :return np.array: The processed image as numpy array (black/white and scaled)
    """

    arr = np.array(image)
    arr = np.delete(arr, np.s_[::2], 0)
    arr = np.delete(arr, np.s_[::2], 1)
    arr = np.sum(arr, axis=2)
    #arr = (np.maximum(np.multiply(100 < arr, arr < 227), arr == 352) * 255).astype('uint8')
    arr = (np.multiply(100 < arr, arr < 227) * 255).astype('uint8')
    return arr


def grab_frame(coordinates):
    """"
    grabs screen within coordinates, and apply process_image

    :param coordinates: dict the coordinates of the screen
    :return image: numpy.array representation of processed images
    """
    sct = mss()
    screen_image = sct.grab(coordinates)
    processed_image = process_image(screen_image)
    return processed_image


def calculate_reward(state):
    """"
    calculate the reward of the state. Flappy is dead when the screen has stopped moving, so when two consecutive frames
    are equal. A point is scored when an obstacle is above flappy, and before it wasn't. An object is above Flappy when
    there are two white pixels in the first 50 pixels on the first row.

    :param state: four consecutive processed frames
    :return reward: int representing the reward if a point is scored or if flappy has died.
    """

    if np.sum((state[:,:,3] - state[:,:,2])) == 0 and np.sum((state[:,:,2] - state[:,:,1])) == 0:# and np.sum((state[:,:,1] - state[:,:,0])) > 10:
        print("flappy is dood")
        return -100
    elif sum(state[0,:50,3]) == 510 and sum(state[0,:50,2]) == 510 and sum(state[0,:50,1]) != 510 and sum(state[0,:50,0]) != 510:
        print("punt gescoord!")
        return 100
    else:
        return 0

def main():

    # Play game and grab screen
    launch_flappy()
    app = get_application()
    x_start, y_start, x_end, y_end = get_window_coordinates(app)
    coordinates = {'top': y_start,
                   'left': x_start,
                   'height': y_end - y_start - 100,
                   'width': x_end - x_start
                   }

    # Set time parameters
    FPS = 30
    NR_STACKED_FRAMES = 4
    GAMMA = 0.9
    BATCH_SIZE = 5
    t_end = time.time() + 60
    framerate = 1/FPS

    # Prepare states, action  and model
    stack = deque(maxlen=NR_STACKED_FRAMES)
    for i in range(NR_STACKED_FRAMES):
        frame = grab_frame(coordinates)
        stack.extend([frame])
        time.sleep(framerate)
    current_state = np.dstack(stack)

    #model = create_model()
    model = load_model('flappy_model.h5')

    replay_memory =  deque(maxlen=500)

    action = 0
    #start episode
    while time.time() < t_end:
        start = time.time()

        '''
        Given a transition < s, a, r, s’ >, the Q-table update rule in the previous algorithm must be replaced with the following:

        1. Do a feedforward pass for the current state s to get predicted Q-values for all actions.
        2. Do a feedforward pass for the next state s’ and calculate maximum overall network outputs max a’ Q(s’, a’).
        3. Set Q-value target for action to r + γmax a’ Q(s’, a’) (use the max calculated in step 2). 
            For all other actions, set the Q-value target to the same as originally returned from step 1, making the error 0 for those outputs.
        4. Update the weights using backpropagation.

        '''

        # action
        # select an action a
        # with probability p select a random action
        # otherwise select a = argmax Q(s, a')
        if np.random.uniform() >= 0.5:
            if np.random.uniform() >= 0.5:
                action = 1
                press_space()
            else:
                action = 0
        else:
            prediction = model.predict(x=current_state[np.newaxis, ...])[0]  # step 1
            if prediction[1] > prediction[0]:
                action = 1
                press_space()
            else:
                action = 0

        # make tuple (previous_states, action, reward, current_state)
        frame = grab_frame(coordinates)
        stack.extend([frame])
        previous_state = current_state
        current_state = np.dstack(stack)
        reward = calculate_reward(previous_state)

        #store experience in <s, a, r, s'> in replay memory D
        the_tuple = dict()
        the_tuple['previous_state'] = previous_state
        the_tuple['action'] = action
        the_tuple['reward'] = reward
        the_tuple['current_state'] = current_state

        replay_memory.append(the_tuple)


    #get minibatch and train network
    minibatch = random.sample(replay_memory, k=BATCH_SIZE)

    y = []
    for mb in minibatch:
        current_state = mb['current_state']
        reward = calculate_reward(current_state)
        if reward == -100:
            target = mb['reward']
        else:
            target = mb['reward'] + GAMMA * np.max(model.predict(x=current_state[np.newaxis, ...])) #model returns a 2D array
        print('target', target)
        y.append(target)
    #
    #calculate target for each minibatch transition
    # if ss'is terminal state then tt == rr
    #otherwise tt = rr + ymax(Q)

    #train network using minibatch
    # updating the model
    print('y', y)
    # y = prediction
    # y[action] = reward + GAMMA * np.max(model.predict(x=current_state[np.newaxis, ...])) #step 2 and 3
    model.fit(x=previous_state[np.newaxis, ...], y=np.array([y]), verbose=0) # Step 4

        # Wait for next frame
    time_to_process = time.time()-start
    print('time to process', time_to_process)
    time.sleep(max(0.0, framerate - time_to_process)) # it not needed to update the nn for every frame, we can skip frames

    kill_app(app)
    model.save('flappy_model.h5')

if __name__ == "__main__":
    main()
