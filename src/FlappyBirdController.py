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

    ;:return: void
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
    arr = (np.maximum(np.multiply(100 < arr, arr < 227), arr == 352) * 255).astype('uint8')
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
    calculate the reward of the state

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
    t_end = time.time() + 10
    framerate = 1/FPS

    # Prepare states, action  and model
    stack = deque(maxlen=4)
    for i in range(4):
        frame = grab_frame(coordinates)
        stack.extend([frame])
        time.sleep(framerate)
    current_state = np.dstack(stack)
    model = create_model()
    action = 0

    while True: # time.time() < t_end:
        start = time.time()

        # action
        prediction = model.predict(x=current_state[np.newaxis, ...])[0]
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

        # updating the model
        y = prediction
        y[action] = reward + 0.9 * np.max(model.predict(x=current_state[np.newaxis, ...]))
        model.fit(x=previous_state[np.newaxis, ...], y=np.array([y]), verbose=0)

        # Wait for next frame
        time_to_process = time.time()-start
        #print(time_to_process)
        time.sleep(max(0.0, framerate - time_to_process))

    kill_app(app)


if __name__ == "__main__":
    main()
