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
    Processes an image by converting it to grey scale, then does the edge detection by using the canny algorithm.

    :param image: an instane of PIL.Image
    :return np.array: The processed image as numpy array (black/white and scaled)
    """

    arr = np.array(image)
    arr = np.delete(arr, np.s_[::2], 0)
    arr = np.delete(arr, np.s_[::2], 1)
    arr = np.sum(arr, axis=2)
    arr = (np.maximum(np.multiply(100 < arr, arr < 227), arr == 352) * 255).astype('uint8')
    return arr


def stream_app(coordinates, frames=3, fps=30, stack=False):
    """"
    Streams the coordinates of the screen per number of frames as specified

    :param coordinates: dict the coordinates to pass to stream
    :param frames: int the number of frames to stack
    :param fps: int the framerate of the stream
    :return stacked: numpy.array the images stacked onto each other
    """
    framerate = 1.0/fps

    images = []
    sct = mss()
    if stack:
        for i in range(frames):
            sct_img = sct.grab(coordinates)
            #im = Image.frombytes('RGB', sct_img.size, sct_img.rgb)
            im = process_image(sct_img)
            images.append(im)
            time.sleep(framerate)
            # stacked = np.dstack(images)
    else:
        sct_img = sct.grab(coordinates)
        #im = Image.frombytes('RGB', sct_img.size, sct_img.rgb)
        im = process_image(sct_img)
        images.append(im)
        #print(np.array(im).shape)
    # stacked = np.dstack(images)
    return images

def reward(state):
    if np.sum((state[:,:,3] - state[:,:,2])) == 0 and np.sum((state[:,:,2] - state[:,:,1])) == 0 and np.sum((state[:,:,1] - state[:,:,0])) > 10:
        print("flappy is dood")
        return -10
    elif sum(state[0,:50,3]) == 510 and sum(state[0,:50,2]) == 510 and sum(state[0,:50,1]) != 510 and sum(state[0,:50,0]) != 510:
        print("punt gescoord!")
        return 1
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

    # Prepare frame stack and model
    stack = deque(maxlen=4)
    stream_before = stream_app(coordinates=coordinates, frames=4, fps=FPS, stack=True)
    stack.extend(stream_before)
    current_state = np.dstack(stack)
    model = create_model()

    while time.time() < t_end:
        start = time.time()

        # action

        # if random.random() > 0.9:
        #    press_space()

        stream_before = stream_app(coordinates=coordinates, frames=1, fps=FPS, stack=False )
        stack.extend(stream_before) 
        previous_state = current_state
        current_state = np.dstack(stack)
        r = reward(previous_state)

        #y0 = r + 0.9 * max(max(model.predict(x=current_state[np.newaxis, ...])))
        #model.fit(x=previous_state[np.newaxis, ...], y=np.array([[y0, y0]]), verbose=0)

        # Wait for next frame
        time_to_process = time.time()-start
        print(time_to_process)
        time.sleep(max(0.0, framerate - time_to_process))

    kill_app(app)




if __name__ == "__main__":
    main()
