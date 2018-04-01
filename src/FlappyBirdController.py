from PIL import ImageGrab, Image
import time
import numpy as np
import cv2
import pandas as pd
import os
import subprocess
import pyautogui
import win32gui
from pywinauto.application import  Application
from pywinauto.keyboard import SendKeys
from pywinauto.controls.hwndwrapper import HwndWrapper
from mss import mss
from io import BytesIO, StringIO


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
    :return np.array: The processed image as numpy array (greyscale and canny)
    """
    arr = convert_image_to_array(image)
    # convert to gray
    processed_img = cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY)

    # edge detection
    processed_img =  cv2.Canny(processed_img, threshold1=160, threshold2=300)
    return processed_img


def is_game_over(frame):
    pass


def stream_app(coordinates, nr_frames=1, fps=30):
    """"
    Streams the coordinates of the screen per number of frames as specified

    :param coordinates: dict the coordinates to pass to stream
    :param nr_frames: int the number of frames to stack
    :param fps: int the framerate of the stream
    :return stacked: numpy.array the images stacked onto each other
    """
    framerate = 1.0/fps

    images = []
    sct = mss()
    for i in range(nr_frames):
        sct_img = sct.grab(coordinates)
        im = Image.frombytes('RGB', sct_img.size, sct_img.rgb)
        # im = process_image(im)
        images.append(im)
        time.sleep(framerate)

    stacked = np.dstack(images)
    return stacked


def main():
    FPS = 30
    launch_flappy()
    app = get_application()

    # Play game and grab screen
    x_start, y_start, x_end, y_end = get_window_coordinates(app)
    coordinates = {'top':y_start,
                   'left': x_start,
                   'height': y_end-y_start,
                   'width': x_end-x_start
                   }

    t_end = time.time() + 15

    stored_bytes = []
    while time.time() < t_end:
        press_space()
        stream = stream_app(coordinates=coordinates, fps=10)
        filebuffer = BytesIO()
        np.save(filebuffer, stream)
        filebuffer.seek(0)
        stored_bytes.append(filebuffer)
        time.sleep(0.2)

    for i in range(len(stored_bytes)):
        print(stored_bytes[i])
        arr = np.load(stored_bytes[i])
        im = Image.fromarray(arr)
        filenam_jpg = "../screendumps/_" + str(i) + ".jpg"
        filenam_by = "../screendumps/_" + str(i) + ".npy"
        im.save(filenam_jpg)

        file = open(filenam_by, "wb")
        np.save(file,arr)
        file.close()


    kill_app(app)





if __name__ == "__main__":
    main()
