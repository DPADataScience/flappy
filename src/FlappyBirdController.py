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


def main():
    FPS = 30
    launch_flappy()
    app = get_application()

    # Play game and grab screen
    x_start, y_start, x_end, y_end = get_window_coordinates(app)
    images = []
    for i in range(20):
        press_space()
        im = ImageGrab.grab(bbox=(x_start, y_start, x_end, y_end))
        arr = process_image(im)
        images.append(arr)
        time.sleep(1.0/FPS)

    kill_app(app)

    folder = '../screendumps/'
    for image in images:
        dumpto = folder + 'screen_capture_' + str(i) + '.jpg'
        image.save(dumpto, "JPEG")
        i += 1

    # print(type(images))
    # for image in images:
    #     Image.fromarray(image)


if __name__ == "__main__":
    main()
