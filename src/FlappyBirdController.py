from PIL import Image
import time
import numpy as np
import cv2
import pandas as pd
import os
import subprocess
import random
from pywinauto.application import  Application
from mss import mss
import src.q  as nn
from pynput.keyboard import Key, Controller

pd.options.mode.chained_assignment = None

def launch_flappy(folder='../FlappyBirdClone/', filename = 'flappy.py', timeout=2):
    """"
    Launches the flappy bird game as a python subprocess

    :param folder: string the folder to launch from
    :param filename: string the filename to launch
    :param timeout: int number of seconds to sleep after launching process
    :return: void
    """

    command = 'python ' + folder + filename
    p = subprocess.Popen(command, cwd=folder)
    time.sleep(timeout)
    return p


def press_space():
    """"
    Presses space if a pywinauto application is active

    :return: void
    """
    keyboard = Controller()
    keyboard.press(Key.space)
    keyboard.release(Key.space)


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
    coordinates = {'top': rect.top,
                   'left': rect.left,
                   'height': rect.bottom - rect.top - 106,
                   'width': rect.right - rect.left
                   }
    return coordinates


def process_screenshot(screenshot):
    """"
    Processes an image by converting it to a numpy array, scaling without losing detail and selecting only edges that matter.

    :param image: an instance of mss.screenshot.Screenshot
    :return np.array: The processed image as numpy array (black/white and scaled)
    """

    arr = np.array(screenshot)
    arr = np.delete(arr, np.s_[::2], 0)
    arr = np.delete(arr, np.s_[::2], 1)
    arr = np.sum(arr, axis=2)
    arr = (np.multiply(arr > 208, arr < 245) * 255).astype('uint8')

    # Show features in processed image
    features = calculate_features(arr)
    arr[:, int(features[2])] = 100
    arr[int(features[1]), :] = 100
    arr[int(features[0]), :] = 100

    return arr


def grab_frame(coordinates):
    """"
    grabs screen within coordinates, and apply process_image

    :param coordinates: dict the coordinates of the screen
    :return image: numpy.array representation of processed images
    """
    sct = mss()
    screenshot = sct.grab(coordinates)
    frame = process_screenshot(screenshot)
    return frame

def calculate_features(frame):
    first_white_pixel_per_row = np.apply_along_axis(lambda row: np.argmax(np.logical_and(row > 100, np.roll(row, -23) > 100)), 1, frame[:, 20:])
    dist_object = int(np.median(first_white_pixel_per_row) + 20)

    height_object = np.argmin(np.flip(frame[:, dist_object], 0))
    height_object = int(203 - height_object - 11)

    kolom = frame[:, 30:42]
    kolom = np.argwhere(np.multiply(np.multiply(kolom != np.roll(kolom, 1, axis=0), kolom != np.roll(kolom, 1, axis=1)), kolom == np.roll(np.roll(kolom, 1, axis=1), 1, axis=0)))
    if kolom.shape[0] == 0:
        height_flappy = 0
    else:
        middelpunt = np.mean(kolom, axis=0)
        height_flappy = int(middelpunt[0])

    features = np.array([height_flappy, height_object, dist_object])
    inds = np.where(np.isnan(features))
    features[inds] = 0

    return features

def calculate_state(current_frame, previous_frame=np.zeros(shape=(203, 144))):
    current_features = calculate_features(current_frame)
    previous_features = calculate_features(previous_frame)
    difference = (current_features[1] - current_features[0]) // 5
    dist_object = current_features[2] // 5

    alive = 1
    if np.sum(current_frame[199:202, 0:50]) / 255 > 10:
        alive = 0
    elif current_features[2] == previous_features[2] and current_features[2] == 45:
        #print('flappy is nu al dood')
        alive = 0
    elif np.sum(current_frame[30:32, :]) == 8415:
        # in menu
        alive = 0

    state = np.array([dist_object, difference, alive])
    inds = np.where(np.isnan(state))
    state[inds] = 0
    return state


def calculate_reward(previous_frame, current_frame, previous_state, current_state):
    """"
    calculate the reward of the state. Flappy is dead when the screen has stopped moving, so when two consecutive frames
    are equal. A point is scored when an obstacle is above flappy, and before it wasn't. An object is above Flappy when
    there are two white pixels in the first 50 pixels on the first row.

    :param frame: ...
    :return reward: int representing the reward if a point is scored or if flappy has died.
    """

    if np.sum(current_frame[30:32, :]) == 8415:
        #print("in menu")
        return -100, True
    elif np.sum(current_frame[:, 0:50]) / 255 == 50:
        #print("buiten scherm")
        return -100, True
    elif np.sum(current_frame[0,12:42]) == 510 and np.sum(current_frame[190,12:42]) == 510:
        #print("punt gescoord!")
        return 100, True
    elif current_state[2] == 0:
        #print("flappy is dood")
        return -100, False
    else:
        r = 50 - np.abs(current_state[1]) * 2
        print(r)
        return r, True

def main():
    # Settings.
    show_processed_image = False
    show_time_to_process = False
    reset_Q = True
    reset_memory = True
    epsilon = random.random() / 1
    time_to_play = 10
    FPS = 30
    framerate = 1/FPS
    print('epsilon: ', epsilon)

    # Open window to show the processed image.
    if show_processed_image:
        cv2.imshow('processed image', np.zeros(shape=(203, 144)).astype('uint8'))
        cv2.moveWindow('processed image', 0, 600)
        cv2.waitKey(1)

    # Launch game and locate screen
    launch_flappy()
    app = get_application()
    coordinates = get_window_coordinates(app)

    # Prepare state and Q
    current_frame = grab_frame(coordinates)
    current_state = calculate_state(current_frame)
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    q = nn.Q(reset_Q)
    new_memories = pd.DataFrame(columns=['previous_state', 'action', 'reward', 'current_state', 'weight'])
    alive = True
    t_end = time.time() + time_to_play

    while time.time() < t_end:
        start = time.time()

        # perform action
        action = q.predict_action(current_state)

        if action == 1:
            if random.random() < epsilon:
                action = 0
            else:
                press_space()
                pass

        # make memory (previous_state, action, reward, current_state)
        previous_frame = current_frame
        current_frame = grab_frame(coordinates)
        previous_state = current_state
        current_state = calculate_state(current_frame, previous_frame)
        reward, alive = calculate_reward(previous_frame, current_frame, previous_state, current_state)
        #print('reward: ',reward, '    alive: ', alive)
        #print('current_state: ',current_state)


        # Store memory.
        new_memories = new_memories.append(pd.Series([previous_state, action, reward, current_state, 1],
                                             index=['previous_state', 'action', 'reward', 'current_state', 'weight']),
                                   ignore_index='True')

        # Learn from memory.
        #q.train(previous_state, action, reward, current_state)

        # Update the processed image.
        if show_processed_image:
            cv2.imshow('processed image', current_frame)
            cv2.waitKey(1)

        # Wait for next frame.
        if show_time_to_process and 'time_to_process' in locals():
                print('time to process', time_to_process)
        time_to_process = time.time()-start
        time.sleep(max(0.0, framerate - time_to_process))

    kill_app(app)
    cv2.destroyAllWindows()

    # Load memories from disk if they exist.
    if reset_memory:
        memories = new_memories
        print('Memories zijn gereset.')
    else:
        try:
            memories = pd.read_pickle('memories.pkl')
            memories = pd.concat([memories, new_memories])
            print('Memories loaded from disk (memories.csv).')
        except:
            memories = new_memories
            print('Kon memories niet laden van schijf. Nieuw memories dataframe aangemaakt.')

    print('Update memories and learn from them.')
    memories = memories.sample(min(memories.shape[0], 1000000))
    memories['previous_state'] = memories['previous_state'].apply(tuple)
    memories['current_state'] = memories['current_state'].apply(tuple)
    memories = memories.groupby(['previous_state', 'action', 'reward', 'current_state']).sum().reset_index()
    memories = memories[memories.current_state.apply(lambda x: x[2] == 1)]
    memories['previous_state'] = memories['previous_state'].apply(np.array)
    memories['current_state'] = memories['current_state'].apply(np.array)
    memories.to_pickle('memories.pkl')
    q.train_with_memory(memories)
    q.save()

if __name__ == "__main__":
    main()
