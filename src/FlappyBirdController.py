import time
import numpy as np
import cv2
import subprocess
from pywinauto.application import  Application
from src.objects import Environment, Agent


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


def get_application(title='Flappy Bird'):
    """"
    Connects to a window and returns this as an pywintauto.Application

    :param title: sting the name of the window
    :return pyWinAuto.application: returns the app
    """
    app = Application()
    app.connect(title=title)
    return app


def main():

    # showing the processed image
    cv2.imshow('processed image', np.zeros(shape=(203, 144)).astype('uint8'))
    cv2.moveWindow('processed image', 0, 600)
    cv2.waitKey(1)

    train = True

    # Play game and grab screen
    launch_flappy()
    app = get_application()

    environment = Environment(app, FPS=10, stacked_frames=4) #the reward functie doet het nog niet helemaal goed! dedecteert onterechte flappy is dood statements

    agent = Agent(environment)
    if train:
        agent.train(5)
    else:
        agent.play()

    environment.close()
    # # Updating the processed image
    # cv2.imshow('image', current_state[:, :, 3])
    # cv2.waitKey(1)

    # cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
