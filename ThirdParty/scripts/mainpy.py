import numpy as np
import cv2
import os

class Tester():
    def __init__(self, alpha=1):
        self.alpha = alpha
        print("item has been constructed")

    def reveal(self):
        print("Alpha is: {}".format(self.alpha))


def logger():
    print(os.path.dirname(os.path.abspath(__file__)))

def shaper(image):
    height = image.shape[-3]
    width = image.shape[-2]
    print("image height is: {} image width is: {}".format(height, width))

def getter():
    b = np.arange(10)
    b = np.reshape(b, (2, 5))
    return b