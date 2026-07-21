import os
import sys
import numpy as np
import cv2
import pyautogui


def resource_path(relative_path):
    """Get absolute path to resource, works for dev and for PyInstaller"""
    try:
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)


def safe_imread(file_path, flag=0):
    """Read image with proper path handling"""
    return cv2.imread(resource_path(file_path), flag)


def grab_screen():
    """Take screenshot in memory (no disk I/O) and return as grayscale numpy array"""
    pil_img = pyautogui.screenshot()
    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2GRAY)
