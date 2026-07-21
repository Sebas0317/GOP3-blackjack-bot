import cv2
import numpy as np
import pyautogui

img = pyautogui.screenshot()
img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2GRAY)
cv2.imwrite("captured.png", img)
print(f"Capturada! ({img.shape[1]}x{img.shape[0]})")
