import cv2
import pyautogui
import numpy as np
from constant import compute_layout
from utils import safe_imread, grab_screen

# Capture screen
screen = grab_screen()
print(f"Screen shape: {screen.shape}")

# Try to find the bet button
bet_img = safe_imread("image/bet/bet1k.png", 0)
if bet_img is None:
    print("ERROR: bet1k.png not found!")
else:
    print(f"Bet image shape: {bet_img.shape}")
    res = cv2.matchTemplate(screen, bet_img, cv2.TM_CCOEFF_NORMED)
    _, val, _, loc = cv2.minMaxLoc(res)
    print(f"Best match for bet: {val:.3f} at {loc}")

# Check result images
for name in ["win", "lose", "bust", "draw", "double", "stand", "blackjack"]:
    img = safe_imread(f"image/en-us/{name}.png", 0)
    if img is None:
        print(f"ERROR: {name}.png not found!")
    else:
        res = cv2.matchTemplate(screen, img, cv2.TM_CCOEFF_NORMED)
        _, val, _, loc = cv2.minMaxLoc(res)
        print(f"Best match for {name}: {val:.3f} at {loc}")
