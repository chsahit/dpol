import numpy as np
import cv2
from PIL import Image

from pih_env import PIHEnv

env = PIHEnv()
obs, _ = env.reset()
img = env.render(mode="rgb_array")
img = Image.fromarray(img, "RGB")
print("saving env image")
img.save("scene.png")
