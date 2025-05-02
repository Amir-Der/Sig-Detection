import cv2
from skimage.feature import hog
import numpy as np

def preprocess_image(file_stream):
    file_bytes = np.frombuffer(file_stream.read(), np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (150, 150))
    img = cv2.GaussianBlur(img, (3, 3), 0)
    img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                 cv2.THRESH_BINARY, 11, 2)
    return img

def extract_hog_feature(img):
    return hog(img,
               orientations=18,
               pixels_per_cell=(3, 3),
               cells_per_block=(2, 2),
               block_norm="L2-Hys")
