import cv2
import numpy as np
import argparse
import matplotlib.pyplot as plt

from tensorflow.keras.models import load_model
from yaml import parse

width=height=256

parser = argparse.ArgumentParser()
parser.add_argument("--input_model",type=str,help='path of the model')
parser.add_argument("--input_image ", type=str,help='Path of the input image')
args = parser.parse_args()

model=load_model(args.input_model)

img=cv2.imread(args.input_image)
img=cv2.cvtColor(img,cv2.BGR2RGB)
img=cv2.resize(img,(256,256)).astype(np.float32)
img=(img/127.5)-1
img = img[np.newaxis,...]

generator = model(img,training=True)
generator= np.squeeze(generator, axis=0)
generator = np.array((generator +1) *127.5).astype(np.uint8)
cv2.imshow('result',generator)

