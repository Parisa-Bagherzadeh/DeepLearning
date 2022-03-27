
import cv2
import numpy as np
import os
import argparse

from tensorflow.keras.models import load_model


parser = argparse.ArgumentParser()
parser.add_argument("--input ", type=str)
args = parser.parse_args()

model = load_model("model/model.h5")

images = []
outputImage = np.zeros((64, 64, 3), dtype="uint8")

for img in os.listdir(args.input):
  image = cv2.imread(os.path.join(args.input_dir, img))
  image = cv2.resize(image, (32, 32))
  images.append(image)

outputImage[0:32, 0:32] = images[0]
outputImage[0:32, 32:64] = images[1]
outputImage[32:64, 32:64] = images[2]
outputImage[32:64, 0:32] = images[3]

outputImage = np.array(outputImage)
outputImage = outputImage / 255.0

outputImage = outputImage.reshape(1, 64, 64, 3)
pred = model.predict([outputImage])
print('House price estimated : ', pred)