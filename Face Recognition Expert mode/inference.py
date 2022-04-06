import argparse
import cv2
from model import MyNet
import numpy as np

parser=argparse.ArgumentParser()
parser.add_argument("--input",type=str,help="Enter the path of the input image")

args=parser.parse_args()

model=MyNet()
image=cv2.imread(args.input)
image=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
image=cv2.resize(image,(244,244))
image=image/255.0
image = image[np.newaxis, ...]
pred=np.argmax(model.predict(image))

name_data= ["Ali Khamenei","Angelina Jolie","Barak Obama","Behnam Bani","Donald Trump","Emma Watson","Han Hye Jin",
"Kim Jong Un","Leyla Hatami","Lionel Messi","Michelle Obama","Morgan Freeman",
    "Queen Elizabeth","Scarlett Johanson"]

print(name_data[pred])    