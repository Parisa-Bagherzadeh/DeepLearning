import argparse
import numpy as np
import cv2
from tensorflow.keras.models import load_model



model=load_model('AgePrediction.h5')


width=height=100

parser=argparse.ArgumentParser(description='Age Prediction deep learning')
parser.add_argument("--input",type=str,help='path of the input image')
parser.add_argument("--output",type=str,help='path of the output image')

args=parser.parse_args()
img=cv2.imread(args.input)
img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
img=cv2.resize(img,(width,height))
img=img/255.0
img=img[np.newaxis,...]
result=model.predict(img)

print("Age :",int(result))
