import imp
from pyexpat import model
from PySide6.QtWidgets import *
from PySide6.QtUiTools import *
from PySide6.QtCore import *
from PySide6.QtGui import *
from soupsieve import select

from tensorflow.keras.models import load_model

import numpy as np
import cv2
import argparse


width=height=256

parser = argparse.ArgumentParser()
parser.add_argument("--input_model",type=str,help='path of the model')
args = parser.parse_args()


class UI(QMainWindow):
    def __init__(self):
        super().__init__()
        loader=QUiLoader()
        self.ui=loader.load('form.ui',None)
        self.ui.show()
        self.ui.btn_choosemodel.clicked.connect(self.choose_model)
        self.ui.btn_chooseimage.clicked.connect(self.choose_image)
        self.ui.btn_result.clicked.connect(self.result)
       
    
    def choose_model(self):
        selected_model=QFileDialog.getOpenFileName(self,"Choose Model",".","All Files (*.*)")
        self.ui.txt_model.setText=selected_model
        self.model=load_model(selected_model)
        
        

    def choose_image(self):
        img_file=QFileDialog.getOpenFileName(self,"Choose Image",".","All Files (*.*)")
        img = cv2.imread(img_file[0])
        self.img_rgb = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        img = QImage(self.img_rgb, self.img_rgb.shape[1], self.img_rgb.shape[0],QImage.Format_RGB888)
        set_image = QPixmap.fromImage(img)
        self.ui.lbl_image.setPixmap(set_image)


    def result(self):
        image = cv2.resize(self.img_rgb,(256,256)).astype(np.float32)
        image = image[np.newaxis,...]
        image = (image / 127.5) - 1
        generator = model(img,training=True)
        generator= np.squeeze(generator, axis=0)
        generator = np.array((generator +1) *127.5).astype(np.uint8)
        img = QImage(generator, generator.shape[1], generator.shape[0],QImage.Format_RGB888)
        result_image = QPixmap.fromImage(img)
        self.ui.lbl_result.setPixmap(result_image)



app=QApplication([])    
window=UI()
app.exec()
