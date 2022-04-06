
from tensorflow.keras import Model
from tensorflow.keras.layers import Conv2D,Flatten,Dense,MaxPooling2D,Dropout

width=height=224

class MyNet(Model):
  def __init__(self):
    super().__init__()

    self.pooling=MaxPooling2D()
    self.dense_1=Dense(128,activation='relu')
    self.dense_2=Dense(64,activation='relu')
    self.dense_3=Dense(14,activation='softmax')
    self.conv2d_1=Conv2D(32,(3,3),activation='relu',input_shape=(width,height,3))
    self.conv2d_2=Conv2D(64,(3,3),activation='relu')
    self.conv2d_3=Conv2D(128,(3,3),activation='relu')
    self.conv2d_4=Conv2D(256,(3,3),activation='relu')
    self.flatten=Flatten()
    self.droupout=Dropout(0.2)

  def call(self,x):
    layer1=self.conv2d_1(x)  
    layer2=self.pooling(layer1)

    layer3=self.conv2d_2(layer2)
    layer4=self.pooling(layer3)

    layer5=self.conv2d_3(layer4)
    layer6=self.pooling(layer5)

    layer7=self.conv2d_4(layer6)
    layer8=self.pooling(layer7)

    layer9=self.flatten(layer8)
    layer13=self.droupout(self.dense_1(layer9))
  
    layer14=self.dense_2(layer13)
    output=self.dense_3(layer14)
    return output