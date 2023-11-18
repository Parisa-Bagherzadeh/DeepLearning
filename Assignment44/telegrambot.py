import imp
import telebot
from tensorflow.keras.models import load_model
import cv2
import numpy as np


model=load_model('Classification_model.h5')

bot=telebot.TeleBot('5203029390:AAFureq_PPboyofeyVzSUVQtJpou-07KFig',parse_mode=None)

@bot.message_handler(commands=['start'])
def send_welcome(message):
    bot.send_message(message.chat.id,'Hello and welcome!')
    bot.send_message(message.chat.id,'This bot classifies 4 objects:'+'\n'+'\n'+
    'car 🚗'+'\n'+'dress 👗'+'\n'+'house🏠'+'\n'+'pizza🍕')

@bot.message_handler(func=lambda m: True, content_types=['photo'])
def get_image(message):
    file = bot.get_file(message.photo[-1].file_id)
    download = bot.download_file(file.file_path)
    path = file.file_path
    with open(path, 'wb') as f:
        f.write(download)

    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224,224))
    img = img/255
    img = img.reshape(1, 224,224, 3)
    pred = model.predict(img)


    result=np.argmax(pred)

    if result==0:
        bot.reply_to(message,"It's a car 🚗")
    elif result==1:
        bot.reply_to(message,"It's a dress 👗")
    elif result==2:
        bot.reply_to(message,"It's a house 🏠")   
    elif result==3:
        bot.reply_to(message,"It's pizza 🍕")
    


bot.polling()    