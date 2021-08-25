# -*- coding: utf-8 -*-
"""
Created on Mon Jul 12 15:52:05 2021

@author: Littfi
"""

import tkinter as tk
from tkinter import filedialog
from tkinter import *
from PIL import ImageTk, Image
import tensorflow as tf
import numpy as np

from keras.models import load_model
import warnings

model = load_model('C:/Users/Littfi/Desktop/monica.h5')
warnings.filterwarnings("ignore", category=DeprecationWarning)

#dictionary to label all traffic signs class.
classes = { 1:"M_RightTurn",
            2:"M_Roundabout",
            3:"M_StraightOrLeftTurn",
            4:"P_NoEntry",
            5:"P_SpeedLimit100",
            6:"P_SpeedLimit120",
            7:"P_SpeedLimit40",
            8:"P_SpeedLimit50",
            9:"P_SpeedLimit60",
            10:"P_SpeedLimit70",
            11:"P_SpeedLimit80",
            12:"W_CurveToLeft",
            13:"W_CurveToRight",
            14:"W_RoadTrafficLights",
            15:"W_SlipperyRoad",
            16:"W_UnevenRoad"      
           }

window = tk.Tk()          
window.geometry('600x500')
window.title('Traffic sign classifier')

window.configure(background='#1e3e64')

heading = Label(window, text="Traffic Sign Classifier",padx=220, font=('Verdana',20,'bold'))
heading.configure(background='#143953',foreground='white')
heading.pack()

sign = Label(window)
sign.configure(background='#1e3e64')

value = Label(window,font=('Helvetica',15,'bold'))
value.configure(background='#1e3e64')

def classify(file_path):
    global label_packed
    image = Image.open(file_path)
    image = image.resize((32,32))
    image = np.expand_dims(image, axis=0)
    image = np.array(image)
    image = tf.cast(image, tf.float32)
    print(image.shape)
    pred = model.predict(image)
    # print(pred.argmax())
    sign = classes[pred.argmax()+1]
    print(sign)
    value.configure(foreground='#ffffff', text=sign)

def show_cb(file_path):
    classify_b=Button(window,text="Classify Image",command=lambda: classify(file_path),padx=20,pady=5)
    classify_b.configure(background='#147a81', foreground='white',font=('arial',10,'bold'))
    classify_b.place(relx=0.6,rely=0.8)
    
def uploader():
    try:
        file_path = filedialog.askopenfilename()
        uploaded = Image.open(file_path)
        uploaded.thumbnail(((window.winfo_width()/2.25),(window.winfo_height()/2.25)))
        im = ImageTk.PhotoImage(uploaded)
        
        sign.configure(image=im)
        sign.image=im
        value.configure(text='')
        show_cb(file_path)
    except:
        pass

upload = Button(window,text="Upload an image",command=uploader,padx=10,pady=5)
upload.configure(background='#e8d08e', foreground='#143953',font=('arial',10,'bold'))
upload.pack()
upload.place(x=100, y=400)

sign.pack()
sign.place(x=230,y=100)
value.pack()
value.place(x=240,y=300)

window.mainloop()  