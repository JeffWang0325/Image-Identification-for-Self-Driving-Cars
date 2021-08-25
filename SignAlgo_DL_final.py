# -*- coding: utf-8 -*-
"""
Created on Sat Jul 24 12:43:27 2021

@author: user
"""

"""
File name: 演算法對接範例程式_v01.py
Date: 2021/07/19
Version: v01
Usage: 演算法對接主程式
"""


#region import package

import os
import numpy as np

# def load_image_files
import cv2
import glob

# def loadModel
from tensorflow.keras.models import load_model

# predict
from tensorflow.keras.models import Model

#endregion

import tensorflow as tf
if tf.test.is_gpu_available():
      print('有啟用GPU')
else:
      print('尚未啟用GPU')  

class AI_Algo():
      """
      A class used to ...

      ...

      Attributes
      ----------
      data : ndarray (n, w, h, 3)
          the input image of training model
      targets : ndarray (n, label)
          the target label of training model
      model : 
          Trained model

      Methods
      -------
      __init__(...)
          Class initialization
      fit(X, y)
          Train model
      saveModel(path):
          Save model
      loadModel(path):
          Load model
      predict(X):
          Predict model
      """

      classLabel= {
                  "0":{"name":"CarOnly","info":"FollowSign","color":"blue", 'Group': 'M'},
                  "1":{"name":"DirectForMotor","info":"FollowSign","color":"blue", 'Group': 'M'},
                  "2":{"name":"KeepLeft","info":"FollowSign","color":"blue", 'Group': 'M'},
                  "3":{"name":"KeepRight","info":"FollowSign","color":"blue", 'Group': 'M'},
                  "4":{"name":"LeftTurn","info":"FollowSign","color":"blue", 'Group': 'M'},
                  "5":{"name":"RightTurn","info":"FollowSign","color":"blue", 'Group': 'M'},
                  "6":{"name":"Roundabout","info":"FollowSign","color":"blue", 'Group': 'M'},
                  "7":{"name":"StraightOnly","info":"FollowSign","color":"blue", 'Group': 'M'},
                  "8":{"name":"StraightOrLeft","info":"FollowSign","color":"blue", 'Group': 'M'},
                  "9":{"name":"StraightOrRight","info":"FollowSign","color":"blue", 'Group': 'M'},
                  "10":{"name":"Other_SignC","info":"Look UP","color":"green", 'Group': 'P'},
                  "11":{"name":"Other_SignD","info":"Look UP","color":"green", 'Group': 'W'},
                  "12":{"name":"Other_SignT","info":"Look UP","color":"green", 'Group': 'W'},
                  "13":{"name":"Other_SignT2","info":"Look UP","color":"green", 'Group': 'W'},
                  "14":{"name":"NoEntry","info":"Be Aware","color":"red", 'Group': 'P'},
                  "15":{"name":"NoLeftTurn","info":"Be Aware","color":"red", 'Group': 'P'},
                  "16":{"name":"NoRandLTurn","info":"Be Aware","color":"red", 'Group': 'P'},
                  "17":{"name":"NoRightTurn","info":"Be Aware","color":"red", 'Group': 'P'},
                  "18":{"name":"NoStopping","info":"Be Aware","color":"red", 'Group': 'P'},
                  "19":{"name":"NoUTurn","info":"Be Aware","color":"red", 'Group': 'P'},
                  "20":{"name":"ALTRightTurn","info":"Be Aware","color":"red", 'Group': 'P'},
                  "21":{"name":"SpdLimit100","info":"Be Aware","color":"red", 'Group': 'P'},
                  "22":{"name":"SpdLimit120","info":"Be Aware","color":"red", 'Group': 'P'},
                  "23":{"name":"SpdLimit20","info":"Be Aware","color":"red", 'Group': 'P'},
                  "24":{"name":"SpdLimit30","info":"Be Aware","color":"red", 'Group': 'P'},
                  "25":{"name":"SpdLimit40","info":"Be Aware","color":"red", 'Group': 'P'},
                  "26":{"name":"SpdLimit50","info":"Be Aware","color":"red", 'Group': 'P'},
                  "27":{"name":"SpdLimit60","info":"Be Aware","color":"red", 'Group': 'P'},
                  "28":{"name":"Spdimit70","info":"Be Aware","color":"red", 'Group': 'P'},
                  "29":{"name":"Spdimit80","info":"Be Aware","color":"red", 'Group': 'P'},
                  "30":{"name":"Children","info":"Caution","color":"orange", 'Group': 'W'},
                  "31":{"name":"CurveToLeft","info":"Caution","color":"orange", 'Group': 'W'},
                  "32":{"name":"CurveToRight","info":"Caution","color":"orange", 'Group': 'W'},
                  "33":{"name":"DoubleCurve1","info":"Caution","color":"orange", 'Group': 'W'},
                  "34":{"name":"DoubleCurve2","info":"Caution","color":"orange", 'Group': 'W'},
                  "35":{"name":"FallingRocks","info":"Caution","color":"orange", 'Group': 'W'},
                  "36":{"name":"OtherDanger","info":"Caution","color":"orange", 'Group': 'W'},
                  "37":{"name":"Pedestrians","info":"Caution","color":"orange", 'Group': 'W'},
                  "38":{"name":"RoadNarrows","info":"Caution","color":"orange", 'Group': 'W'},
                  "39":{"name":"RoadLights","info":"Caution","color":"orange", 'Group': 'W'},
                  "40":{"name":"SeparateIsland","info":"Caution","color":"orange", 'Group': 'W'},
                  "41":{"name":"SlipperyRoad","info":"Caution","color":"orange", 'Group': 'W'},
                  "42":{"name":"Slow","info":"Caution","color":"orange", 'Group': 'W'},
                  "43":{"name":"Stop","info":"Caution","color":"orange", 'Group': 'W'},
                  "44":{"name":"UnevenRoad","info":"Caution","color":"orange", 'Group': 'W'},
                  }
    
      DictImage_stan_Sign = {}
      Dict_name_2_Group = {}

      def __init__(self): # 參數
          """
          Constructor: Class initialization
          """
          print('__init__')
          self.initial_DictImage_stan_Sign()
          self.initial_Dict_name_2_Group()
          return

      def initial_DictImage_stan_Sign(self):
          dir_image = 'GUI Image/Sign/Standard_Sample/'
          for i in range(len(self.classLabel)):
              path = dir_image + self.classLabel[str(i)]['Group'] + '/' + self.classLabel[str(i)]['name'] + '.jpg'
              self.DictImage_stan_Sign[self.classLabel[str(i)]['name']] = cv2.imdecode(np.fromfile(path, dtype=np.uint8), 1)
          return

      def initial_Dict_name_2_Group(self):
          for i in range(len(self.classLabel)):
              self.Dict_name_2_Group[self.classLabel[str(i)]['name']] = self.classLabel[str(i)]['Group']
          return
            
      def loadModel(self, path):
          """
          Load model
          '''
          Parameters
          ----------
          path : string
              Loading model path
          """
          with tf.device('/cpu:0'):
            self.model = load_model(path)
          return  

      def predict(self, X_test):
          with tf.device('/cpu:0'):
            y_pred = self.model.predict(X_test)
          # 整理y_pred的格式
          for i in range(len(y_pred)):
            temp_list = []
      
            temp_list.append(y_pred[i])
            arr_total = np.sum(temp_list)
            y_pred[i] = y_pred[i]/arr_total

          # 得到y_pred的機率
          y_pred_probability = []
          for i in range(len(y_pred)):
      
            probability = y_pred[i].max()
            y_pred_probability.append(probability)
      
          # 得到y_pred的label
          predictresult = [] 
          for i in range(X_test.shape[0]): 
            predictresult.append(y_pred[i].argmax())
      
          # 最終輸出格式: [('M_CarOnly', 0.35248378, 'M', "red")]
          output = []
          for i in range(len(y_pred)):
            int_dict = str(predictresult[i])
            name_dict = self.classLabel[int_dict]['name']
            shorthand = self.classLabel[int_dict]['info']
            color = self.classLabel[int_dict]['color']
      
            output.append(tuple([name_dict, y_pred_probability[i], shorthand, color]))
          return output
      
      def predict2(self, listImg: list, dimension=(32, 32), BGR=False): # Jeff Revised!
          # Preprocess
          raw_data_test = []
          for img in listImg:
              img = img[:, :, ::(-1 if BGR else 1)]
              resized = cv2.resize(img, dimension, interpolation = cv2.INTER_AREA)
              raw_data_test.append(resized)
          X_test = np.array(raw_data_test)
          
          with tf.device('/cpu:0'):
            y_pred = self.model.predict(X_test)
          # 整理y_pred的格式
          for i in range(len(y_pred)):
            temp_list = []
      
            temp_list.append(y_pred[i])
            arr_total = np.sum(temp_list)
            y_pred[i] = y_pred[i]/arr_total

          # 得到y_pred的機率
          y_pred_probability = []
          for i in range(len(y_pred)):
      
            probability = y_pred[i].max()
            y_pred_probability.append(probability)
      
          # 得到y_pred的label
          predictresult = [] 
          for i in range(X_test.shape[0]): 
            predictresult.append(y_pred[i].argmax())
      
          # 最終輸出格式: [('M_CarOnly', 0.35248378, 'M', "red")]
          output = []
          for i in range(len(y_pred)):
            int_dict = str(predictresult[i])
            name_dict = self.classLabel[int_dict]['name']
            shorthand = self.classLabel[int_dict]['info']
            color = self.classLabel[int_dict]['color']
      
            output.append(tuple([name_dict, y_pred_probability[i], shorthand, color]))
          return output
      
      def load_image_files(self, container_path, dimension=(32, 32)):
          """
          Load image files with categories as subfolder names 
          which performs like scikit-learn sample dataset
          Parameters
          ----------
          container_path : string or unicode
              Path to the main folder holding one subfolder per category
          dimension : tuple
              size to which image are adjusted to
          Returns
          -------
          Bunch
          """

          raw_data_test = []
          test_len = []
          size = dimension
          
          
          # link = container_path + "*.jpg"
          link = os.path.join(container_path, '*.jpg')
          path = glob.glob(link)
          
          pre_size = len(raw_data_test)
          #print(pre_size)

          
          for img in path:
            #n = cv2.imread(img)
            n = cv2.imdecode(np.fromfile(img, dtype=np.uint8), 1) # Jeff Revised!
            n = cv2.cvtColor(n, cv2.COLOR_BGR2RGB)
            resized = cv2.resize(n, size, interpolation = cv2.INTER_AREA)
            raw_data_test.append(resized)
          
          real_size = len(raw_data_test) - pre_size
          
          test_len.append(real_size)
          
          test_len = np.array(raw_data_test)

          return test_len

if __name__ == "__main__": # 演算法測試
    ai_Algo = AI_Algo()
    
    ai_Algo.loadModel(r"D:\Machine Learning\工研院產業新尖兵\專題\交通號誌辨識\Model\sign_batch.h5")

    #region Preprocess

    #讀取圖片需要Monica的奧援
    data = ai_Algo.load_image_files(r"D:\Machine Learning\工研院產業新尖兵\專題\交通號誌辨識\UNclear")
    #endregion
    x = ai_Algo.predict(data)
    print(x)

    #region Train model

    #endregion
    
    #region Test model

    print('-' * 30)
    
    # import glob
    # listImg = []
    # dir_img = 'D:/Machine Learning/工研院產業新尖兵/專題/Code/yolov5/yolov5/runs/detect/exp6/crops/sign/'
    # for path in glob.glob(dir_img + '*.jpg'):
    #     print(path)
    #     listImg.append(cv2.imdecode(np.fromfile(path, dtype=np.uint8), 1))
    
    # print(ai_Algo.predict2(listImg, BGR=True))

    #endregion
