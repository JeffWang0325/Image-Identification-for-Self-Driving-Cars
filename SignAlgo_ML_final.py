"""
File name: SignAlgo_ML_final.py.py
Date: 2021/07/27
Version: v01
Usage: 演算法對接主程式
"""


#region import package

import os
import numpy as np

# def load_image_files
import cv2
import glob
import joblib

#endregion

#plot
from matplotlib import pyplot as plt
# from sklearn.linear_model import LogisticRegression
# from sklearn.svm import SVC

class AI_Algo():
    classLabel= {
                "0":{"name":"CarOnly","info":"FollowSign","color":"blue"},
                "1":{"name":"DirectForMotor","info":"FollowSign","color":"blue"},
                "2":{"name":"KeepLeft","info":"FollowSign","color":"blue"},
                "3":{"name":"KeepRight","info":"FollowSign","color":"blue"},
                "4":{"name":"LeftTurn","info":"FollowSign","color":"blue"},
                "5":{"name":"RightTurn","info":"FollowSign","color":"blue"},
                "6":{"name":"Roundabout","info":"FollowSign","color":"blue"},
                "7":{"name":"StraightOnly","info":"FollowSign","color":"blue"},
                "8":{"name":"StraightOrLeft","info":"FollowSign","color":"blue"},
                "9":{"name":"StraightOrRight","info":"FollowSign","color":"blue"},
                "10":{"name":"Other_SignC","info":"Look UP","color":"green"},
                "11":{"name":"Other_SignD","info":"Look UP","color":"green"},
                "12":{"name":"Other_SignT","info":"Look UP","color":"green"},
                "13":{"name":"Other_SignT2","info":"Look UP","color":"green"},
                "14":{"name":"NoEntry","info":"Be Aware","color":"red"},
                "15":{"name":"NoLeftTurn","info":"Be Aware","color":"red"},
                "16":{"name":"NoRandLTurn","info":"Be Aware","color":"red"},
                "17":{"name":"NoRightTurn","info":"Be Aware","color":"red"},
                "18":{"name":"NoStopping","info":"Be Aware","color":"red"},
                "19":{"name":"NoUTurn","info":"Be Aware","color":"red"},
                "20":{"name":"ALTRightTurn","info":"Be Aware","color":"red"},
                "21":{"name":"SpdLimit100","info":"Be Aware","color":"red"},
                "22":{"name":"SpdLimit120","info":"Be Aware","color":"red"},
                "23":{"name":"SpdLimit20","info":"Be Aware","color":"red"},
                "24":{"name":"SpdLimit30","info":"Be Aware","color":"red"},
                "25":{"name":"SpdLimit40","info":"Be Aware","color":"red"},
                "26":{"name":"SpdLimit50","info":"Be Aware","color":"red"},
                "27":{"name":"SpdLimit60","info":"Be Aware","color":"red"},
                "28":{"name":"Spdimit70","info":"Be Aware","color":"red"},
                "29":{"name":"Spdimit80","info":"Be Aware","color":"red"},
                "30":{"name":"Children","info":"Caution","color":"orange"},
                "31":{"name":"CurveToLeft","info":"Caution","color":"orange"},
                "32":{"name":"CurveToRight","info":"Caution","color":"orange"},
                "33":{"name":"DoubleCurve1","info":"Caution","color":"orange"},
                "34":{"name":"DoubleCurve2","info":"Caution","color":"orange"},
                "35":{"name":"FallingRocks","info":"Caution","color":"orange"},
                "36":{"name":"OtherDanger","info":"Caution","color":"orange"},
                "37":{"name":"Pedestrians","info":"Caution","color":"orange"},
                "38":{"name":"RoadNarrows","info":"Caution","color":"orange"},
                "39":{"name":"RoadLights","info":"Caution","color":"orange"},
                "40":{"name":"SeparateIsland","info":"Caution","color":"orange"},
                "41":{"name":"SlipperyRoad","info":"Caution","color":"orange"},
                "42":{"name":"Slow","info":"Caution","color":"orange"},
                "43":{"name":"Stop","info":"Caution","color":"orange"},
                "44":{"name":"UnevenRoad","info":"Caution","color":"orange"},
                }
    
    preXclassLabel= {
               "0":{"name":"Blue Channel", "Model_path":"B_Logist_Model.sav"},
               "1":{"name":"Green Channel", "Model_path":"G_Logist_Model.sav"},
               "2":{"name":"Red Channel", "Model_path":"R_Logist_Model.sav"},
               "3":{"name":"RGB", "Model_path":"RGB_Logist_Model.sav"},
               "4":{"name":"Hue Channel","Model_path":"H_Logist_Model.sav"},
               "5":{"name":"Saturation Channel","Model_path":"S_Logist_Model.sav"},
               "6":{"name":"Value channel","Model_path":"V_Logist_Model.sav"},
               "7":{"name":"HSV","Model_path":"HSV_Logist_Model.sav"},
               "8":{"name":"Grayscale", "Model_path":"Gray_Logist_Model.sav"},
               "9":{"name":"Do EqualizeHist","Model_path":"EqualizeHist_Logist_Model.sav"},
               "10":{"name":"Do Canny","Model_path":"Canny_Logist_Model.sav"},
               "11":{"name":"Do EqualizeHist & Canny","Model_path":"GetEquCanny_Logist_Model.sav"}
               }

    class preX(): 
        def __init__(self):
        
            
            print('__init__')
            pass
        
        
        def GetRGB_Channel(data):
          B, G, R = [], [], []
          if data.ndim==3:
            b, g, r = data[:,:,0], data[:,:,1], data[:,:,2]     
            B, G, R = np.array(b), np.array(g), np.array(r)
          elif data.ndim==4:
            for i in data: 
              b, g, r = i[:,:,0], i[:,:,1], i[:,:,2]     
              B.append(b)
              G.append(g)
              R.append(r)
            B, G, R = np.array(B), np.array(G), np.array(R)
          return R, G, B      
        
        def GetRGB(data):
          data_rgb = []
          if data.ndim==3:
            a = cv2.cvtColor(data, cv2.COLOR_BGR2RGB)
            data_rgb = np.array(a)
              
          elif data.ndim==4:
            for i in range(len(data)): 
              a = cv2.cvtColor(data[i], cv2.COLOR_BGR2RGB)
              data_rgb.append(a)
            data_rgb = np.array(data_rgb)
          return data_rgb
         
        def GetHSV(data):
          data_hsv = []
          if data.ndim==3:
            a = cv2.cvtColor(data, cv2.COLOR_BGR2HSV)
            data_hsv = np.array(a)
        
          elif data.ndim==4: 
            for i in range(len(data)): 
              a = cv2.cvtColor(data[i], cv2.COLOR_BGR2HSV)
              data_hsv.append(a)
            data_hsv = np.array(data_hsv)
          return data_hsv
        
        def GetHSV_Channel(data):
          H, S, V = [], [], []
          if data.ndim==3:
              h, s, v = data[:,:,0], data[:,:,1], data[:,:,2]
              H, S, V = np.array(h), np.array(s), np.array(v)
          
          elif data.ndim==4:
            for i in data: 
              h, s, v = i[:,:,0], i[:,:,1], i[:,:,2]
              H.append(h)
              S.append(s)
              V.append(v)
            H, S, V = np.array(H), np.array(S), np.array(V)
          return H, S, V
        
        def GetGRAY(data):
          data_gray = []
          if data.ndim==3:
            a = cv2.cvtColor(data, cv2.COLOR_BGR2GRAY)
            data_gray = np.array(a)
        
          elif data.ndim==4:
            for i in range(len(data)):     
              a = cv2.cvtColor(data[i], cv2.COLOR_BGR2GRAY)
              data_gray.append(a)
            data_gray = np.array(data_gray)
          return data_gray
        
        def GetEqualizeHist(data):
          x = []
          if data.ndim==3:
            img = cv2.cvtColor(data, cv2.COLOR_BGR2GRAY)
            img = cv2.equalizeHist(img)
            EqualizeHist = np.array(img)
        
          elif data.ndim==4:
            for i in range(len(data)): 
              img = cv2.cvtColor(data[i], cv2.COLOR_BGR2GRAY)
              img = cv2.equalizeHist(img)
              x.append(img)
            EqualizeHist = np.array(x)
          return EqualizeHist
        
        def GetCanny(data):
          x = []
          if data.ndim==3:
            blurred = cv2.GaussianBlur(data, (3, 3), 0)
            canny = cv2.Canny(blurred, 20, 180)
            Canny = np.array(canny)
        
          elif data.ndim==4:
            for i in range(len(data)): 
              blurred = cv2.GaussianBlur(data[i], (3, 3), 0)
              canny = cv2.Canny(blurred, 20, 180)
              x.append(canny)
            Canny = np.array(x)
          return Canny
        
        def GetEquCanny(data):
          x = []
          if data.ndim==3:
            img = cv2.cvtColor(data, cv2.COLOR_BGR2GRAY)
            img = cv2.equalizeHist(img)
            blurred = cv2.GaussianBlur(img, (3, 3), 0)
            canny = cv2.Canny(blurred, 20, 180)
            EquCanny = np.array(canny)
        
          elif data.ndim==4:
            for i in range(len(data)): 
              img = cv2.cvtColor(data[i], cv2.COLOR_BGR2GRAY)
              img = cv2.equalizeHist(img)
              blurred = cv2.GaussianBlur(img, (3, 3), 0)
              canny = cv2.Canny(blurred, 20, 180)
              x.append(canny)
            EquCanny = np.array(x)
          return EquCanny
        
        def X_3channel(data):
            if data.ndim==3:
                X = data/ 255.
                n = len(data)
                X_3channel = X.reshape(1,3072).astype('float32')
            elif data.ndim==4:
                X = data/ 255.
                n = len(data)
                X_3channel = X.reshape(n,3072).astype('float32')
            return X_3channel
    
        def X_1channel(data):
            if data.ndim==3:
                X = data/ 255.
                n = len(data)
                X_1channel = X.reshape(n,1024).astype('float32')
            elif data.ndim==4:
                X = data/ 255.
                n = len(data)
                X_1channel = X.reshape(n,1024).astype('float32')
            return X_1channel

    def __init__(self): # 參數
        """
        Constructor: Class initialization
        """
        
        print('__init__')
        pass

    def loadModel(self, path):
        """
        Load model
        '''
        Parameters
        ----------
        path : string
            Loading model path
        """

        self.model = joblib.load(path)
        self.model_name = os.path.basename(path)
        print(self.model_name)
        return
    
    def predict(self, X_test):
        
        # Preprocessing X_test

        preXclassLabel = self.model_name
        
        if preXclassLabel == "B_Logist_Model.sav":
            preXclassLabel = '0'

        elif preXclassLabel == "G_Logist_Model":
            preXclassLabel = '1'
        elif preXclassLabel == "R_Logist_Model.sav":
            preXclassLabel = '2'            
        elif preXclassLabel == "RGB_Logist_Model.sav":
            preXclassLabel = '3'            
        elif preXclassLabel == "H_Logist_Model.sav":
            preXclassLabel = '4'
        elif preXclassLabel == "S_Logist_Model.sav":
            preXclassLabel = '5'
        elif preXclassLabel == "V_Logist_Model.sav":
            preXclassLabel = '6'
        elif preXclassLabel == "HSV_Logist_Model.sav":
            preXclassLabel = '7'
        elif preXclassLabel == "Gray_Logist_Model.sav":
            preXclassLabel = '8'
        elif preXclassLabel == "EqualizeHist_Logist_Model.sav":
            preXclassLabel = '9'
        elif preXclassLabel == "Canny_Logist_Model.sav":
            preXclassLabel = '10'
        elif preXclassLabel == "GetEquCanny_Logist_Model.sav":
            preXclassLabel = '11'
       
        if preXclassLabel == "0":
            r, g, b= self.preX.GetRGB_Channel(X_test)
            X = b
            
        elif preXclassLabel == "1":
            r, g, b= self.preX.GetRGB_Channel(X_test)
            X = g
        
        elif preXclassLabel == "2":
            r, g, b= self.preX.GetRGB_Channel(X_test)
            X = r
            
        elif preXclassLabel == "3": 
            X = self.preX.GetRGB(X_test)
            
        elif preXclassLabel == "4":
            h, s, v = self.preX.GetHSV_Channel(X_test)
            X = h
            
        elif preXclassLabel == "5":
            h, s, v = self.preX.GetHSV_Channel(X_test)
            X = s
            
        elif preXclassLabel == "6":
            h, s, v = self.preX.GetHSV_Channel(X_test)
            X = v
            
        elif preXclassLabel == "7":
            X = self.preX.GetHSV(X_test)
            
        elif preXclassLabel == "8":
            X = self.preX.GetGRAY(X_test)
            
        elif preXclassLabel == "9":
            X = self.preX.GetEqualizeHist(X_test)
            
        elif preXclassLabel == "10":
            X = self.preX.GetCanny(X_test)
            
        elif preXclassLabel == "11":
            X = self.preX.GetEquCanny(X_test)
        
        else:
            X = X_test
        

        # resized
        dimension = (32,32)
        raw_data_test = []
        if X.ndim==3:
            resized = cv2.resize(X, dimension, interpolation = cv2.INTER_AREA)
        elif X.ndim==4:
            for i in range(len(X)):
                resized = cv2.resize(X[i], dimension, interpolation = cv2.INTER_AREA)
                raw_data_test.append(resized)   

        # Flatten
        channel = X.shape
        if len(channel)==4 and channel[-1]==3:
            X_test = self.preX.X_3channel(X)
        
        elif len(channel)==3:
            X_test = self.preX.X_1channel(X)
            
        # Model
        
        model = self.model
        probability = model.predict_proba(X_test)
        y_pred_probability = list(probability)
        
        yy_pred = model.predict(X_test)
        predictresult = list(yy_pred)
        
        # 最終輸出格式: [('M_CarOnly', 0.35248378, 'M', "red")]
        output = []
        for i in range(len(yy_pred)):
          int_dict = str(predictresult[i])
          name_dict = self.classLabel[int_dict]['name']
          shorthand = self.classLabel[int_dict]['info']
          color = self.classLabel[int_dict]['color']
          x = y_pred_probability[i]


          prob = x.max()

          # prob = y_pred_probability[i].index(np.max(y_pred_probability[i]))
    
    
          output.append(tuple([name_dict, prob, shorthand, color]))
        return output
    
    def predict2(self, listImg: list, dimension=(32, 32), BGR=False): # Jeff Revised!
        # Preprocess
        raw_data_test = []
        for img in listImg:
            img = img[:, :, ::(-1 if BGR else 1)]
            resized = cv2.resize(img, dimension, interpolation = cv2.INTER_AREA)
            raw_data_test.append(resized)
        X_test = np.array(raw_data_test)

        # Preprocessing X_test
        preXclassLabel = self.model_name
        
        if preXclassLabel == "B_Logist_Model.sav":
            preXclassLabel = '0'
        elif preXclassLabel == "G_Logist_Model":
            preXclassLabel = '1'
        elif preXclassLabel == "R_Logist_Model.sav":
            preXclassLabel = '2'            
        elif preXclassLabel == "RGB_Logist_Model.sav":
            preXclassLabel = '3'            
        elif preXclassLabel == "H_Logist_Model.sav":
            preXclassLabel = '4'
        elif preXclassLabel == "S_Logist_Model.sav":
            preXclassLabel = '5'
        elif preXclassLabel == "V_Logist_Model.sav":
            preXclassLabel = '6'
        elif preXclassLabel == "HSV_Logist_Model.sav":
            preXclassLabel = '7'
        elif preXclassLabel == "Gray_Logist_Model.sav":
            preXclassLabel = '8'
        elif preXclassLabel == "EqualizeHist_Logist_Model.sav":
            preXclassLabel = '9'
        elif preXclassLabel == "Canny_Logist_Model.sav":
            preXclassLabel = '10'
        elif preXclassLabel == "GetEquCanny_Logist_Model.sav":
            preXclassLabel = '11'
       
        if preXclassLabel == "0":
            r, g, b= self.preX.GetRGB_Channel(X_test)
            X = b
            
        elif preXclassLabel == "1":
            r, g, b= self.preX.GetRGB_Channel(X_test)
            X = g
        
        elif preXclassLabel == "2":
            r, g, b= self.preX.GetRGB_Channel(X_test)
            X = r
            
        elif preXclassLabel == "3": 
            X = self.preX.GetRGB(X_test)
            
        elif preXclassLabel == "4":
            h, s, v = self.preX.GetHSV_Channel(X_test)
            X = h
            
        elif preXclassLabel == "5":
            h, s, v = self.preX.GetHSV_Channel(X_test)
            X = s
            
        elif preXclassLabel == "6":
            h, s, v = self.preX.GetHSV_Channel(X_test)
            X = v
            
        elif preXclassLabel == "7":
            X = self.preX.GetHSV(X_test)
            
        elif preXclassLabel == "8":
            X = self.preX.GetGRAY(X_test)
            
        elif preXclassLabel == "9":
            X = self.preX.GetEqualizeHist(X_test)
            
        elif preXclassLabel == "10":
            X = self.preX.GetCanny(X_test)
            
        elif preXclassLabel == "11":
            X = self.preX.GetEquCanny(X_test)
        
        else:
            X = X_test
        
        # resized
        raw_data_test = []
        if X.ndim==3:
            resized = cv2.resize(X, dimension, interpolation = cv2.INTER_AREA)
        elif X.ndim==4:
            for i in range(len(X)):
                resized = cv2.resize(X[i], dimension, interpolation = cv2.INTER_AREA)
                raw_data_test.append(resized)   

        # Flatten
        channel = X.shape
        if len(channel)==4 and channel[-1]==3:
            X_test = self.preX.X_3channel(X)
        
        elif len(channel)==3:
            X_test = self.preX.X_1channel(X)
            
        # Model
        model = self.model
        probability = model.predict_proba(X_test)
        y_pred_probability = list(probability)
        
        yy_pred = model.predict(X_test)
        predictresult = list(yy_pred)
        
        # 最終輸出格式: [('M_CarOnly', 0.35248378, 'M', "red")]
        output = []
        for i in range(len(yy_pred)):
          int_dict = str(predictresult[i])
          name_dict = self.classLabel[int_dict]['name']
          shorthand = self.classLabel[int_dict]['info']
          color = self.classLabel[int_dict]['color']
          x = y_pred_probability[i]


          prob = x.max()

          # prob = y_pred_probability[i].index(np.max(y_pred_probability[i]))
    
    
          output.append(tuple([name_dict, prob, shorthand, color]))
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

    #region Other functions

    #endregion

if __name__ == "__main__": # 演算法測試
   
    ai_Algo = AI_Algo()
    X_test = ai_Algo.load_image_files(r"D:\Machine Learning\工研院產業新尖兵\專題\交通號誌辨識\UNclear")
    
    ai_Algo.loadModel(r"D:\Machine Learning\工研院產業新尖兵\專題\結案報告\Demo\Sign\ML\Model\RGB_Logist_Model.sav")
    x = ai_Algo.predict(X_test)
    print(x)
    
    print('-' * 30)
    
    import glob
    listImg = []
    dir_img = r"D:\Machine Learning\工研院產業新尖兵\專題\交通號誌辨識\UNclear"
    for path in glob.glob(os.path.join(dir_img, '*.jpg')):
        print(path)
        listImg.append(cv2.imdecode(np.fromfile(path, dtype=np.uint8), 1))
    
    print(ai_Algo.predict2(listImg, BGR=True))