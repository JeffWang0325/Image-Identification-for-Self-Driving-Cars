"""
File name: TrafficLightAlgo_v01.py
Date: 2021/07/22
Version: v01
Usage: 紅綠燈演算法對接主程式
"""

#region import package

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from sklearn import svm, metrics, datasets
from sklearn.utils import Bunch
from sklearn.model_selection import GridSearchCV, train_test_split

from skimage.io import imread
from skimage.transform import resize
import skimage
import pickle
import cv2

#endregion

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
    
    dir_image = 'GUI Image/Light/'
    DictImage_stan_Light = {'Green': cv2.imdecode(np.fromfile(dir_image + 'green_light.png', dtype=np.uint8), 1),
                            'Red': cv2.imdecode(np.fromfile(dir_image + 'red_light.png', dtype=np.uint8), 1),
                            'Others': cv2.imdecode(np.fromfile(dir_image + 'warning.png', dtype=np.uint8), 1)}

    def __init__(self): # 參數
        """
        Constructor: Class initialization
        """
        
        print('__init__')
        pass

    def fit(self, X_train, y_train):
        """
        Train model
        '''
        Parameters
        ----------
        X : ndarray (n, w, h, 3) or list(image array) or list(image array flatten)
            the input image of training model
        y : ndarray (n, label)
            the target label of training model
        
        """
        param_grid = [
        {'C': [1, 10, 100, 1000], 'kernel': ['linear']},
        {'C': [1, 10, 100, 1000], 'gamma': [0.001, 0.0001], 'kernel': ['rbf']},
        ]
        svc = svm.SVC(probability=True)
        clf = GridSearchCV(svc, param_grid)
        clf.fit(X_train, y_train)
        self.model = clf.best_estimator_
        return

    def saveModel(self, path):
        """
        Save model
        '''
        Parameters
        ----------
        path : string
            Saving model path
        """
        pickle.dump(self.model, open(path, 'wb'))
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
        # self.model = ...
        self.model = pickle.load(open(path, 'rb'))
        return

    def score(self, X_test, y_test):
        return self.model.score(X_test, y_test)

    def predict(self, X_test, probability=False):
        """
        Predict model
        '''
        Parameters
        ----------
        X : ndarray (n, w * h * 3)
            the input images
        '''
        return
        ------
        (n, information), where information: (label, confidence, description or warning, color)
        e.g. [('Red', 0.94, 'Stop!', 'red'), ('Green', 0.64, 'Go!', 'green'), ('Red', 0.74, 'Stop!', 'red')]
        """
        classLabel = ['Green', 'Red', 'Others']
        classHint = ['GO', 'STOP', 'WARNING']
        classColor = ['green', 'red', 'yellow']

        y_pred = self.model.predict(X_test)
        if probability: # 待驗證!!!
            y_pred_proba_ = self.model.predict_proba(X_test)
            y_pred_proba = []
            for i in range(len(y_pred)):
                y_pred_proba.append(y_pred_proba_[i, y_pred[i]])
        else:
            y_pred_proba = [1.0 for i in range(len(y_pred))]

        return [(classLabel[y_pred[i]], y_pred_proba[i], classHint[y_pred[i]], classColor[y_pred[i]]) for i in range(len(y_pred))]

    def predict2(self, listImg: list, probability=False, dimension=(20, 20), BGR=False):
        """
        Predict model
        '''
        Parameters
        ----------
        X : ndarray (n, w, h, 3)
            the input images
        '''
        return
        ------
        (n, information), where information: (label, confidence, description or warning, color)
        e.g. [('Red', 0.94, 'Stop!', 'red'), ('Green', 0.64, 'Go!', 'green'), ('Red', 0.74, 'Stop!', 'red')]
        """
        if len(listImg) == 0:
            return []

        flat_data = []
        for img in listImg:
            img = img[:, :, ::(-1 if BGR else 1)]
            img_resized = resize(img, dimension, anti_aliasing=True, mode='reflect')
            flat_data.append(img_resized.flatten())

        X_test = np.array(flat_data)

        classLabel = ['Green', 'Red', 'Others']
        classHint = ['GO', 'STOP', 'WARNING']
        classColor = ['green', 'red', 'goldenrod']

        y_pred = self.model.predict(X_test)
        if probability: # 待驗證!!!
            y_pred_proba_ = self.model.predict_proba(X_test)
            y_pred_proba = []
            for i in range(len(y_pred)):
                y_pred_proba.append(y_pred_proba_[i, y_pred[i]])
        else:
            y_pred_proba = [1.0 for i in range(len(y_pred))]

        return [(classLabel[y_pred[i]], y_pred_proba[i], classHint[y_pred[i]], classColor[y_pred[i]]) for i in range(len(y_pred))]
        

    #region Other functions

    def load_image_files(self, container_path, dimension=(20, 20)):
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
        image_dir = Path(container_path)
        folders = [directory for directory in image_dir.iterdir() if directory.is_dir()]
        categories = [fo.name for fo in folders]

        descr = "A image classification dataset"
        images = []
        flat_data = []
        target = []
        for i, direc in enumerate(folders):
            for file in direc.iterdir():
                if str(file).split('.')[-1] != 'jpg':
                    continue
                print(str(file))
                img = skimage.io.imread(file)
                img_resized = resize(img, dimension, anti_aliasing=True, mode='reflect')
                flat_data.append(img_resized.flatten()) 
                images.append(img_resized)
                target.append(i)
        
        flat_data = np.array(flat_data)
        target = np.array(target)
        images = np.array(images)

        return Bunch(data=flat_data,target=target,target_names=categories,images=images,DESCR=descr)

    #endregion



if __name__ == "__main__": # 演算法測試
    
    ai_Algo = AI_Algo()

    #region Preprocess

    path = r"D:\Machine Learning\工研院產業新尖兵\專題\Traffic_Light\lightTest"
    # path = r"D:\Machine Learning\工研院產業新尖兵\專題\Traffic_Light\Traffic_Light_All\images"
    # image_dataset = ai_Algo.load_image_files(path)
    # X_train, X_test, y_train, y_test = train_test_split(image_dataset.data, image_dataset.target, test_size=0.3,random_state=109)

    #endregion
    
    #region Train model

    # ai_Algo.fit(X_train, y_train)
    # ai_Algo.saveModel('D:/Machine Learning/工研院產業新尖兵/專題/Traffic_Light/Traffic_Light_All/svm_model_2.sav')

    #endregion
    
    #region Test model

    # ai_Algo.loadModel('D:/Machine Learning/工研院產業新尖兵/專題/Traffic_Light/Model/svm_model.sav')
    ai_Algo.loadModel('D:\Machine Learning\工研院產業新尖兵\專題\Traffic_Light\Traffic_Light_All/svm_model_2.sav')
    # print(f'score: {ai_Algo.score(X_test, y_test)}')

    print('-' * 30)

    # print(ai_Algo.predict(X_test, probability=True))

    print('-' * 30)
    
    import glob
    listImg = []
    dir_img = 'D:/Machine Learning/工研院產業新尖兵/專題/Traffic_Light/test/'
    # dir_img = 'D:/Machine Learning/工研院產業新尖兵/專題/Code/yolov5/yolov5/runs/detect/exp4/crops/light/'
    for path in glob.glob(dir_img + '*.jpg'):
        print(path)
        # listImg.append(cv2.imdecode(np.fromfile(path, dtype=np.uint8), 1))
        listImg.append(skimage.io.imread(path))
    
    print(ai_Algo.predict2(listImg, probability=True))

    #endregion