# Description
This project achieves some functions of image identification for Self-Driving Cars.

First, use **yolov5** for **object detection** whose class includes car, truck, pedestrian, bicyclist, traffic light, traffic sign, motor and large vehicle.

Second, crop the images of **traffic light** and **traffic sign** to execute the **image classification** respectively.

Furthermore, the GUI of this project makes it more user-friendly for users to realize the image identification for Self-Driving Cars.  
For example: input source (e.g., **Folder, Image, YouTube, DroidCam, WebCam**), image display, parameter adjustment, information page, etc.

It is written in Python and uses Tkinter for its graphical user interface (GUI).

# Software and Hardware Environment
| IDE (optional)              | Visual Studio Code       |
| :-------------------------- | :----------------------- |
| Extensions                  | Python                   |
| Programming Language        | Python                   |
| Python Version              | Python 3.7.10            |
| Python Package              | Refer to [requirements.txt](https://github.com/JeffWang0325/Image-Identification-for-Self-Driving-Cars/blob/master/requirements.txt)|
| GPU (preferred)             | GTX 1080 Ti or higher    |

# Quick Start Examples
<details open>
<summary>Install</summary>

Install [Visual Studio Code](https://code.visualstudio.com/download) and Python 3.7.10 required with all [requirements.txt](https://github.com/JeffWang0325/Image-Identification-for-Self-Driving-Cars/blob/master/requirements.txt) dependencies installed:
<!-- $ sudo apt update && apt install -y libgl1-mesa-glx libsm6 libxext6 libxrender-dev -->
```bash
$ git clone https://github.com/JeffWang0325/Image-Identification-for-Self-Driving-Cars.git
$ cd Image-Identification-for-Self-Driving-Cars
$ pip install -r requirements.txt
```
</details>

<details open>
<summary>Execute GUI</summary>
  
* **Step1**: [Install all models and images](https://drive.google.com/file/d/1F4fnQ4ZiFhmI6yc-XiQ3TS9MGayoeTe4/view?usp=sharing)
* **Step2**: Move the files to the appropriate path
* **Step3**: Execute detect_Main_Jeff.py
  
</details>

# GUI Demo & Report:

Please click the following figures or links to watch GUI demo videos or report:  
[自駕車影像辨識系統 (Image Identification for Self-Driving Cars using Python Tkinter )-English Version](https://youtu.be/l6lmuLPjNUY)  
[![](http://img.youtube.com/vi/l6lmuLPjNUY/sddefault.jpg)](https://youtu.be/l6lmuLPjNUY)  

[專題報告: 自駕車影像辨識系統 (Image Identification for Self-Driving Cars)-HD](https://youtu.be/PqvCH86_cIs)  
[![](http://img.youtube.com/vi/PqvCH86_cIs/sddefault.jpg)](https://youtu.be/PqvCH86_cIs)  

[專題報告: 自駕車影像辨識系統 (Image Identification for Self-Driving Cars)](https://youtu.be/6i0p-vnwRN4)  
[![](http://img.youtube.com/vi/6i0p-vnwRN4/sddefault.jpg)](https://youtu.be/6i0p-vnwRN4)   

[GUI Demo1 using Python Tkinter (Image Identification for Self Driving Cars)-中文版](https://youtu.be/SS-Cb4kZask)  
[![](http://img.youtube.com/vi/SS-Cb4kZask/sddefault.jpg)](https://youtu.be/SS-Cb4kZask)   

[GUI Demo2 using Python Tkinter (Image Identification for Self Driving Cars)-中文版](https://youtu.be/lewPH9_w_-U)  
[![](http://img.youtube.com/vi/lewPH9_w_-U/sddefault.jpg)](https://youtu.be/lewPH9_w_-U)   

[GUI Demo3 using Python Tkinter (Image Identification for Self Driving Cars)](https://youtu.be/R0lHuH2iOAE)  
[![](http://img.youtube.com/vi/R0lHuH2iOAE/sddefault.jpg)](https://youtu.be/R0lHuH2iOAE)   

## ※Outline:   
***0. Introduction***
![](https://github.com/JeffWang0325/Image-Identification-for-Self-Driving-Cars/blob/master/README%20Image/0.jpg "Logo Title Text 1")

●4 Parts of GUI: Model Input Setting, Input Sources, Image Display, Information Page

***1. Input Source - Folder***
![](https://github.com/JeffWang0325/Image-Identification-for-Self-Driving-Cars/blob/master/README%20Image/1.jpg "Logo Title Text 1")

●Inside the folder, they can be either images or videos.

***2. Input Source - Image***
![](https://github.com/JeffWang0325/Image-Identification-for-Self-Driving-Cars/blob/master/README%20Image/2.jpg "Logo Title Text 1")

***3. Information Page***
![](https://github.com/JeffWang0325/Image-Identification-for-Self-Driving-Cars/blob/master/README%20Image/3-1.jpg "Logo Title Text 1")

![](https://github.com/JeffWang0325/Image-Identification-for-Self-Driving-Cars/blob/master/README%20Image/3-2.jpg "Logo Title Text 1")

※More Information:  
●Image height and width  
●Object detection result  
●Computation time of Yolov5, traffic light and traffic sign  

***4. Input Source - YouTube***
![](https://github.com/JeffWang0325/Image-Identification-for-Self-Driving-Cars/blob/master/README%20Image/4.jpg "Logo Title Text 1")

***5. Input Source - DroidCam***
![](https://github.com/JeffWang0325/Image-Identification-for-Self-Driving-Cars/blob/master/README%20Image/5-1.jpg "Logo Title Text 1")

●Write the **IP Cam Access** in the textbox.

![](https://github.com/JeffWang0325/Image-Identification-for-Self-Driving-Cars/blob/master/README%20Image/5-2.jpg "Logo Title Text 1")

***6. Parameter Adjustment - Yolo v5***
![](https://github.com/JeffWang0325/Image-Identification-for-Self-Driving-Cars/blob/master/README%20Image/6.jpg "Logo Title Text 1")

***7. Parameter Adjustment - Sign (DL)***
![](https://github.com/JeffWang0325/Image-Identification-for-Self-Driving-Cars/blob/master/README%20Image/7.jpg "Logo Title Text 1")

***8. Information Page - Others***
![alt text](https://github.com/JeffWang0325/Image-Identification-for-Self-Driving-Cars/blob/master/README%20Image/8.jpg "Logo Title Text 1")

---
# Contact Information:
If you have any questions or suggestions about code, project or any other topics, please feel free to contact me and discuss with me. 😄😄😄

<a href="https://www.linkedin.com/in/tzu-wei-wang-a09707157" target="_blank"><img src="https://github.com/JeffWang0325/JeffWang0325/blob/master/Icon%20Image/linkedin_64.png" width="64"></a>
<a href="https://www.youtube.com/channel/UC9nOeQSWp0PQJPtUaZYwQBQ" target="_blank"><img src="https://github.com/JeffWang0325/JeffWang0325/blob/master/Icon%20Image/youtube_64.png" width="64"></a>
<a href="https://www.facebook.com/tzuwei.wang.33/" target="_blank"><img src="https://github.com/JeffWang0325/JeffWang0325/blob/master/Icon%20Image/facebook_64.png" width="64"></a>
<a href="https://www.instagram.com/tzuweiw/" target="_blank"><img src="https://github.com/JeffWang0325/JeffWang0325/blob/master/Icon%20Image/instagram_64.png" width="64"></a>
<a href="https://www.kaggle.com/tzuweiwang" target="_blank"><img src="https://github.com/JeffWang0325/JeffWang0325/blob/master/Icon%20Image/kaggle_64.png" width="64"></a>
<a href="https://github.com/JeffWang0325" target="_blank"><img src="https://github.com/JeffWang0325/JeffWang0325/blob/master/Icon%20Image/github_64.png" width="64"></a>
