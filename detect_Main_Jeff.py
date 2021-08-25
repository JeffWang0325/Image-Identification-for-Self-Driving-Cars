"""Run inference with a YOLOv5 model on images, videos, directories, streams

Usage:
    $ python path/to/detect.py --source path/to/img.jpg --weights yolov5s.pt --img 640
"""

import argparse
import sys
import time
from pathlib import Path

import cv2
import torch
import torch.backends.cudnn as cudnn

FILE = Path(__file__).absolute()
sys.path.append(FILE.parents[0].as_posix())  # add yolov5/ to path

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, check_imshow, colorstr, non_max_suppression, \
    apply_classifier, scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path, save_one_box, get_bbox_area
from utils.plots import colors, plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized

import PIL
import tkinter
import numpy as np

# import os
# os.system(" python SignAlgo_DL_final.py")

import TrafficLightAlgo_v01 as LightAlgo
import SignAlgo_DL_final as SignAlgo_DL
import SignAlgo_ML_final as SignAlgo_ML
import gc


B_StopRun = False

class Parameters_Project:
    link_Youtube = 'https://www.youtube.com/watch?v=nHUbCoPS-Xc&ab_channel=PinoSu' # Youtube
    link_DroidCam = 'http://192.168.0.101:4747/video' # DroidCam

    def __init__(self):
        pass


class Parameters_YOLO():
    def __init__(self, WaitTime = 10, save_img = False, save_crop = False, weights_path = 'yolov5x.pt',
                 source_path = 'D:/Machine Learning/工研院產業新尖兵/專題/yolo/yolov5/plate/images/valid',
                 Imgsz = 512, Conf_thres = 0.45, Iou_thres = 0.45, Max_det = 1000, save_txt = False, line_thickness = 3, hide_labels=False, hide_conf=False):
        self.WaitTime = WaitTime
        self.save_img = save_img
        self.save_crop = save_crop
        self.weights_path = weights_path
        self.source_path = source_path

        self.Imgsz = Imgsz
        self.Conf_thres = Conf_thres
        self.Iou_thres = Iou_thres
        self.Max_det = Max_det
        self.save_txt = save_txt
        self.line_thickness = line_thickness
        self.hide_labels = hide_labels
        self.hide_conf = hide_conf

    def Update_Parameters(self, e = None):
        self.save_crop = True if str(root.getvar('chk_btnsavebboxImg')) == '1' else False
        self.save_txt = True if str(root.getvar('chk_btnsavetxt')) == '1' else False
        self.WaitTime = int(root.getvar('ent_WaitTime'))
        self.Imgsz = int(root.getvar('ent_Imgsz'))
        self.Conf_thres = float(root.getvar('ent_Conf_thres'))
        self.Iou_thres = float(root.getvar('ent_Iou_thres'))
        self.Max_det = int(root.getvar('ent_Max_det'))
        self.line_thickness = int(root.getvar('ent_line_thickness'))
        self.hide_labels = True if str(root.getvar('chk_hidelabels')) == '1' else False
        self.hide_conf = True if str(root.getvar('chk_hideconf')) == '1' else False

class Parameters_Light(LightAlgo.AI_Algo):
    def __init__(self, enable = True, ModelPath = 'D:\Machine Learning\工研院產業新尖兵\專題\結案報告\Demo\Light\Model/svm_model_3.sav'):
        self.enable = enable
        self.ModelPath = ModelPath
        self.ai_Algo = LightAlgo.AI_Algo()
        self.loadModel()

    def Update_Parameters(self):
        self.enable = True if root.getvar('chk_btnLightEnable') == '1' else False

    def loadModel(self):
        if self.enable:
            try:
                self.ai_Algo.loadModel(self.ModelPath)
            except Exception as ex:
                print(ex)

    def predict(self, listImg: list, probability=True, dimension=(20, 20), BGR=False):
        return self.ai_Algo.predict2(listImg, probability, dimension, BGR)

class Parameters_SignDL(SignAlgo_DL.AI_Algo):
    def __init__(self, enable = True, ModelPath = 'D:\Machine Learning\工研院產業新尖兵\專題\結案報告\Demo\Sign\DL\h5\VGG19_combinedata_Kfold.h5'):
        self.enable = enable
        self.ModelPath = ModelPath
        self.ai_Algo = SignAlgo_DL.AI_Algo()
        self.loadModel()

    def Update_Parameters(self):
        self.enable = True if root.getvar('chk_btnSignDLEnable') == '1' else False

    def loadModel(self):
        if self.enable:
            try:
                self.ai_Algo.loadModel(self.ModelPath)
            except Exception as ex:
                print(ex)

    def predict(self, listImg: list, probability=True, dimension=(32, 32), BGR=False):
        return self.ai_Algo.predict2(listImg, dimension, BGR)


class Parameters_SignML(SignAlgo_ML.AI_Algo):
    def __init__(self, enable = False, ModelPath = 'D:\Machine Learning\工研院產業新尖兵\專題\結案報告\Demo\Sign\ML\Model\RGB_Logist_Model.sav'):
        self.enable = enable
        self.ModelPath = ModelPath
        self.ai_Algo = SignAlgo_ML.AI_Algo()
        self.loadModel()

    def Update_Parameters(self):
        self.enable = True if root.getvar('chk_btnSignMLEnable') == '1' else False
        self.loadModel()

    def loadModel(self):
        if self.enable:
            try:
                self.ai_Algo.loadModel(self.ModelPath)
            except Exception as ex:
                print(ex)

    def predict(self, listImg: list, probability=True, dimension=(32, 32), BGR=False):
        return self.ai_Algo.predict2(listImg, dimension, BGR)


ParametersProject = Parameters_Project()
# YOLO v5
ParametersYOLO = Parameters_YOLO()
# 呼叫分類演算法
ParametersLight = Parameters_Light()
ParametersSignDL = Parameters_SignDL()
ParametersSignML = Parameters_SignML()


@torch.no_grad()
def run(weights='yolov5s.pt',  # model.pt path(s)
        source='data/images',  # file/dir/URL/glob, 0 for webcam
        imgsz=640,  # inference size (pixels)
        conf_thres=0.25,  # confidence threshold
        iou_thres=0.45,  # NMS IOU threshold
        max_det=1000,  # maximum detections per image
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        view_img=False,  # show results
        save_txt=False,  # save results to *.txt
        save_conf=False,  # save confidences in --save-txt labels
        save_crop=False,  # save cropped prediction boxes
        nosave=False,  # do not save images/videos
        classes=None,  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms=False,  # class-agnostic NMS
        augment=False,  # augmented inference
        visualize=False,  # visualize features
        update=False,  # update all models
        project='runs/detect',  # save results to project/name
        name='exp',  # save results to project/name
        exist_ok=False,  # existing project/name ok, do not increment
        line_thickness=3,  # bounding box thickness (pixels)
        hide_labels=False,  # hide labels
        hide_conf=False,  # hide confidences
        half=False,  # use FP16 half-precision inference
        ):

    global ParametersYOLO, ParametersLight, B_StopRun # Jeff Revised!
    # ParametersYOLO.save_img = not nosave and not source.endswith('.txt')  # save inference images
    webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))

    # Directories
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Initialize
    set_logging()
    device = select_device(device)
    half &= device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check image size # Jeff Revised! 不加會有錯誤: RuntimeError: Sizes of tensors must match except in dimension 2. Got 43 and 44 (The offending index is 0)
    names = model.module.names if hasattr(model, 'module') else model.names  # get class names
    if half:
        model.half()  # to FP16

    # Second-stage classifier
    classify = False
    if classify:
        modelc = load_classifier(name='resnet50', n=2)  # initialize
        modelc.load_state_dict(torch.load('resnet50.pt', map_location=device)['model']).to(device).eval()

    # Dataloader
    if webcam:
        view_img = check_imshow()
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz, stride=stride)
        bs = len(dataset)  # batch_size
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride)
        bs = 1  # batch_size
    vid_path, vid_writer = [None] * bs, [None] * bs

    if view_img: # Jeff Revised!
        # cv2.namedWindow('frame')
        pass

    # Run inference
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
    t0 = time.time()
    for path, img, im0s, vid_cap in dataset: # for each image
        if B_StopRun: # Jeff Revised!
            B_StopRun = False
            dataset.StopRun()
            break

        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        t1 = time_synchronized()
        pred = model(img,
                     augment=augment,
                     visualize=increment_path(save_dir / 'features', mkdir=True) if visualize else False)[0]

        # Apply NMS
        pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
        t2 = time_synchronized()

        # Apply Classifier
        if classify:
            pred = apply_classifier(pred, modelc, img, im0s)

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            if webcam:  # batch_size >= 1
                p, s, im0, frame = path[i], f'{i}: ', im0s[i].copy(), dataset.count
            else:
                p, s, im0, frame = path, '', im0s.copy(), getattr(dataset, 'frame', 0)

            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # img.jpg
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # img.txt
            s += '%gx%g ' % img.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            # imc = im0.copy() if save_crop else im0  # for save_crop
            imc = im0.copy() # Jeff Revised!

            # Jeff Revised!
            resetDisplay()
            ListInfo_Light = []
            ListInfo_Sign = []

            if len(det):

                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # Write results
                for *xyxy, conf, cls in reversed(det): # for each detected object
                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
                        with open(txt_path + '.txt', 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')

                    if ParametersYOLO.save_img or save_crop or view_img:  # Add bbox to image
                        c = int(cls)  # integer class
                        label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
                        plot_one_box(xyxy, im0, label=label, color=colors(c, True), line_thickness=line_thickness)
                        if save_crop:
                            save_one_box(xyxy, imc, file=save_dir / 'crops' / names[c] / f'{p.stem}.jpg', BGR=True)

                        # Jeff Revised!
                        if ParametersLight.enable:
                            if str(names[c]) == 'light':
                                ListInfo_Light.append([save_one_box(xyxy, imc, BGR=True, save=False), get_bbox_area(xyxy), conf])
                        if ParametersSignDL.enable or ParametersSignML.enable:
                            if str(names[c]) == 'sign':
                                ListInfo_Sign.append([save_one_box(xyxy, imc, BGR=True, save=False), get_bbox_area(xyxy), conf])

            # Print time (inference + NMS)
            infoText = f'{s}Done. ({t2 - t1:.3f}s)'
            print(infoText)
            scrolledtext_info.insert(tk.INSERT, infoText + '\n\n')

            #region GUI update + 分類演算法(紅綠燈+交通號誌) (Jeff Revised!)

            # 分類演算法(紅綠燈)
            if ParametersLight.enable:
                t3 = time_synchronized()
                if len(ListInfo_Light) > 0:
                    ListInfo_Light = sorted(ListInfo_Light, key = lambda s: s[1], reverse = True)
                    ListImage_orig_Light = []
                    for info in ListInfo_Light:
                        ListImage_orig_Light.append(info[0])         
                    info_light = ParametersLight.predict(ListImage_orig_Light, probability=True, BGR=True)
                    ListImage_stan_Light = [ParametersLight.DictImage_stan_Light[info[0]] for info in info_light]
                    LightBlocks.updateDisplay(ListImage_orig_Light, ListImage_stan_Light, info_light)
                t4 = time_synchronized()
                infoText = f'Traffic Light Algo Done. ({t4 - t3:.3f}s)'
                print(infoText)
                scrolledtext_info.insert(tk.INSERT, infoText + '\n\n')

            # 分類演算法(交通號誌)
            if ParametersSignDL.enable or ParametersSignML.enable:
                if ParametersSignDL.enable:
                    parameters_sign = ParametersSignDL
                else:
                    parameters_sign = ParametersSignML
                    
                t3 = time_synchronized()
                if len(ListInfo_Sign) > 0:
                    ListInfo_Sign = sorted(ListInfo_Sign, key = lambda s: s[1], reverse = True)
                    ListImage_orig_Sign = []
                    for info in ListInfo_Sign:
                        ListImage_orig_Sign.append(info[0])
                    info_sign = parameters_sign.predict(ListImage_orig_Sign, probability=True, BGR=True)
                    ListImage_stan_Sign = [ParametersSignDL.DictImage_stan_Sign[info[0]] for info in info_sign]

                    # Group
                    ListImage_orig_Sign_M = []
                    ListImage_orig_Sign_P= []
                    ListImage_orig_Sign_W = []
                    ListImage_stan_Sign_M = []
                    ListImage_stan_Sign_P = []
                    ListImage_stan_Sign_W = []
                    info_sign_M = []
                    info_sign_P = []
                    info_sign_W = []
                    for i in range(len(ListImage_orig_Sign)):
                        orig_img = ListImage_orig_Sign[i]
                        stan_img = ListImage_stan_Sign[i]
                        info_ = info_sign[i]
                        group = ParametersSignDL.Dict_name_2_Group[info_sign[i][0]]
                        if group == 'M':
                            ListImage_orig_Sign_M.append(orig_img)
                            ListImage_stan_Sign_M.append(stan_img)
                            info_sign_M.append(info_)
                        elif group == 'P':
                            ListImage_orig_Sign_P.append(orig_img)
                            ListImage_stan_Sign_P.append(stan_img)
                            info_sign_P.append(info_)
                        elif group == 'W':
                            ListImage_orig_Sign_W.append(orig_img)
                            ListImage_stan_Sign_W.append(stan_img)
                            info_sign_W.append(info_)

                    # SignBlocks1.updateDisplay(ListImage_orig_Sign, ListImage_stan_Sign, info_sign)
                    SignBlocks1.updateDisplay(ListImage_orig_Sign_M, ListImage_stan_Sign_M, info_sign_M)
                    SignBlocks2.updateDisplay(ListImage_orig_Sign_P, ListImage_stan_Sign_P, info_sign_P)
                    SignBlocks3.updateDisplay(ListImage_orig_Sign_W, ListImage_stan_Sign_W, info_sign_W)

                t4 = time_synchronized()
                infoText = f'Traffic Sign Algo Done. ({t4 - t3:.3f}s)'
                print(infoText)
                scrolledtext_info.insert(tk.INSERT, infoText + '\n\n')

            #endregion

            # Stream results
            if view_img:
                # cv2.imshow(str(p), im0)
                # cv2.imshow('frame', im0)
                # Use PIL (Pillow) to convert the NumPy ndarray to a PhotoImage
                im0_resize = cv2.resize(im0, (1000, 800))

                # Show FPS
                t5 = time_synchronized()
                fps = f'{(1.0 / (t5 - t1)):.1f} fps'
                cv2.putText(im0_resize, fps, (850, 60), 0, 1, [0, 0, 255], thickness=2, lineType=cv2.LINE_AA)

                global photo
                photo = PIL.ImageTk.PhotoImage(image = PIL.Image.fromarray(im0_resize[:, :, ::-1]))
                # Add a PhotoImage to the Canvas
                ca.create_image(0, 0, image=photo, anchor=tkinter.NW)
                root.update()
                cv2.waitKey(ParametersYOLO.WaitTime) # WaitTime millisecond # Jeff Revised!

            #region Save results (image with detections)

            if ParametersYOLO.save_img:
                if dataset.mode == 'image':
                    cv2.imwrite(save_path, im0)
                else: # 'video' or 'stream'
                    if vid_path[i] != save_path:  # new video
                        vid_path[i] = save_path
                        if isinstance(vid_writer[i], cv2.VideoWriter):
                            vid_writer[i].release()  # release previous video writer
                        if vid_cap:  # video
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:  # stream
                            fps, w, h = 30, im0.shape[1], im0.shape[0]
                            save_path += '.mp4'
                        vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                    vid_writer[i].write(im0)

            #endregion

    if save_txt or ParametersYOLO.save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        infoText = f"Results saved to {save_dir}{s}"
        print(infoText)
        scrolledtext_info.insert(tk.INSERT, infoText + '\n\n')

    if update:
        strip_optimizer(weights)  # update model (to fix SourceChangeWarning)

    if view_img: # Jeff Revised!
        # cv2.destroyAllWindows()
        pass

    infoText = f'Done. ({time.time() - t0:.3f}s)'
    print(infoText)
    scrolledtext_info.insert(tk.INSERT, infoText + '\n\n' + '-' * 50 + '\n\n')


def parse_opt():
    # Jeff Revised!
    global ParametersYOLO
    # ParametersYOLO.weights_path = 'yolov5x.pt'
    # ParametersYOLO.weights_path = r'D:\Machine Learning\工研院產業新尖兵\專題\yolo\yolov5\best.pt'
    # ParametersYOLO.source_path = './data/testing_images'
    # ParametersYOLO.source_path = 'D:\Machine Learning\工研院產業新尖兵\專題\Self_Driving_Cars\data\TrafficLight'
    # ParametersYOLO.source_path = 'D:\Machine Learning\工研院產業新尖兵\專題\Self_Driving_Cars\data\TrafficLight Video'
    # ParametersYOLO.source_path = 'D:\Machine Learning\工研院產業新尖兵\專題\Video'
    # ParametersYOLO.source_path = r'D:\Machine Learning\工研院產業新尖兵\專題\yolo\yolov5\plate\images\valid'
    # ParametersYOLO.source_path = 'D:/Machine Learning/工研院產業新尖兵/專題/yolo/yolov5/plate/images/valid'
    # ParametersYOLO.source_path = 'http://192.168.0.101:4747/video' # DroidCam
    # ParametersYOLO.source_path = 'https://www.youtube.com/watch?v=nHUbCoPS-Xc&ab_channel=PinoSu' # Youtube
    # ParametersYOLO.save_img = True if root.getvar('chk_btnSaved') == 1 else False
    ParametersYOLO.save_img = True if str(root.getvar('chk_btnSaved')) == '1' else False
    ParametersYOLO.save_crop = True if str(root.getvar('chk_btnsavebboxImg')) == '1' else False
    # print(root.getvar('chk_btnSaved'))
    view_img = True

    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default=ParametersYOLO.weights_path, help='model.pt path(s)') # Jeff Revised!
    parser.add_argument('--source', type=str, default=ParametersYOLO.source_path, help='file/dir/URL/glob, 0 for webcam') # Jeff Revised!
    parser.add_argument('--imgsz', '--img', '--img-size', type=int, default=ParametersYOLO.Imgsz, help='inference size (pixels)') # Jeff Revised!
    parser.add_argument('--conf-thres', type=float, default=ParametersYOLO.Conf_thres, help='confidence threshold') # Jeff Revised!
    parser.add_argument('--iou-thres', type=float, default=ParametersYOLO.Iou_thres, help='NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=ParametersYOLO.Max_det, help='maximum detections per image')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    # parser.add_argument('--view-img', action='store_true', help='show results')
    parser.add_argument('--view-img', type=bool, default=view_img, help='show results') # Jeff Revised!
    parser.add_argument('--save-txt', type=bool, default=ParametersYOLO.save_txt, help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    # parser.add_argument('--save-crop', action='store_true', help='save cropped prediction boxes') # Jeff Revised!
    parser.add_argument('--save-crop', type=bool, default=ParametersYOLO.save_crop, help='save cropped prediction boxes') # Jeff Revised!
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--visualize', action='store_true', help='visualize features')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default='runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--line-thickness', type=int, default=ParametersYOLO.line_thickness, help='bounding box thickness (pixels)')
    parser.add_argument('--hide-labels', type=bool, default=ParametersYOLO.hide_labels, help='hide labels')
    parser.add_argument('--hide-conf', type=bool, default=ParametersYOLO.hide_conf, help='hide confidences')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    opt = parser.parse_args()
    return opt

def main(opt):
    infoText = colorstr('detect: ') + ', '.join(f'{k}={v}' for k, v in vars(opt).items())
    print(infoText)
    scrolledtext_info.insert(tk.INSERT, infoText + '\n\n\n')
    check_requirements(exclude=('tensorboard', 'thop'))
    run(**vars(opt))


#region GUI

# import tkinter as tk
from tkinter import *
from tkinter import ttk
from tkinter import filedialog
import easygui as gui
import glob
import os

root = Tk()
root.geometry('%dx%d+%d+%d' % (1650, 1000, 30, 5))
root.config(bg="#da0")
##root.config(bg="khaki")
# root.title("AI專題神人鯊鯊團")
root.title("自駕車影像辨識系統 (Image Identification for Self-Driving Cars)")
root.iconbitmap('GUI Image/鯊鯊4-2.ico')

#region Style Setting

s = ttk.Style()
s.configure('my.TButton', font=('Helvetica', 14,'bold'), foreground='green')
s.configure('my2.TButton', font=('Helvetica', 12), background='#7AC5CD', foreground='magenta')
s.configure('my3.TButton', font=('Helvetica', 14,'bold'), foreground='red')
s.configure('my.TLabel', font=('Arial', 18,'bold','italic'), background='blue', foreground='white', anchor = 'center')
s.configure('my2.TLabel', font=('Arial', 14,'bold','italic'), background='purple', foreground='yellow', anchor = 'center')
s.configure('my3.TLabel', font=('Arial', 14,'bold','italic'), background='gray', foreground='white', anchor = 'center')
s.configure('my4.TLabel', font=('Arial', 12,'bold','italic'), background='red', foreground='white', anchor = 'center')
s.configure('param.TLabel', font=('Arial', 14,'bold'), background='orange', foreground='white')
s.configure('param.TCheckbutton', font=('Arial', 14,'bold'), background='orange', foreground='white')
s.configure('param.TEntry', font=('Arial', 14,'bold'), background='white', foreground='#da0')
s.configure('TNotebook.Tab', font=('URW Gothic L','14','bold'))
s.configure('my.TRadiobutton', font=('Arial','10','bold'), foreground='green')

#endregion

source_dir_img = ''
source_dir_img2 = ''
source_dir_img3 = ''

def btn_Browse(e):
    global ParametersYOLO
    if e.widget is btn_brow_Model: # Model
        # s = gui.fileopenbox("MSG","TITLE", default = "*.pt")
        s = gui.fileopenbox("MSG","TITLE", default = "D:\Machine Learning\工研院產業新尖兵\專題\結案報告\Demo\YOLO\Model\*.pt")
        if s is not None:
            root.setvar('ent_Model', s)
            ParametersYOLO.weights_path = s
        
    elif e.widget is btn_brow_Test: # Test
        s = filedialog.askdirectory(initialdir="D:\Machine Learning\工研院產業新尖兵\專題\結案報告\Demo\Test Image", title='Please select a directory')
        if s != '':
            root.setvar('ent_Test', s)
            ParametersYOLO.source_path = s

    elif e.widget is btn_brow_Image: # Image
        global source_dir_img
        s = filedialog.askdirectory(initialdir="D:\Machine Learning\工研院產業新尖兵\專題\結案報告\Demo\Test Image", title='Please select a directory')
        if s != '':
            root.setvar('ent_Image', s)
            source_dir_img = s
        
        list_imgName = []
        # for filename in glob.glob(s + '/*.jpg'):
        for filename in glob.glob(os.path.join(s, '*.jpg')):
            list_imgName.append(filename.split('\\')[-1]) # Windows系統
            # list_imgName.append(filename.split('/')[-1]) # Mac系統
        cb_Image.config(values=list_imgName)

    elif e.widget is btn_brow_Image2: # Image 2
        global source_dir_img2
        s = filedialog.askdirectory(initialdir="D:\Machine Learning\工研院產業新尖兵\專題\結案報告\Demo\Test Image", title='Please select a directory')
        if s != '':
            root.setvar('ent_Image2', s)
            source_dir_img2 = s
        
        list_imgName = []
        # for filename in glob.glob(s + '/*.jpg'):
        for filename in glob.glob(os.path.join(s, '*.jpg')):
            list_imgName.append(filename.split('\\')[-1]) # Windows系統
            # list_imgName.append(filename.split('/')[-1]) # Mac系統
        cb_Image2.config(values=list_imgName)

    elif e.widget is btn_brow_Image3: # Image 3
        global source_dir_img3
        s = filedialog.askdirectory(initialdir="D:\Machine Learning\工研院產業新尖兵\專題\結案報告\Demo\Test Image", title='Please select a directory')
        if s != '':
            root.setvar('ent_Image3', s)
            source_dir_img3 = s
        
        list_imgName = []
        # for filename in glob.glob(s + '/*.jpg'):
        for filename in glob.glob(os.path.join(s, '*.jpg')):
            list_imgName.append(filename.split('\\')[-1]) # Windows系統
            # list_imgName.append(filename.split('/')[-1]) # Mac系統
        cb_Image3.config(values=list_imgName)

def execute_mode_changed():
    if v_exe.get() == 1: # batch
        root.setvar('ent_Test', '')
    elif v_exe.get() == 2: # single
        root.setvar('ent_Test', '')
    elif v_exe.get() == 3: # Youtube
        root.setvar('ent_Test', ParametersProject.link_Youtube)
    elif v_exe.get() == 4: # DroidCam
        root.setvar('ent_Test', ParametersProject.link_DroidCam)
    elif v_exe.get() == 5: # WebCam
        root.setvar('ent_Test', '')

import tkinter as tk
v_exe = tk.IntVar()
v_exe.set(1)
#rbt_batch = ttk.Radiobutton(root, text = 'Batch', variable=v_exe, value=1, command=execute_mode_changed, style = 'my.TRadiobutton')
#rbt_batch.place(x=160, y=135, width=60, height=30)
photo_icon_batch = PhotoImage(file = "GUI Image/folder32.png")
rbt_batch = ttk.Radiobutton(root, image = photo_icon_batch, variable=v_exe, value=1, command=execute_mode_changed, style = 'my.TRadiobutton')
rbt_batch.place(x=160, y=130, width=60, height=55)
#rbt_single = ttk.Radiobutton(root, text = 'Single', variable=v_exe, value=2, command=execute_mode_changed, style = 'my.TRadiobutton')
#rbt_single.place(x=230, y=135, width=65, height=30)
photo_icon_single = PhotoImage(file = "GUI Image/image.png")
rbt_single = ttk.Radiobutton(root, image = photo_icon_single, variable=v_exe, value=2, command=execute_mode_changed, style = 'my.TRadiobutton')
rbt_single.place(x=230, y=130, width=60, height=55)
#rbt_Youtube = ttk.Radiobutton(root, text = 'Youtube', variable=v_exe, value=3, command=execute_mode_changed, style = 'my.TRadiobutton')
#rbt_Youtube.place(x=305, y=135, width=75, height=30)
photo_icon_Youtube = PhotoImage(file = "GUI Image/youtube.png")
rbt_Youtube = ttk.Radiobutton(root, image = photo_icon_Youtube, variable=v_exe, value=3, command=execute_mode_changed, style = 'my.TRadiobutton')
rbt_Youtube.place(x=300, y=130, width=60, height=55)
#rbt_camera = ttk.Radiobutton(root, text = 'DroidCam', variable=v_exe, value=4, command=execute_mode_changed, style = 'my.TRadiobutton')
#rbt_camera.place(x=370, y=130, width=90, height=35)
photo_icon_camera = PhotoImage(file = "GUI Image/DroidCam2_50.png")
rbt_camera = ttk.Radiobutton(root, image = photo_icon_camera, variable=v_exe, value=4, command=execute_mode_changed, style = 'my.TRadiobutton')
rbt_camera.place(x=370, y=130, width=80, height=55)
#rbt_WebCam = ttk.Radiobutton(root, text = 'WebCam', variable=v_exe, value=5, command=execute_mode_changed, style = 'my.TRadiobutton')
#rbt_WebCam.place(x=500, y=135, width=80, height=30)
#rbt_WebCam = ttk.Radiobutton(root, text = 'WebCam', variable=v_exe, value=5, command=execute_mode_changed, style = 'my.TRadiobutton')
#rbt_WebCam.place(x=460, y=130, width=80, height=55)
photo_icon_WebCam = PhotoImage(file = "GUI Image/WebCam.png")
rbt_WebCam = ttk.Radiobutton(root, image = photo_icon_WebCam, variable=v_exe, value=5, command=execute_mode_changed, style = 'my.TRadiobutton')
rbt_WebCam.place(x=460, y=130, width=60, height=55)
# print(v_exe.get())

def btn_Execute_press(e):
    btn_Execute.config(style='my3.TButton')
    global ParametersYOLO, source_dir_img
    if v_exe.get() == 1: # batch
        dir = root.getvar('ent_Test')
        if os.path.isdir(dir):
            ParametersYOLO.source_path = dir
    elif v_exe.get() == 2: # single
        try:
            path = source_dir_img + '/' + root.getvar('cb_Image')
            if os.path.exists(path):
                ParametersYOLO.source_path = path
        except:
            btn_Execute.config(style='my.TButton')
            return
    elif v_exe.get() == 3: # Youtube
        ParametersProject.link_Youtube = root.getvar('ent_Test')
        ParametersYOLO.source_path = ParametersProject.link_Youtube
    elif v_exe.get() == 4: # DroidCam
        ParametersProject.link_DroidCam = root.getvar('ent_Test')
        ParametersYOLO.source_path = ParametersProject.link_DroidCam
    elif v_exe.get() == 5: # WebCam
        ParametersYOLO.source_path = '0'
    opt = parse_opt()
    main(opt)
    gc.collect() # or gc.collect(2)
    btn_Execute.config(style='my.TButton')

# lb_Model = Label(root, text="Model", fg="white", bg="blue")
lb_Model = ttk.Label(root, text="Model", style='my.TLabel')
lb_Model.place(x=10, y=10, width=100, height=30)
# lb_Model.config(font="Arial 22 bold italic")

lb_Test = ttk.Label(root, text="Test", style='my.TLabel')
lb_Test.place(x=10, y=50, width=100, height=30)

ent_Model = ttk.Entry(root, width=60, textvariable= 'ent_Model')
ent_Model.place(x=130, y=10, width = 550, height=30)
root.setvar('ent_Model', 'Model path')

ent_Test = ttk.Entry(root, width=60, textvariable= 'ent_Test')
ent_Test.place(x=130, y=50, width = 550, height=30)
root.setvar('ent_Test', 'Test image directory')

btn_brow_Model = ttk.Button(root, text="Browse", style='my2.TButton')
btn_brow_Model.place(x=700, y=10, width = 80, height=30)
btn_brow_Model.bind('<ButtonRelease-1>', btn_Browse)

btn_brow_Test = ttk.Button(root, text="Browse", style='my2.TButton')
btn_brow_Test.place(x=700, y=50, width = 80, height=30)
btn_brow_Test.bind('<ButtonRelease-1>', btn_Browse)

photo_icon_Execute = PhotoImage(file = "GUI Image/execution.png")
btn_Execute = ttk.Button(root, text=" Execute", style='my.TButton', image = photo_icon_Execute, compound = LEFT)
# btn_Execute = Button(root, text=" Execute", fg='green', image = photo_icon_Execute, compound = LEFT)
btn_Execute.place(x=20, y=130, width=130, height=40)
btn_Execute.bind('<ButtonRelease-1>', btn_Execute_press)

def btn_player_press(e):
    if e.widget is btn_Stop: # Stop
        global B_StopRun
        B_StopRun = True

photo_icon_Stop = PhotoImage(file = "GUI Image/stop.png")
btn_Stop = ttk.Button(root, image = photo_icon_Stop)
btn_Stop.place(x=540, y=135, width=40, height=40)
btn_Stop.bind('<ButtonRelease-1>', btn_player_press)

ca = Canvas(root, width=1000, height=800, bg='white')
ca.place(x=10, y=190, width = 1000, height = 800)

from PIL import Image, ImageTk
import PIL
import numpy as np
# import tkinter
# photo_1 = ImageTk.PhotoImage(Image.open("plates.jpg"))
# ca.create_image(0, 0, image = photo_1)
import cv2
# Use PIL (Pillow) to convert the NumPy ndarray to a PhotoImage
# img = cv2.imread('plates.jpg')
# print(type(img))
# photo = ImageTk.PhotoImage(image = Image.fromarray(img))
# Add a PhotoImage to the Canvas
# ca.create_image(0, 0, image=photo, anchor=tkinter.NW)
# ca.create_image(0, 0, image=photo, anchor=NW)
# ca.create_image(0, 0, image=photo, anchor='nw')

#region 鯊鯊icon

# w_icon = 200
# h_icon = 150
# size_icon = (w_icon, h_icon)

# ca_icon1 = Canvas(root, width=w_icon, height=h_icon, bg='white')
# ca_icon1.place(x=1290, y=10, width = w_icon, height = h_icon)
# # img = cv2.imdecode(np.fromfile('yolov5/GUI Image/鯊鯊1.png', dtype=np.uint8), 1)
# img = cv2.imdecode(np.fromfile('GUI Image/鯊鯊1.png', dtype=np.uint8), 1)
# img = cv2.resize(img, size_icon)
# photo_icon1 = ImageTk.PhotoImage(image = Image.fromarray(img[:, :, ::-1]))
# ca_icon1.create_image(0, 0, image=photo_icon1, anchor=tkinter.NW)

# ca_icon2 = Canvas(root, width=w_icon, height=h_icon, bg='white')
# ca_icon2.place(x=1290, y=170, width = w_icon, height = h_icon)
# # img = cv2.imdecode(np.fromfile('yolov5/GUI Image/鯊鯊2.jpg', dtype=np.uint8), 1)
# img = cv2.imdecode(np.fromfile('GUI Image/鯊鯊2.jpg', dtype=np.uint8), 1)
# img = cv2.resize(img, size_icon)
# photo_icon2 = ImageTk.PhotoImage(image = Image.fromarray(img[:, :, ::-1]))
# ca_icon2.create_image(0, 0, image=photo_icon2, anchor=tkinter.NW)

# ca_icon3 = Canvas(root, width=w_icon, height=h_icon, bg='white')
# ca_icon3.place(x=1290, y=330, width = w_icon, height = h_icon)
# # img = cv2.imdecode(np.fromfile('yolov5/GUI Image/鯊鯊3.jpg', dtype=np.uint8), 1)
# img = cv2.imdecode(np.fromfile('GUI Image/鯊鯊3.jpg', dtype=np.uint8), 1)
# img = cv2.resize(img, size_icon)
# photo_icon3 = ImageTk.PhotoImage(image = Image.fromarray(img[:, :, ::-1]))
# ca_icon3.create_image(0, 0, image=photo_icon3, anchor=tkinter.NW)

# ca_icon4 = Canvas(root, width=w_icon, height=h_icon, bg='white')
# ca_icon4.place(x=1290, y=490, width = w_icon, height = h_icon)
# # img = cv2.imdecode(np.fromfile('yolov5/GUI Image/鯊鯊4.png', dtype=np.uint8), 1)
# img = cv2.imdecode(np.fromfile('GUI Image/鯊鯊4.png', dtype=np.uint8), 1)
# img = cv2.resize(img, size_icon)
# photo_icon4 = ImageTk.PhotoImage(image = Image.fromarray(img[:, :, ::-1]))
# ca_icon4.create_image(0, 0, image=photo_icon4, anchor=tkinter.NW)

#endregion

chk_btn_saveImg =ttk.Checkbutton(root, text="Save Image", takefocus = 0, textvariable = 'chk_btn_saveImg', variable= 'chk_btnSaved')
chk_btn_saveImg.place(x=800, y=50, width = 95, height = 30)
root.setvar('chk_btn_saveImg', 'Save Image')
root.setvar('chk_btnSaved', False)
print(type(root.getvar('chk_btnSaved')))
#root.getvar('chk_btnSaved')

B_synImage = False
def chk_btn_synImage_changed():
    print(root.getvar('chk_btnsynImage'))
    print(type(root.getvar('chk_btnsynImage')))
    global B_synImage
    B_synImage = True if root.getvar('chk_btnsynImage') == '1' else False
    print(B_synImage)

chk_btn_synImage =ttk.Checkbutton(root, text="Syn Image", takefocus = 0, textvariable = 'chk_btn_synImage', variable= 'chk_btnsynImage', command=chk_btn_synImage_changed)
chk_btn_synImage.place(x=800, y=130, width = 95, height = 30)
root.setvar('chk_btn_synImage', 'Syn Image')
root.setvar('chk_btnsynImage', False)

btn_brow_Image = ttk.Button(root, text="Browse", style='my2.TButton')
btn_brow_Image.place(x=700, y=90, width = 80, height=30)
btn_brow_Image.bind('<ButtonRelease-1>', btn_Browse)

# lb_Image = Label(root, text="Image", fg="white", bg="blue")
lb_Image = ttk.Label(root, text="Image", style='my.TLabel')
lb_Image.place(x=10, y=90, width=100, height=30)
# lb_Image.config(font="Arial 22 bold italic")

ent_Image = ttk.Entry(root, width=60, textvariable= 'ent_Image')
ent_Image.place(x=130, y=90, width = 550, height=30)
root.setvar('ent_Image', 'Inspect image directory')

#region Load & Inspect image

def cb_Image_Update(e = None):
    global photo_cb_Image
    #print(e)
    print(root.getvar('cb_Image'))
    path = source_dir_img + '/' + root.getvar('cb_Image')
    #print(path)
    img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), 1)
    # Resize
    img = cv2.resize(img, (1000, 800))
    photo_cb_Image = PIL.ImageTk.PhotoImage(image = PIL.Image.fromarray(img[:, :, ::-1]))
    # ca.create_image(0, 0, image=photo_cb_Image, anchor=tkinter.NW)
    ca.create_image(0, 0, image=photo_cb_Image, anchor='nw')
    # root.update()

#cb_Image = ttk.Combobox(root,font="Arial 12", height=10, width=10, textvariable= 'cb_Image', command=cb_Image_Update)
cb_Image = ttk.Combobox(root,font="Arial 12", height=10, width=10, textvariable= 'cb_Image')
cb_Image.place(x=800, y=90, height=30, width = 100)
cb_Image.config(values=[])
cb_Image.bind("<<ComboboxSelected>>", cb_Image_Update)

#photo_icon_next = PhotoImage(file = "GUI Image/next.png")
img = Image.open("GUI Image/next.png")
img = img.resize((32, 32), Image.ANTIALIAS)
photo_icon_next =  ImageTk.PhotoImage(img)
btn_Next = ttk.Button(root, text="Next", image = photo_icon_next)
# btn_Next.pack(side = TOP)
btn_Next.place(x=960, y=85, height=40)

img = Image.open("GUI Image/prev.png")
img = img.resize((32, 32), Image.ANTIALIAS)
photo_icon_prev =  ImageTk.PhotoImage(img)
btn_Prev = ttk.Button(root, text="Prev", image = photo_icon_prev)
# btn_Prev.pack(side = TOP)
btn_Prev.place(x=910, y=85, height=40)

def btn_NextPrevImg(e):
    # print('-' * 10)
    # print(root.getvar('cb_Image'))
    # print(cb_Image.get())
    index_now = cb_Image.current()
    # print(index_now)
    bNext = False
    if e.widget is btn_Next:
        bNext = True
        index_max = len(cb_Image['values']) - 1
        if index_now >= index_max:
            return
        index_now += 1
    elif e.widget is btn_Prev:
        if index_now <= 0:
            return
        index_now -= 1
    cb_Image.current(index_now)
    cb_Image_Update() # 更新影像顯示

    global B_synImage
    if B_synImage:
        btn_NextPrevImg2(None, bNext)
        btn_NextPrevImg3(None, bNext)

btn_Next.bind('<ButtonRelease-1>', btn_NextPrevImg)
btn_Prev.bind('<ButtonRelease-1>', btn_NextPrevImg)

#endregion



tabPage_frame = Frame(root, bg="white", relief=SUNKEN)
tabPage_frame.place(x=1020, y=10, width = 620, height=980)

a_notebook = ttk.Notebook(tabPage_frame, width=200, height=200)
# a_notebook.config(font="Arial 25 bold italic")
# tab1 = Frame(a_notebook, bg='red')
tab1 = ttk.Frame(a_notebook)
a_notebook.add(tab1, text = 'Real-time Detection')
tab2 = ttk.Frame(a_notebook)
a_notebook.add(tab2, text = 'More Information')
tab3 = ttk.Frame(a_notebook)
a_notebook.add(tab3, text = 'Advanced Setting')
tab4 = ttk.Frame(a_notebook)
a_notebook.add(tab4, text = 'Others')
# a_notebook.pack(expand=True, fill=tk.BOTH)
a_notebook.pack(expand=True, fill='both')
# a_notebook.pack(expand=True)

#region 【Real-time Detection】

class singleBlock():
    def __init__(self, parentFrame_: ttk.Frame, x: int, y: int, w: int, h: int):
        """
        Constructor: Class initialization
        """
        # print('__init__')
        self.parentFrame = parentFrame_
        self.block = Frame(parentFrame_, bg="orange")
        self.block.place(x=x, y=y, width=w, height=h)
        self.canvas1 = Canvas(self.block, bg='white')
        self.canvas1.place(x=5, y=5, width=138, height=130)
        self.canvas2 = Canvas(self.block, bg='white')
        self.canvas2.place(x=148, y=5, width=138, height=130)
        self.labelConf = None
        self.labelInfo = Label(self.block, text='', bg="white")
        self.labelInfo.place(x=5, y=140, width=281, height=28)
        self.labelInfo.config(font="Arial 20 bold italic")
        # self.disable()
        # self.enable()

    def destroy(self):
        """
        Destroy block
        """
        self.block.destroy()

    def enable(self):
        for child in self.block.winfo_children():
            # print(child['state'])
            child.configure(state='normal')

    def disable(self):
        for child in self.block.winfo_children():
            # print(child['state'])
            child.configure(state='disable')

    def updateDisplay(self, image_orig, image_stan, label: str, confidence: float, info: str, color: str):
        # Resize
        img = cv2.resize(image_orig, (138, 130))
        self.photo_orig = PIL.ImageTk.PhotoImage(image = PIL.Image.fromarray(img[:, :, ::-1]))
        self.canvas1.create_image(0, 0, image=self.photo_orig, anchor='nw')
        if image_stan is not None:
            img = cv2.resize(image_stan, (138, 130))
            self.photo_stan = PIL.ImageTk.PhotoImage(image = PIL.Image.fromarray(img[:, :, ::-1]))
            self.canvas2.create_image(0, 0, image=self.photo_stan, anchor='nw')

        self.labelConf = ttk.Label(self.block, text=f'{label}:{confidence:.2f}', style='my4.TLabel')
        self.labelConf.place(x=70, y=3, width=160, height=25)

        # self.labelInfo = Label(self.block, text=info, fg=color, bg="white")
        self.labelInfo['text'] = info
        self.labelInfo['fg'] = color

    def reset(self):
        if self.labelConf is not None:
            self.labelConf.destroy()
            self.labelConf = None
        self.labelInfo['text'] = ''
        self.canvas1.delete(ALL)
        self.canvas2.delete(ALL)

class updateBlocks():
    def __init__(self, parentFrame_: ttk.Frame, scrollableFrame_: ttk.Frame, count: int):
        """
        Constructor: Class initialization
        """
        # print('__init__')
        self.parentFrame = parentFrame_
        self.scrollableFrame = scrollableFrame_
        self.listBlocks = []
        self.update(count)

    def update(self, count: int):
        """
        Update all blocks
        """
        self.clear()
        y = 30
        w = 290
        h = 170
        w_parentFrame = 595
        # Adjust width of parentFrame
        # x_max = 5 + (w + 5) * (count - 1) + w
        # if x_max > w_parentFrame - 5:
        #     self.parentFrame.config(width=x_max + 5)
        #     self.scrollableFrame.config(width=x_max + 5)
        # else:
        #     self.parentFrame.config(width=w_parentFrame)
        #     self.scrollableFrame.config(width=w_parentFrame)

        for i in range(count):
            self.listBlocks.append(singleBlock(self.parentFrame, x = 5 + (w + 5) * i, y = y, w = w, h = h))

    def clear(self):
        """
        Clear all blocks
        """
        for block in self.listBlocks:
            block.destroy()
        self.listBlocks = []

    def updateDisplay(self, listImage_orig_: list, listImage_stan_: list, info_: list):
        """
        Update display
        '''
        Parameters
        ----------
        info_ : list (n, information), where information: (label, confidence, description or warning, color)
        e.g. [('Red', 0.94, 'Stop!', 'red'), ('Green', 0.64, 'Go!', 'green'), ('Red', 0.74, 'Stop!', 'red')]
        
        Note: 順序依照面積由大到小
        """
        count = min(len(self.listBlocks), len(listImage_orig_))
        for i in range(count):
            if listImage_stan_ is None:
                self.listBlocks[i].updateDisplay(listImage_orig_[i], None, info_[i][0], info_[i][1], info_[i][2], info_[i][3])
            else:
                self.listBlocks[i].updateDisplay(listImage_orig_[i], listImage_stan_[i], info_[i][0], info_[i][1], info_[i][2], info_[i][3])

    def reset(self):
        for block in self.listBlocks:
            block.reset()


control_frame1 = Frame(tab1, bg="lightblue", relief=SUNKEN)
# control_frame1 = ttk.Frame(tab1, relief=SUNKEN)
control_frame1.place(x=10, y=10, width = 595, height=220)
scrollbar1 = ttk.Scrollbar(control_frame1, orient="horizontal")
scrollbar1.pack(side="bottom", fill="x")
lb_frame1 = ttk.Label(control_frame1, text="Traffic Light", style='my2.TLabel')
lb_frame1.place(x=2, y=2, width=180, height=25)
# lb_frame1.config(style='my3.TLabel')

control_frame2 = Frame(tab1, bg="lightblue", relief=SUNKEN)
control_frame2.place(x=10, y=240, width = 595, height=220)
scrollbar2 = ttk.Scrollbar(control_frame2, orient="horizontal")
scrollbar2.pack(side="bottom", fill="x")
lb_frame2 = ttk.Label(control_frame2, text="Sign: Mandatory", style='my2.TLabel')
lb_frame2.place(x=2, y=2, width=180, height=25)

control_frame3 = Frame(tab1, bg="lightblue", relief=SUNKEN)
control_frame3.place(x=10, y=470, width = 595, height=220)
scrollbar3 = ttk.Scrollbar(control_frame3, orient="horizontal")
scrollbar3.pack(side="bottom", fill="x")
lb_frame3 = ttk.Label(control_frame3, text="Sign: Prohibitory", style='my2.TLabel')
lb_frame3.place(x=2, y=2, width=180, height=25)

control_frame4 = Frame(tab1, bg="lightblue", relief=SUNKEN)
control_frame4.place(x=10, y=700, width = 595, height=220)
scrollbar4 = ttk.Scrollbar(control_frame4, orient="horizontal")
scrollbar4.pack(side="bottom", fill="x")
lb_frame4 = ttk.Label(control_frame4, text="Sign: Warning", style='my2.TLabel')
lb_frame4.place(x=2, y=2, width=180, height=25)

LightBlocks = updateBlocks(control_frame1, control_frame1, 2)
SignBlocks1 = updateBlocks(control_frame2, control_frame2, 2)
SignBlocks2 = updateBlocks(control_frame3, control_frame3, 2)
SignBlocks3 = updateBlocks(control_frame4, control_frame4, 2)

def resetDisplay():
    global LightBlocks, SignBlocks1, SignBlocks2, SignBlocks3
    LightBlocks.reset()
    SignBlocks1.reset()
    SignBlocks2.reset()
    SignBlocks3.reset()

# For Initial Display
LightBlocks.updateDisplay([cv2.imdecode(np.fromfile('GUI Image/red_light.jpg', dtype=np.uint8), 1), cv2.imdecode(np.fromfile('GUI Image/green_light.jpg', dtype=np.uint8), 1)],
                          [ParametersLight.DictImage_stan_Light['Red'], ParametersLight.DictImage_stan_Light['Green']],
                          [('Red', 0.94, 'Stop!', 'red'), ('Green', 0.98, 'Go!', 'green')])

SignBlocks1.updateDisplay([cv2.imdecode(np.fromfile('GUI Image/Sign/GUI/M/DFMsample.jpg', dtype=np.uint8), 1), cv2.imdecode(np.fromfile('GUI Image/Sign/GUI/M/KeepRightsample.jpg', dtype=np.uint8), 1)],
                          [ParametersSignDL.DictImage_stan_Sign['DirectForMotor'], ParametersSignDL.DictImage_stan_Sign['KeepRight']],
                          [('DirectForMotor', 0.95, 'FollowSign', 'blue'), ('KeepRight', 0.99, 'FollowSign', 'blue')])

SignBlocks2.updateDisplay([cv2.imdecode(np.fromfile('GUI Image/Sign/GUI/P/ALTRightTurn_sample.jpg', dtype=np.uint8), 1), cv2.imdecode(np.fromfile('GUI Image/Sign/GUI/P/SpdLimit30_sample.jpg', dtype=np.uint8), 1)],
                          [ParametersSignDL.DictImage_stan_Sign['ALTRightTurn'], ParametersSignDL.DictImage_stan_Sign['SpdLimit30']],
                          [('ALTRightTurn', 0.91, 'Be Aware', 'red'), ('SpdLimit30', 0.92, 'Be Aware', 'red')])

SignBlocks3.updateDisplay([cv2.imdecode(np.fromfile('GUI Image/Sign/GUI/W/Children_sample.jpg', dtype=np.uint8), 1), cv2.imdecode(np.fromfile('GUI Image/Sign/GUI/W/CurveToRight_sample.jpg', dtype=np.uint8), 1)],
                          [ParametersSignDL.DictImage_stan_Sign['Children'], ParametersSignDL.DictImage_stan_Sign['CurveToRight']],
                          [('Children', 0.91, 'Caution', 'orange'), ('CurveToRight', 0.92, 'Caution', 'orange')])

#endregion

#region 【More Information】

import tkinter.scrolledtext as tkst
scrolledtext_info = tkst.ScrolledText(master = tab2, wrap = tk.WORD)
scrolledtext_info.place(x=10, y=10, width = 600, height=930)
# scrolledtext_info.insert(tk.INSERT,
# """\
# Integer posuere erat a ante venenatis dapibus.
# Posuere velit aliquet.
# Aenean eu leo quam. Pellentesque ornare sem.
# Lacinia quam venenatis vestibulum.
# Nulla vitae elit libero, a pharetra augue.
# Cum sociis natoque penatibus et magnis dis.
# Parturient montes, nascetur ridiculus mus.
# """ * 30)

#endregion

#region 【Advanced Setting】

#region YOLO v5

control_frame_YOLO = Frame(tab3, bg="lightblue", relief=SUNKEN)
control_frame_YOLO.place(x=10, y=10, width = 595, height=435)

lb_param_YOLO = ttk.Label(control_frame_YOLO, text="YOLO v5", style='my2.TLabel')
lb_param_YOLO.place(x=2, y=2, width=180, height=25)

photo_icon_updateParam = PhotoImage(file = "GUI Image/settings.png")
btn_updateParam = ttk.Button(control_frame_YOLO, text=" Update Parameters", style='my.TButton', image = photo_icon_updateParam, compound = LEFT)
btn_updateParam.place(x=300, y=25, width=250, height=40)
btn_updateParam.bind('<ButtonRelease-1>', ParametersYOLO.Update_Parameters)

chk_btn_savebboxImg =ttk.Checkbutton(control_frame_YOLO, takefocus = 0, style = 'param.TCheckbutton', textvariable = 'chk_btn_savebboxImg', variable= 'chk_btnsavebboxImg')
chk_btn_savebboxImg.place(x=20, y=35, width = 190, height = 30)
root.setvar('chk_btn_savebboxImg', 'Save bbox Image')
root.setvar('chk_btnsavebboxImg', ParametersYOLO.save_crop)

chk_btn_save_txt = ttk.Checkbutton(control_frame_YOLO, takefocus = 0, style = 'param.TCheckbutton', textvariable = 'chk_btn_save_txt', variable= 'chk_btnsavetxt')
chk_btn_save_txt.place(x=20, y=75, width = 210, height = 25)
root.setvar('chk_btn_save_txt', 'Save results to *.txt')
root.setvar('chk_btnsavetxt', ParametersYOLO.save_txt)

lb_WaitTime = ttk.Label(control_frame_YOLO, text="WaitTime (ms)", style = 'param.TLabel')
lb_WaitTime.place(x=20, y=115, width=140, height=25)

ent_WaitTime = ttk.Entry(control_frame_YOLO, textvariable= 'ent_WaitTime', style = 'param.TEntry', justify = CENTER)
ent_WaitTime.place(x=170, y=115, width = 50, height=25)
root.setvar('ent_WaitTime', str(ParametersYOLO.WaitTime))

lb_Imgsz = ttk.Label(control_frame_YOLO, text="Image size (pixels)", style = 'param.TLabel')
lb_Imgsz.place(x=20, y=155, width=175, height=25)

ent_Imgsz = ttk.Entry(control_frame_YOLO, textvariable= 'ent_Imgsz', style = 'param.TEntry', justify = CENTER)
ent_Imgsz.place(x=205, y=155, width = 50, height=25)
root.setvar('ent_Imgsz', str(ParametersYOLO.Imgsz))

lb_Conf_thres = ttk.Label(control_frame_YOLO, text="Confidence Threshold", style = 'param.TLabel')
lb_Conf_thres.place(x=20, y=195, width=215, height=25)

ent_Conf_thres = ttk.Entry(control_frame_YOLO, textvariable= 'ent_Conf_thres', style = 'param.TEntry', justify = CENTER)
ent_Conf_thres.place(x=245, y=195, width = 50, height=25)
root.setvar('ent_Conf_thres', str(ParametersYOLO.Conf_thres))

lb_Iou_thres = ttk.Label(control_frame_YOLO, text="NMS IOU threshold", style = 'param.TLabel')
lb_Iou_thres.place(x=20, y=235, width=185, height=25)

ent_Iou_thres = ttk.Entry(control_frame_YOLO, textvariable= 'ent_Iou_thres', style = 'param.TEntry', justify = CENTER)
ent_Iou_thres.place(x=215, y=235, width = 50, height=25)
root.setvar('ent_Iou_thres', str(ParametersYOLO.Iou_thres))

lb_Max_det = ttk.Label(control_frame_YOLO, text="Maximum detections per image", style = 'param.TLabel')
lb_Max_det.place(x=20, y=275, width=295, height=25)

ent_Max_det = ttk.Entry(control_frame_YOLO, textvariable= 'ent_Max_det', style = 'param.TEntry', justify = CENTER)
ent_Max_det.place(x=325, y=275, width = 50, height=25)
root.setvar('ent_Max_det', str(ParametersYOLO.Max_det))

lb_line_thickness = ttk.Label(control_frame_YOLO, text="Line Thickness", style = 'param.TLabel')
lb_line_thickness.place(x=20, y=315, width=150, height=25)

ent_line_thickness = ttk.Entry(control_frame_YOLO, textvariable= 'ent_line_thickness', style = 'param.TEntry', justify = CENTER)
ent_line_thickness.place(x=180, y=315, width = 50, height=25)
root.setvar('ent_line_thickness', str(ParametersYOLO.line_thickness))

chk_hide_labels =ttk.Checkbutton(control_frame_YOLO, takefocus = 0, style = 'param.TCheckbutton', textvariable = 'chk_hide_labels', variable= 'chk_hidelabels')
chk_hide_labels.place(x=20, y=355, width = 130, height = 30)
root.setvar('chk_hide_labels', 'hide labels')
root.setvar('chk_hidelabels', ParametersYOLO.hide_labels)

chk_hide_conf =ttk.Checkbutton(control_frame_YOLO, takefocus = 0, style = 'param.TCheckbutton', textvariable = 'chk_hide_conf', variable= 'chk_hideconf')
chk_hide_conf.place(x=20, y=395, width = 190, height = 30)
root.setvar('chk_hide_conf', 'hide confidences')
root.setvar('chk_hideconf', ParametersYOLO.hide_conf)

#endregion

#region Traffic Light

control_frame_Light = Frame(tab3, bg="lightblue", relief=SUNKEN)
control_frame_Light.place(x=10, y=455, width = 595, height=110)

lb_param_Light = ttk.Label(control_frame_Light, text="Traffic Light", style='my2.TLabel')
lb_param_Light.place(x=2, y=2, width=180, height=25)

chk_btn_Light_Enable =ttk.Checkbutton(control_frame_Light, takefocus = 0, style = 'param.TCheckbutton', 
                                      textvariable = 'chk_btn_Light_Enable', variable= 'chk_btnLightEnable', command=ParametersLight.Update_Parameters)
chk_btn_Light_Enable.place(x=20, y=35, width = 90, height = 30)
root.setvar('chk_btn_Light_Enable', 'Enable')
root.setvar('chk_btnLightEnable', ParametersLight.enable)

lb_Light_Model = ttk.Label(control_frame_Light, text="Model", style = 'param.TLabel')
lb_Light_Model.place(x=20, y=75, width=65, height=25)

ent_Light_Model = ttk.Entry(control_frame_Light, textvariable= 'ent_Light_Model', style = 'param.TEntry')
ent_Light_Model.place(x=95, y=75, width = 410, height=25)
root.setvar('ent_Light_Model', str(ParametersLight.ModelPath))

def btn_Browse_Light_Model(e):
    s = gui.fileopenbox("MSG","TITLE","D:\Machine Learning\工研院產業新尖兵\專題\結案報告\Demo\Light\Model\*.sav")
    if s is not None:
        root.setvar('ent_Light_Model', s)
        ParametersLight.ModelPath = s
        ParametersLight.loadModel()

btn_Light_Model = ttk.Button(control_frame_Light, text="Browse", style='my2.TButton')
btn_Light_Model.place(x=508, y=72, width = 80, height=30)
btn_Light_Model.bind('<ButtonRelease-1>', btn_Browse_Light_Model)

#endregion

#region Sign (Deep Learning)

control_frame_SignDL = Frame(tab3, bg="lightblue", relief=SUNKEN)
control_frame_SignDL.place(x=10, y=575, width = 595, height=110)

lb_frame_Sign = ttk.Label(control_frame_SignDL, text="Sign (Deep Learning)", style='my2.TLabel')
lb_frame_Sign.place(x=2, y=2, width=220, height=25)

chk_btn_SignDL_Enable =ttk.Checkbutton(control_frame_SignDL, takefocus = 0, style = 'param.TCheckbutton', 
                                       textvariable = 'chk_btn_SignDL_Enable', variable= 'chk_btnSignDLEnable', command=ParametersSignDL.Update_Parameters)
chk_btn_SignDL_Enable.place(x=20, y=35, width = 90, height = 30)
root.setvar('chk_btn_SignDL_Enable', 'Enable')
root.setvar('chk_btnSignDLEnable', ParametersSignDL.enable)

lb_SignDL_Model = ttk.Label(control_frame_SignDL, text="Model", style = 'param.TLabel')
lb_SignDL_Model.place(x=20, y=75, width=65, height=25)

ent_SignDL_Model = ttk.Entry(control_frame_SignDL, textvariable= 'ent_SignDL_Model', style = 'param.TEntry')
ent_SignDL_Model.place(x=95, y=75, width = 410, height=25)
root.setvar('ent_SignDL_Model', str(ParametersSignDL.ModelPath))

def btn_Browse_SignDL_Model(e):
    s = gui.fileopenbox("MSG","TITLE","D:\Machine Learning\工研院產業新尖兵\專題\結案報告\Demo\Sign\DL\h5\*.h5")
    if s is not None:
        root.setvar('ent_SignDL_Model', s)
        ParametersSignDL.ModelPath = s
        ParametersSignDL.loadModel()

btn_SignDL_Model = ttk.Button(control_frame_SignDL, text="Browse", style='my2.TButton')
btn_SignDL_Model.place(x=508, y=72, width = 80, height=30)
btn_SignDL_Model.bind('<ButtonRelease-1>', btn_Browse_SignDL_Model)

#endregion

#region Sign (Machine Learning)

control_frame_SignML = Frame(tab3, bg="lightblue", relief=SUNKEN)
control_frame_SignML.place(x=10, y=695, width = 595, height=290)

lb_frame_SignML = ttk.Label(control_frame_SignML, text="Sign (Machine Learning)", style='my2.TLabel')
lb_frame_SignML.place(x=2, y=2, width=240, height=25)

chk_btn_SignML_Enable =ttk.Checkbutton(control_frame_SignML, takefocus = 0, style = 'param.TCheckbutton', 
                                       textvariable = 'chk_btn_SignML_Enable', variable= 'chk_btnSignMLEnable', command=ParametersSignML.Update_Parameters)
chk_btn_SignML_Enable.place(x=20, y=35, width = 90, height = 30)
root.setvar('chk_btn_SignML_Enable', 'Enable')
root.setvar('chk_btnSignMLEnable', ParametersSignML.enable)

lb_SignML_Model = ttk.Label(control_frame_SignML, text="Model", style = 'param.TLabel')
lb_SignML_Model.place(x=20, y=75, width=65, height=25)

ent_SignML_Model = ttk.Entry(control_frame_SignML, textvariable= 'ent_SignML_Model', style = 'param.TEntry')
ent_SignML_Model.place(x=95, y=75, width = 410, height=25)
root.setvar('ent_SignML_Model', str(ParametersSignML.ModelPath))

def btn_Browse_SignML_Model(e):
    s = gui.fileopenbox("MSG","TITLE","D:\Machine Learning\工研院產業新尖兵\專題\結案報告\Demo\Sign\ML\Model\*.sav")
    if s is not None:
        root.setvar('ent_SignML_Model', s)
        ParametersSignML.ModelPath = s
        ParametersSignML.loadModel()

btn_SignML_Model = ttk.Button(control_frame_SignML, text="Browse", style='my2.TButton')
btn_SignML_Model.place(x=508, y=72, width = 80, height=30)
btn_SignML_Model.bind('<ButtonRelease-1>', btn_Browse_SignML_Model)



# lb_SignML_Preprocess = ttk.Label(control_frame_SignML, text="Preprocess", style = 'param.TLabel')
# lb_SignML_Preprocess.place(x=20, y=75, width=120, height=25)

# def cb_SignML_Preprocess_Update(e = None):
#     pass
#     global photo_SignML_Orig, photo_SignML_Preprocess
#     # print(root.getvar('cb_Image2'))
#     path = source_dir_img2 + '/' + root.getvar('cb_Image2')
#     img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), 1)
#     # Resize
#     img = cv2.resize(img, (600, 410))
#     photo_SignML_Orig = PIL.ImageTk.PhotoImage(image = PIL.Image.fromarray(img[:, :, ::-1]))
#     ca_Image2.create_image(0, 0, image=photo_SignML_Orig, anchor='nw')

# cb_SignML_Preprocess = ttk.Combobox(control_frame_SignML, font="Arial 12", height=10, width=10, textvariable= 'cb_Image2')
# cb_SignML_Preprocess.place(x=20, y=115, width = 200, height=30)
# cb_SignML_Preprocess.config(values=['Blue Channel', 'Green Channel', 'Red Channel', 'RGB', 'Hue Channel', 'Saturation Channel', 'Value channel', 'HSV', 'Grayscale', 'Do EqualizeHist', 'Do Canny', 'Do EqualizeHist & Canny'])
# cb_SignML_Preprocess.bind("<<ComboboxSelected>>", cb_SignML_Preprocess_Update)
# cb_SignML_Preprocess.current(0)

# ca_SignML_Orig = Canvas(control_frame_SignML, bg='white')
# ca_SignML_Orig.place(x=150, y=80, width = 180, height = 125)

# def make_label(master, x, y, w, h, img, *args, **kwargs):
#     f = Frame(master, height = h, width = w)
#     f.pack_propagate(0) 
#     f.place(x = x, y = y)
#     label = Label(f, image = img, *args, **kwargs)
#     label.pack(fill = BOTH, expand = 1)
#     return label

# photo_SignML_arrow = ImageTk.PhotoImage(Image.open('GUI Image/arrow_right.png'))
# make_label(control_frame_SignML, 340, 125, 40, 40, photo_SignML_arrow, background = 'lightblue')

# def btn_SignML_image_press(e):
#     pass

# photo_SignML_image = PhotoImage(file = "GUI Image/image.png")
# btn_SignML_image = Button(control_frame_SignML, image = photo_SignML_image, background = 'lightblue')
# btn_SignML_image.place(x=340, y=85, width=40, height=40)
# btn_SignML_image.bind('<ButtonRelease-1>', btn_SignML_image_press)

# ca_SignML_Preprocess = Canvas(control_frame_SignML, bg='white')
# ca_SignML_Preprocess.place(x=390, y=80, width = 180, height = 125)

#endregion

#endregion

#region 【Others】->images2 & 3

lb_Image2 = ttk.Label(tab4, text="Image", style='my.TLabel')
lb_Image2.place(x=10, y=10, width=100, height=30)

ent_Image2 = ttk.Entry(tab4, width=60, textvariable= 'ent_Image2')
ent_Image2.place(x=120, y=10, width = 200, height=30)
root.setvar('ent_Image2', 'Inspect image directory 2')

btn_brow_Image2 = ttk.Button(tab4, text="Browse", style='my2.TButton')
btn_brow_Image2.place(x=330, y=10, width = 80, height=30)
btn_brow_Image2.bind('<ButtonRelease-1>', btn_Browse)

ca_Image2 = Canvas(tab4, width=500, height=400, bg='white')
ca_Image2.place(x=10, y=45, width = 600, height = 410)

def cb_Image2_Update(e = None):
    global photo_cb_Image2
    # print(root.getvar('cb_Image2'))
    path = source_dir_img2 + '/' + root.getvar('cb_Image2')
    img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), 1)
    # Resize
    img = cv2.resize(img, (600, 410))
    photo_cb_Image2 = PIL.ImageTk.PhotoImage(image = PIL.Image.fromarray(img[:, :, ::-1]))
    ca_Image2.create_image(0, 0, image=photo_cb_Image2, anchor='nw')

cb_Image2 = ttk.Combobox(tab4,font="Arial 12", height=10, width=10, textvariable= 'cb_Image2')
cb_Image2.place(x=420, y=10, height=30, width = 90)
cb_Image2.config(values=[])
cb_Image2.bind("<<ComboboxSelected>>", cb_Image2_Update)



lb_Image3 = ttk.Label(tab4, text="Image", style='my.TLabel')
lb_Image3.place(x=10, y=460, width=100, height=30)

ent_Image3 = ttk.Entry(tab4, width=60, textvariable= 'ent_Image3')
ent_Image3.place(x=120, y=460, width = 200, height=30)
root.setvar('ent_Image3', 'Inspect image directory 3')

btn_brow_Image3 = ttk.Button(tab4, text="Browse", style='my2.TButton')
btn_brow_Image3.place(x=330, y=460, width = 80, height=30)
btn_brow_Image3.bind('<ButtonRelease-1>', btn_Browse)

ca_Image3 = Canvas(tab4, width=500, height=400, bg='white')
ca_Image3.place(x=10, y=495, width = 600, height = 410)

def cb_Image3_Update(e = None):
    global photo_cb_Image3
    # print(root.getvar('cb_Image3'))
    path = source_dir_img3 + '/' + root.getvar('cb_Image3')
    img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), 1)
    # Resize
    img = cv2.resize(img, (600, 410))
    photo_cb_Image3 = PIL.ImageTk.PhotoImage(image = PIL.Image.fromarray(img[:, :, ::-1]))
    ca_Image3.create_image(0, 0, image=photo_cb_Image3, anchor='nw')

cb_Image3 = ttk.Combobox(tab4, font="Arial 12", width=10, height=10, textvariable= 'cb_Image3')
cb_Image3.place(x=420, y=460, width = 90, height=30)
cb_Image3.config(values=[])
cb_Image3.bind("<<ComboboxSelected>>", cb_Image3_Update)

btn_Next2 = ttk.Button(tab4, text="Next", image = photo_icon_next)
btn_Next2.pack(side = TOP)
btn_Next2.place(x=570, y=5, height=40)

btn_Prev2 = ttk.Button(tab4, text="Prev", image = photo_icon_prev)
btn_Prev2.pack(side = TOP)
btn_Prev2.place(x=520, y=5, height=40)

def btn_NextPrevImg2(e, b_next = None):
    index_now = cb_Image2.current()
    if b_next == None:
        if e.widget is btn_Next2:
            index_max = len(cb_Image2['values']) - 1
            if index_now >= index_max:
                return
            index_now += 1
        elif e.widget is btn_Prev2:
            if index_now <= 0:
                return
            index_now -= 1
    else:
        if b_next:
            index_max = len(cb_Image2['values']) - 1
            if index_now >= index_max:
                return
            index_now += 1
        else:
            if index_now <= 0:
                return
            index_now -= 1
    cb_Image2.current(index_now)
    cb_Image2_Update() # 更新影像顯示

btn_Next2.bind('<ButtonRelease-1>', btn_NextPrevImg2)
btn_Prev2.bind('<ButtonRelease-1>', btn_NextPrevImg2)


btn_Next3 = ttk.Button(tab4, text="Next", image = photo_icon_next)
btn_Next3.pack(side = TOP)
btn_Next3.place(x=570, y=455, height=40)

btn_Prev3 = ttk.Button(tab4, text="Prev", image = photo_icon_prev)
btn_Prev3.pack(side = TOP)
btn_Prev3.place(x=520, y=455, height=40)

def btn_NextPrevImg3(e, b_next = None):
    index_now = cb_Image3.current()
    if b_next == None:
        if e.widget is btn_Next3:
            index_max = len(cb_Image3['values']) - 1
            if index_now >= index_max:
                return
            index_now += 1
        elif e.widget is btn_Prev3:
            if index_now <= 0:
                return
            index_now -= 1
    else:
        if b_next:
            index_max = len(cb_Image3['values']) - 1
            if index_now >= index_max:
                return
            index_now += 1
        else:
            if index_now <= 0:
                return
            index_now -= 1
    cb_Image3.current(index_now)
    cb_Image3_Update() # 更新影像顯示

btn_Next3.bind('<ButtonRelease-1>', btn_NextPrevImg3)
btn_Prev3.bind('<ButtonRelease-1>', btn_NextPrevImg3)

#endregion

# root.mainloop()

#endregion



if __name__ == "__main__":
    
    root.mainloop()
