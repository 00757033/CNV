import pathlib as pl
from collections import Counter
import pandas as pd
import os
import re
import shutil
import sys
import os
import cv2
import tools.tools as tools
import matplotlib.pyplot as plt

def check_label(image_folder,label_folder,data = "../../Data/",show = False,save=True,save_path = "../../Data/checklabel/"):
    layers = {"3":"OR","4":"CC"}
    if save:
        for layer in layers:
            tools.makefolder(os .path.join(save_path,layers[layer],'images'))
            tools.makefolder(os .path.join(save_path,layers[layer],'labels'))
    for patient in pl.Path(data+label_folder).iterdir():
        for date in patient.iterdir():
            for eye in date.iterdir():
                for label in eye.iterdir():
                    print(label)
                    img_path = str(label).replace(label_folder,image_folder)
                    img_path = img_path.replace("label_" , '')
                    print(img_path)
                    img = cv2.imread(str(img_path))
                    labelimg = cv2.imread(str(label))
                    labelimg[ : , : , 0] = 0
                    labelimg[ : , : , 2] = 0
                    
                    vis = cv2.addWeighted(img, 0.7, labelimg, 0.2, 0)
                    labelname = label.stem.replace("label_",'')
                    if show:
                        plt.imshow(vis)
                        plt.axis('off')
                        # plt.title( patient.name + ' '+  date.name + ' '+ eye.name + ' '+ labelname)
                        plt.show()
                        print(save_path+ patient.name+ '_' +eye.name+ '_' +date.name  + '_' + labelname + '.png')
                    if save:
                        cv2.imwrite(save_path+ '/' + layers[labelname] + '/' + 'labels/' + patient.name+ '_' +eye.name+ '_' +date.name  + '_' + labelname + '.png', vis)
                        cv2.imwrite(save_path+ '/' + layers[labelname] + '/' + 'images/' + patient.name+ '_' +eye.name+ '_' +date.name  + '_' + labelname + '.png', img)



if __name__ == '__main__':
    Data = "../../Data/"
    image_folder = "OCTA"
    label_folder = "labeled"
    check_label(image_folder,label_folder,data = Data)
    