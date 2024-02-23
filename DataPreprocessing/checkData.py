import cv2
import numpy as np
from tools import tools
import os
import pathlib as pl
import shutil
import re

class checkData():
    def __init__(self, PATH, image_path,layers = {"3":"OR","4":"CC"}):
        self.PATH = PATH
        self.image_path = image_path
        self.layers = layers

    def check(self): 
        # if file in layers 3 not in layers 4, then delete the file
        # if file in layers 4 not in layers 3, then delete the file
        # if file in layers 3 and layers 4, then check the size of the file
        for layer in self.layers:
            for image in pl.Path(os.path.join(self.image_path,self.layers[layer],'images')).iterdir():
                image_name = image.name
                image_stem = image.stem
                # delete  the CC_ in CC_id_eye_date.png or  OR_ in OR_id_eye_date.png
                filtered_image_stem  = re.sub(r'CC_|OR_','',image_stem)

                print(filtered_image_stem)
                # check if the file in layers 3 not in layers 4
                if layer == '3':
                    if not os.path.exists(os.path.join(self.image_path,self.layers['4'],'images',self.layers['4'] + '_' + filtered_image_stem + '.png')):
                        os.remove(image)   
                        # remove the mask file
                        os.remove(os.path.join(self.image_path,self.layers['3'],'masks',self.layers['3'] + '_' + filtered_image_stem + '.png')) 
                # check if the file in layers 4 not in layers 3
                if layer == '4':
                    if not os.path.exists(os.path.join(self.image_path,self.layers['3'],'images',self.layers['3'] + '_' + filtered_image_stem + '.png')):
                        os.remove(image)
                        # remove the mask file
                        os.remove(os.path.join(self.image_path,self.layers['4'],'masks',self.layers['4'] + '_' + filtered_image_stem + '.png'))
                # check the size of the file
                











if __name__ == "__main__":
    import cv2
    date = '1203'
    disease = 'PCV'
    PATH = "../../Data/"
    FILE = disease + "_"+ date

    check = checkData(PATH,os.path.join(PATH,FILE))
    check.check()

