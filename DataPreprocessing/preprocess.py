import cv2
import numpy as np
from matplotlib import pyplot as plt
from pathlib import Path
from segmentation.function  import setFolder , otsu

class PreprocessData():
    
    def preprocess(self, path, layer):
        self.output_path = Path(path+'/otsu')
        setFolder(self.output_path) # set the output folder
        self.output_path = str(self.output_path)
        image_path = Path(path + '/'+layer)
        for image_name in image_path.glob("*.png"):
            name = str(image_name.name)
            image = cv2.imread(str(image_path) + '/' + name, 0)
            image_mask = otsu(image)
            cv2.imwrite(self.output_path + '/' + name, image_mask)

if __name__ == '__main__':
    data_class = 'CNV'
    date = '0324'
    path_base = Path("../../Data") 
    data_groups = ["OR", "CC"] 
    dict_origin = {'OR': "3" , 'CC': "4"}
    path_output =  path_base/ Path(data_class+ '_' + date + '/')
    for data_group in data_groups:
        path_preprocess = str(path_output) + '/' + data_group
        preprocess = PreprocessData(str(path_output), data_group)
        for it in dict_origin[data_group] : 
            preprocess.preprocess(path_preprocess, it)
