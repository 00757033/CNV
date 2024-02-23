import cv2
import os
import numpy as np
import shutil
import matplotlib
matplotlib.use('TkAgg')  # 使用TkAgg后端
import tools.tools as tools
import pathlib as pl
class Concatenation() :
    def __init__(self, PATH, image_path,layers = {"3":"OR","4":"CC"}):
        self.PATH = PATH
        self.image_path = image_path
        self.layers = layers
    #concateLayer需要合併的影像集
    def getConcate(self,input_name,ratio = 0.5):
        for layer in self.layers:
            input_path = os.path.join(self.image_path,  self.layers[layer],'images')
            tools.makefolder(os.path.join(self.image_path,input_name + '_concate_' + self.layers[layer],'images'))
            for image_path in pl.Path(input_path).iterdir():
                    image_name = image_path.name
                    image_stem = image_path.stem
                    split_image_name = image_stem.split("_")
                    
                    image_path = os.path.join(self.PATH,split_image_name[1],split_image_name[3],split_image_name[2])
                    image_path1 = os.path.join(image_path,'3.png')
                    image1   = cv2.imread(image_path1)
                    gray1   = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
                    gray1 = np.expand_dims(gray1, axis = 2)
                    image_path2 = os.path.join(image_path,'4.png')
                    image2   = cv2.imread(image_path2)
                    gray2   = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
                    gray2 = np.expand_dims(gray2, axis = 2)
                    if split_image_name[0] == 'OR':
                        merged_image = np.dstack([gray1, gray2, np.zeros_like(gray1)])
                    else:
                        merged_image = np.dstack([gray2, gray1, np.zeros_like(gray1)])

                    output_path = os.path.join(self.image_path,input_name + '_concate_' + self.layers[layer],'images',image_name)
                    # cv2.imwrite(output_path,merged_image)
            mask_path = os.path.join(self.image_path, 'otsu_'+ self.layers[layer],'masks')
            print(os.path.join(self.image_path,input_name + '_concate_' + self.layers[layer],'masks'))
            # shutil.copytree(mask_path, os.path.join(self.image_path,input_name + '_concate_' + self.layers[layer],'masks'))
            
                    




if __name__ == "__main__":
    import cv2
    print("OpenCV version:", cv2.__version__)

    image_path = '..\\..\\Data\\PCV_1120'
    path = '..\\..\\Data\\OCTA'
    label= Concatenation(path,image_path)
    label.getConcate('concate')        