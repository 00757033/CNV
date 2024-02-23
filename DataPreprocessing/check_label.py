import cv2
import shutil
import pathlib as pl
import numpy as np
import pandas as pd
import os
import tools.tools as tools
import matplotlib.pyplot as plt

class getData():
    def __init__(self,path,image_path,label_path,all_layers = ["1_OCT", "2_OCT", "3_OCT", "4_OCT", "1", "2", "3", "4"],layers = {"3":"OR","4":"CC"}):
        self.path = path
        self.label_path = label_path
        self.image_path = image_path
        self.layers = layers
        self.all_layers = all_layers
        self.dataframe = list()
        
    # collect the label data
    def labelDict(self,show = True,data_name = "OCTA",label_name = "labeled",mask_name = "label"):
        self.label_dict = dict()  
        for patientID in pl.Path(self.label_path).iterdir():
            patient_dict = dict()
            for date in patientID.iterdir():
                date_dict = dict()
                for eyes in date.iterdir():
                    eyes_list = list()
                    for label in eyes.iterdir():
                        print(label)
                        img_path = str(label).replace(label_name,data_name)
                        img_path = img_path.replace(mask_name + '_' , '')
                        print(img_path)
                        img = cv2.imread(str(img_path))
                        labelimg = cv2.imread(str(label), cv2.IMREAD_GRAYSCALE)

                        labelimg = cv2.cvtColor(labelimg, cv2.COLOR_GRAY2BGR)
                        labelimg[ : , : , 0] = 0
                        labelimg[ : , : , 2] = 0
                        
                        vis = cv2.addWeighted(img, 0.7, labelimg, 0.3, 0)
                        labelname = label.stem.replace(mask_name + '_','')
                        if show:
                            plt.imshow(vis)
                            plt.axis('off')
                            plt.title( patientID.name + ' '+  date.name + ' '+ eyes.name + ' '+ self.layers[labelname])
                            plt.show()
                        # save the image
                        os.makedirs(os.path.join(self.path,'checklabel'), exist_ok=True)
                        cv2.imwrite(os.path.join(self.path,'checklabel',self.layers[labelname] + '_' + patientID.name+ '_' +eyes.name+ '_' +date.name + '.png'), vis)
                        
                        
                            
                            
                            
                        




if __name__ == '__main__':
    date = '0203'
    disease = 'PCV'
    PATH = "../../Data/"
    PATH_BASE =  PATH + "/" + disease + "_"+ date
    label_name = "labeled"
    Data_name = "OCTA"
    PATH_LABEL = PATH + "/" + label_name
    
    PATH_IMAGE = PATH + "/" + Data_name
    tools.makefolder(PATH_BASE)
    data = getData(PATH,PATH_IMAGE,PATH_LABEL)
    data.labelDict(show = False,data_name = Data_name,label_name = label_name,mask_name = "label")
    
    
    