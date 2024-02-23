import shutil
import os
import tools.tools as tools
import cv2
import numpy as np
import pandas as pd
import pathlib as pl
class getgroup():
    def __init__(self,path,layers = {"3":"OR","4":"CC"}):
        self.path = path
        self.layers = layers
        self.dataframe = []
        pass
    def getGroupDataframe(self):
        return pd.DataFrame(self.dataframe,columns = ['type','image','group'])
    
    def saveGroupDataframe(self,path = './record/group.csv'):
        self.getGroupDataframe().to_csv(path,index = False)

    def getgroup(self):
        self.dataframe.clear()

        for layer in self.layers:
            print(self.layers[layer])
            for image in pl.Path(self.path+'/'+self.layers[layer]+'/'+'masks').iterdir():
                image = image.stem
                name = image.split('_')
                group = "_".join(name[:3])
                self.dataframe.append([self.layers[layer],image,group])
            print(len(self.dataframe))
        self.saveGroupDataframe()
        return self.dataframe


if __name__ == '__main__':
    date = '0918'
    disease = 'PCV'
    PATH = "../../Data/"
    PATH_BASE =  PATH + "/" + disease + "_"+ date
    PATH_LABEL = PATH + "/" + "labeled"
    PATH_IMAGE = PATH + "/" + "OCTA"
    group = getgroup(PATH_BASE)
    group.getgroup()
    group.saveGroupDataframe()
    pd = group.getGroupDataframe()
    print(pd)
