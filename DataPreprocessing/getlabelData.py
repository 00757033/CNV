import glob
import os
from pathlib import Path

import cv2
import numpy as np
from function import setFolder


#get the oct and octa images from the OCTA folder according to the label
class getLabelData():
    def __init__(self, path_label,path_image , path_output,layers = ["1_OCT", "2_OCT", "3_OCT", "4_OCT", "1", "2", "3", "4"], types = ["OR", "CC", "ALL"] ):
        self.path_label = Path(path_label)
        self.path_image = path_image
        self.path_output = Path(path_output)
        self.layers = layers
        self.types = types
        for type in types:
            setFolder(self.path_output / type)
            setFolder(self.path_output / type / "label")
            if type == "CC":
                layers = ["4","4_OCT"]
            elif type == "OR":
                layers = ["3","3_OCT"]
            else:
                layers = ["1_OCT", "2_OCT", "3_OCT", "4_OCT", "1", "2", "3", "4"]
            for layer in layers:
                setFolder(self.path_output / type / layer)

    def getLabelData(self, data_class):
        set1 = set()# set the patient number
        print("start get label data :"+data_class)
        for patient in self.path_label.glob("*"):
            patient_number = str(patient.stem)            
            set1.add(patient_number) # add the patient number to the set
            number = 0
            date_list = []
            for date in patient.glob("*"): 
                number = number+1
                for eye in date.glob("*"): 
                    eye_type = str(eye.stem)
                    path_image = str(eye).replace(str(self.path_label), str(self.path_image)) # get the path of the OCTA images
                    for data in eye.glob("*.png"): #
                        data = str(data)
                        label = cv2.imread(data)*255
                        label = cv2.resize(label, (304,304))
                        data_image = data.replace(str(eye), "") # get the name of the image
                        for type in self.types:
                            if type + '_' + data_class in data_image : # get the image of the type and class( ex CNV image)
                                saveImg(path_image, str(self.path_output)+ '/'+'ALL',type, patient_number,number, eye_type, self.layers)
                                if type == "CC": #save the CC layer image
                                    saveImg(path_image, str(self.path_output)+ '/'+type ,type, patient_number,number, eye_type, ["4","4_OCT"])
                                elif type == "OR": # save the OR layer image
                                    saveImg(path_image, str(self.path_output)+ '/'+type,type, patient_number ,number, eye_type, ["3","3_OCT"])
                                cv2.imwrite(str(self.path_output / type / "label" / Path(type + "_" + patient_number + "_" + eye_type + "_" + str(number) +".png")), label)
                                cv2.imwrite(str(self.path_output / 'ALL' / "label" / Path(type + "_" + patient_number + "_" + eye_type + "_" + str(number) +".png")), label)

def saveImg(path_image,path_output,layer_type, patient_number,number, eye,layers):
    for layer in layers:
        image = cv2.imread(path_image +  "/" + layer + ".png")
        image = cv2.resize(image, (304,304))
        cv2.imwrite(path_output + '/' + layer + "/" + layer_type + "_" + patient_number  + "_" + eye  + "_" + str(number) +".png", image)

if __name__ == '__main__':
    data_class = 'CNV'
    date = '0324'
    path_base = Path("../../Data") 
    path_label = path_base / "label"
    path_image = path_base / "OCTA"
    path_output =  path_base/ Path(data_class+ '_' + date + '/')
    setFolder(path_output)
    data = getLabelData(path_label, path_image, path_output)
    data.getLabelData(data_class)