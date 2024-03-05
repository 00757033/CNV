import cv2
import shutil
import pathlib as pl
import numpy as np
import pandas as pd
import os
from inpaint import inpaint
from destripe import destripe_octa_image
import json


import tools.tools as tools
class getData():
    def __init__(self,path,image_path,label_path,label_title,all_layers = ["1_OCT", "2_OCT", "3_OCT", "4_OCT", "1", "2", "3", "4"],layers = {"4":"CC"}):
        self.path = path
        self.label_path = label_path
        self.image_path = image_path
        self.layers = layers
        self.all_layers = all_layers
        self.label_title = label_title
        self.dataframe = list()
        self.label_dict = dict()

    # collect the label data
    def labelDict(self,mask_name = "label"):
        self.label_dict = dict()  
        for patientID in pl.Path(self.label_path).iterdir():
            patient_dict = dict()
            for date in patientID.iterdir():
                date_dict = dict()
                for eyes in date.iterdir():
                    eyes_list = list()
                    for label in eyes.iterdir():
                        for layer in self.layers.keys():
                            if label.stem == mask_name + '_'+ layer:
                                eyes_list.append(layer)
                    if eyes_list :
                        date_dict[eyes.name] = eyes_list
                if date_dict:
                    patient_dict[date.name] = date_dict
            if patient_dict:
                self.label_dict[patientID.name] = patient_dict
    
    # get the label dictionary
    def getLabelDict(self):
        if self.label_dict:
            return self.label_dict
        else:
            self.labelDict()
            return self.label_dict

    def writeLabelJson(self,file= './record/label_dict.json'):
        tools.write_to_json_file(file,self.label_dict)


    def health(self,disease_file):
        # read json file
        disease = json.load(open(disease_file))
        dictionary = {"L" :"OS" , "R" : "OD"}


        self.health_dict = dict()  
        for patientID in pl.Path(self.image_path).iterdir():
            patient_dict = dict()
            for date in patientID.iterdir():
                date_dict = dict()
                for eyes in date.iterdir():
                    eyes_list = list()

                    if patientID.name in disease:
                        if dictionary[eyes.name] in disease[patientID.name]:
                            continue
                    for image in eyes.iterdir():
                        if image.suffix == '.png':
                            eyes_list.append(image.stem)
                    # eyes_list the same as the all_layers
                    if len(eyes_list) == len(self.all_layers):
                        date_dict[eyes.name] = eyes_list
                if date_dict:
                    patient_dict[date.name] = date_dict
            if patient_dict:
                self.health_dict[patientID.name] = patient_dict

        return self.health_dict



        


    # copy the image and label to the output path        
    def getData(self,label_dict,output_path,type,toinpaint = True,copy_image = True, copy_label = True):
        if toinpaint:
            preprocess = inpaint(self.path)
        if not  label_dict:
            if type == 'disease':
                label_dict = self.label_dict
            elif type == 'health':
                label_dict = self.health_dict
                
        layers = [value for value in self.layers.values()]
        # layers.append('ALL')
        folder_list = ['images','masks']
        for layer_name in layers:
            for folder in folder_list:
                tools.makefolder(output_path + '/' + layer_name + '/' + folder)
        self.dataframe.clear()
        for lay in self.all_layers:
            tools.makefolder( output_path + '/ALL/' + lay )

        for patientID in label_dict :
            for date in label_dict[patientID]:
                for eyes in label_dict[patientID][date]:
                    print("-------------")
                    new_image_name = patientID + '_' + eyes + '_' + date
                    if not self.errorimage('CC_'+new_image_name) and not self.errorimage('OR_'+new_image_name): # 刪除錯誤圖片
                        for lay in self.all_layers:
                            for image in pl.Path(os.path.join(self.image_path,patientID,date,eyes)).iterdir():
                                if image.stem == lay and image.suffix == '.png':
                                    if '_OCT' not in lay and toinpaint: # inpaint
        
                                        save_image = preprocess.extraneous_information(str(image))
                                    else :

                                        save_image = cv2.imread(os.path.join(self.image_path,patientID,date,eyes,lay+'.png'))
                                        if save_image is None:
                                            continue
                                    if save_image is None:
                                        continue
                                    save_image = cv2.resize(save_image,(304,304))
                                    cv2.imwrite(output_path + '/ALL/' + lay + '/' + new_image_name+'.png' ,save_image)
                                    if copy_image:
                                        if type == 'health':
                                            if image.stem =='3':
                                                cv2.imwrite(output_path + '/OR/images/'+ 'OR_' + new_image_name+'.png' ,save_image)
                                            elif image.stem =='4':
                                    
                                                cv2.imwrite(output_path + '/CC/images/'+ 'CC_' + new_image_name+'.png' ,save_image)
                                        elif type == 'disease':
                                            # label_dict[patientID][date][eyes] data split('_')[1] == image.stem 
                                            
                                            if image.stem in label_dict[patientID][date][eyes]:
                                                cv2.imwrite(output_path + '/'+self.layers[image.stem]+'/images/'+ new_image_name+'.png' ,save_image)
                                    if copy_label and type == 'disease':
                                        for label in label_dict[patientID][date][eyes]:
                                            origin_label_path = os.path.join(self.label_path,patientID,date,eyes,self.label_title + '_' + label+'.png')
                                            output_label_path = os.path.join(output_path,self.layers[label],'masks', new_image_name+'.png')

                                            shutil.copy(origin_label_path,output_label_path)


                        
                        if type == 'disease' and copy_label:
                            print('disease',)
                            # for label in label_dict[patientID][date][eyes]:
                            
                            #     layer_name = label.split('_')[1]
                            
                            #     origin_label_path = os.path.join(self.label_path,patientID,date,eyes,label+'.png')

                            #     if layer_name in self.layers:
                            #         new_image_name = self.layers[layer_name] + '_'+ patientID + '_' + eyes + '_' + date 

                            #         output_image_path = os.path.join(output_path,self.layers[layer_name],'images',new_image_name+'.png')
                            #         output_label_path = os.path.join(output_path,self.layers[layer_name],'masks',new_image_name+'.png')
                            #         if not self.errorimage('CC_'+new_image_name) and not self.errorimage('OR_'+new_image_name): # 刪除錯誤圖片
                            #             if copy_image  : 
                            #                 if toinpaint:
                            #                     new_image = preprocess.extraneous_information(origin_image_path)
                            #                 else :
                            #                     new_image = cv2.imread(origin_image_path)
                            #                 # new_image  = destripe_octa_image(new_image)
                            #                 new_image = cv2.resize(new_image,(304,304))
                            #                 cv2.imwrite(output_image_path,new_image)
                            #             if copy_label :
                            #                 shutil.copy(origin_label_path,output_label_path)

    def errorimage(self,image_name): # 判斷是否為錯誤圖片
        file = '..//..//Data//刪除.xlsx'
        data = pd.read_excel(file, engine='openpyxl')
        data = data.values.tolist()
        for i in data:
            if image_name == i[0]:
                return True
        return False


            
                
if __name__ == '__main__':
    date = '0305'
    disease = 'PCV'
    PATH = "../../Data/"
    PATH_BASE =  PATH + "/" + disease + "_"+ date
    PATH_LABEL = PATH + "/" + "labeled"
    PATH_IMAGE = PATH + "/" + "OCTA"
    tools.makefolder(PATH_BASE)
    data = getData(PATH_BASE,PATH_IMAGE,PATH_LABEL,"label")
    # data.labelDict("label")
    disease = data.getLabelDict()
    health = data.health('../DataAnalyze/PCV_disease.json')
    # data.writeLabelJson('./record/label_dict.json')
    types = ['disease','health']
    data.getData(disease,PATH_BASE,'disease',toinpaint = False,copy_image = True,copy_label = True)


