import cv2
import glob
import os
import numpy as np
import pandas as pd
from datetime import datetime
import tools.tools as tools
#取得治療前後對應的影像
class getCompareData():
    def __init__(self, PATH_BASE, types,file = '../../Data/打針資料.xlsx'):
        self.PATH_BASE = PATH_BASE
        self.types = types
        self.inject(file)

    def getData(self, data_class, data_group, layer, data_label): 
        path = self.PATH_BASE  + 'align/'+ data_group + '/'
        path_output = self.PATH_BASE + 'compare/' + data_group
        print(path_output)
        setFloder(path_output)
        data_list = sorted(os.listdir(path + data_label))
        patient = {}
        eyes = {'R':'OD','L':'OS'}
        for data_name in data_list:
            if data_name.endswith(".png"):
                patient_id, eye, date = data_name.split('.png')[0].split('_')
                mapped_eye = eyes.get(eye, eye)
                
                disease_eye = self.inject_df[((self.inject_df['病歷號'] == patient_id) & (self.inject_df['眼睛'] == eyes[eye]) )]
                
                label_post = cv2.imread(path + 'masks' + '/' + data_name)  
                image_post = cv2.imread(path + 'images' + '/' + data_name)                  
                # find the key
                columns_of_interest = ["打針前門診日期","三針後門診","六針後門診","九針後門診","十二針後門診"]				
                for index, row in disease_eye.iterrows():
                    for col_name, col_value in row.items():
                        if pd.notna(col_value) and isinstance(col_value, datetime):
                                col_value = datetime.strftime(col_value, '%Y%m%d')
                                if date == col_value:
                                    treatment_count = columns_of_interest.index(col_name)
                                    if patient_id + '_' + eye not in patient :
                                        patient[patient_id + '_' + eye] = []
                                    patient[patient_id + '_' + eye] .append(treatment_count)
                                    print(patient_id, eye, date ,col_name,treatment_count)    
                                    if not os.path.exists(path_output + '/' + patient_id + '_' + eye+ '/' + 'masks' ):
                                        setFloder(path_output + '/' + patient_id + '_' + eye+ '/' + 'masks' )
                                        setFloder(path_output + '/' + patient_id + '_' + eye+ '/' + 'images' )                                          
                                    data_post = patient_id + '_'+ eye + '.png'     
                                    cv2.imwrite(path_output + '/' + patient_id + '_' + eye + '/' + 'masks' + '/' +  str(date) + '.png', label_post)
                                    cv2.imwrite(path_output + '/' + patient_id + '_' + eye + '/' + 'images' + '/' +  str(date) + '.png', image_post)                                         
                                                                         

                # label_post = cv2.imread(path + 'masks' + '/' + data_name)  
                # image_post = cv2.imread(path + 'images' + '/' + data_name)   
                # data_post = patient_id + '_'+ eye + '.png' 
                # if patient_id + '_' + eye not in patient :
                #     patient[patient_id + '_' + eye] = 1
                #     treatment_count = 1 
                # else :
                #     patient[patient_id + '_' + eye]+=1
                #     treatment_count = patient[patient_id + '_' + eye]
                # if not os.path.exists(path_output + '/' + patient_id + '_' + eye+ '/' + 'masks' ):
                #     setFloder(path_output + '/' + patient_id + '_' + eye+ '/' + 'masks' )
                #     setFloder(path_output + '/' + patient_id + '_' + eye+ '/' + 'images' )
                    
                # cv2.imwrite(path_output + '/' + patient_id + '_' + eye + '/' + 'masks' + '/' + str(date) + '.png', label_post)
                # cv2.imwrite(path_output + '/' + patient_id + '_' + eye + '/' + 'images' + '/' + str(date) + '.png', image_post)      
        return patient        

    def inject(self,file = '../../Data/打針資料.xlsx',label = ["診斷","病歷號","眼睛","打針前門診日期","三針後門診","六針後門診","九針後門診","十二針後門診"]):
        # self.inject_df = pd.DataFrame()
        self.inject_df = pd.read_excel(file, sheet_name="20230830",na_filter = False, engine='openpyxl')
        self.inject_df['病歷號'] = self.inject_df['病歷號'].apply(lambda x: '{:08d}'.format(x))
        self.inject_df = self.inject_df.sort_values(by=["病歷號","眼睛"])


def setFloder(path):
    os.makedirs(path, exist_ok=True)

def run():
    
    disease   = 'PCV'
    date    = '0205'
    PATH_BASE    = "../../Data/" + disease + '_' + date + '/'
    data_groups  = ["OR", "CC"]
    dict_origin  = {'OR': "3" , 'CC': "4"}
    data_label = "masks"
    data = getCompareData(PATH_BASE,  types = data_groups , file = '../../Data/打針資料.xlsx')
    setFloder('./record/'+disease + '_' + date + '/')
    for data_group in data_groups:
        patient = data.getData(disease, data_group, dict_origin[data_group], data_label)
        json_file = './record/'+ disease + '_' + date + '/'+ data_group + '_Treatment.json'
        tools.write_to_json_file(json_file, patient)


if __name__ == '__main__':
    run()   