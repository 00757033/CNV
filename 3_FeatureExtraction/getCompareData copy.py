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
        self.label_list = {'OR': "3" , 'CC': "4"}

    def getData(self, data_class, data_group, layer, data_label): 
        path = self.PATH_BASE  + 'align/'+ data_group + '/'
        path_output = self.PATH_BASE + 'compare/' 
        data_list = sorted(os.listdir(path + data_label))
        patient = {}
        eyes = {'R':'OD','L':'OS'}
        patient_eyes = set()
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
                                    # if not os.path.exists(path_output + '/' + str(treatment_count) + '/' + 'masks' ):
                                    #     setFloder(path_output + '/' + str(treatment_count) + '/' + 'masks' )
                                    #     setFloder(path_output + '/' + str(treatment_count) + '/' + 'images' ) 
                                    # data_post = patient_id + '_'+ eye + '_'+  str(date) + '.png'
                                    # cv2.imwrite(path_output + '/' + str(treatment_count) + '/' + 'masks' + '/' +  data_post, label_post)  
                                    # cv2.imwrite(path_output + '/' + str(treatment_count) + '/' + 'images' + '/' +  data_post, image_post) 
                                    if not os.path.exists(path_output + '/' + patient_id + '_' + eye + '/' + 'masks' ):
                                        setFloder(path_output + '/' + patient_id + '_' + eye + '/' + 'masks' )
                                        setFloder(path_output + '/' + patient_id + '_' + eye + '/' + 'images' )      
                                    # data_post = patient_id + '_'+ eye + '.png'     
                                    cv2.imwrite(path_output + '/' + patient_id + '_' + eye + '/' + 'masks' + '/' +  str(date)+ '_' + data_group+ '.png', label_post)
                                    cv2.imwrite(path_output + '/' + patient_id + '_' + eye + '/' + 'images' + '/' +  str(date) + '_' + data_group + '.png', image_post)                                         
                                                                         

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
                    
                # cv2.imwrite(path_output + '/' + patient_id + '_' + eye + '/' + 'masks' + '/' + str(date) + '_' + data_group+ '.png', label_post)
                # cv2.imwrite(path_output + '/' + patient_id + '_' + eye + '/' + 'images' + '/' + str(date) + '_' + data_group+ '.png', image_post)      
        return patient        


    def getData_all(self, data_class, data_group, layer, data_label,data_file = '../../Data/OCTA/'):
        path = self.PATH_BASE  + 'align/'+ data_group + '/'
        path_output = self.PATH_BASE + 'compare/' + data_group
        print(path_output)
        setFloder(path_output)
        data_list = sorted(os.listdir(path + data_label))
        patient = {}
        patient_eyes = set()
        eyes = {'R':'OD','L':'OS'}
        count = 0
        for data_name in data_list:
            count += 1
            # print(count)
            if data_name.endswith(".png"):
                patient_id, eye, date = data_name.split('.png')[0].split('_')
                mapped_eye = eyes.get(eye, eye)

                disease_eye = self.inject_df[((self.inject_df['病歷號'] == patient_id) & (self.inject_df['眼睛'] == eyes[eye]) )]
                columns_of_interest = ["三針後門診","六針後門診","九針後門診","十二針後門診"]	
                pre_date = disease_eye['打針前門診日期']
                if pre_date.empty:
                    continue

                pre_date_str = str(pre_date.values[0])  # Convert to string
                if 'T' in pre_date_str:
                    pre_date_str = pre_date_str.split('T')[0]  # Remove the time part
                if ' ' in pre_date_str:
                    pre_date_str = pre_date_str.split(' ')[0]
                formatted_date_str = datetime.strptime(pre_date_str, '%Y-%m-%d').strftime('%Y%m%d')  # Convert to the desired format
                
                if date == formatted_date_str:
                    image_post = cv2.imread(path + 'images' + '/' + data_name)
                    label_post = cv2.imread(path + 'masks' + '/' + data_name)
                    setFloder(path_output + '/' + '0' + '/' + 'images' )
                    cv2.imwrite(path_output + '/' + '0' + '/' + 'images' + '/' + patient_id + '_' + eye + '_' + str(date) + '.png', image_post)
                    cv2.imwrite(path_output + '/' + '0' + '/' + 'masks' + '/' + patient_id + '_' + eye + '_' + str(date) + '.png', label_post)
                    # cv2.imwrite(path_output + '/' + patient_id + '_' + eye + '/' + 'masks' + '/' + str(date) + '.png', label_post)
                    for columns in columns_of_interest:
                        if pd.notna(disease_eye[columns].values[0]) and disease_eye[columns].values[0] != '':
                            treatment_date = disease_eye[columns].values[0]
                            treatment_date_str = str(treatment_date)
                            if 'T' in treatment_date_str:
                                treatment_date_str = treatment_date_str.split('T')[0]
                            if ' ' in treatment_date_str:
                                treatment_date_str = treatment_date_str.split(' ')[0]
                            formatted_treatment_date_str = datetime.strptime(treatment_date_str, '%Y-%m-%d').strftime('%Y%m%d')
                            if os.path.exists( os.path.join(data_file, patient_id, formatted_treatment_date_str, eye)):
                                image_post = cv2.imread(os.path.join(data_file, patient_id, formatted_treatment_date_str, eye, self.label_list[data_group] + '.png'))
                                print(os.path.join(data_file, patient_id, formatted_treatment_date_str, eye, 'images', self.label_list[data_group] + '.png'))
                                # columns 為 columns_of_interest 的第幾個
                                index = columns_of_interest.index(columns) + 1
                                print('index',columns,index)
                                setFloder(path_output + '/' + str(index) + '/' + 'images' )
                                cv2.imwrite (path_output + '/' + str(index) + '/' + 'images' + '/' + patient_id + '_' + eye + '_' + str(date) + '.png', image_post)
                                
                    #             label_post = cv2.imread(os.path.join(data_file, patient_id, formatted_treatment_date_str, eye, 'masks', '0.png'))
                    #             setFloder(path_output + '/' + str(columns_of_interest) + '/' + 'images' )
                    #             cv2.imwrite(path_output + '/' + str(columns_of_interest) + '/' + 'images' + '/' + patient_id + '_' + eye + '_' + str(date) + '.png', image_post)
                    #             cv2.imwrite(path_output + '/' + str(columns_of_interest) + '/' + 'masks' + '/' + patient_id + '_' + eye + '_' + str(date) + '.png', label_post)
                    # # for index, row in disease_eye.iterrows():
                    #     for col_name, col_value in row.items():
                    #         if col_name in columns_of_interest:
                    #             if pd.notna(col_value) and isinstance(col_value, datetime) :
                    #                 col_value = datetime.strftime(col_value, '%Y%m%d')
                                    
                    #                 file = patient_id + '_' + eye  + '_' + str(date) + '.png'
                    #                 if os.path.exists( os.path.join(data_file, patient_id, str(date), eye)):
                                        
                                    
                                
                    		
                    # for index, row in disease_eye.iterrows():
                    #     for col_name, col_value in row.items():
                    #         if pd.notna(col_value) and isinstance(col_value, datetime):
                    #                 col_value = datetime.strftime(col_value, '%Y%m%d')
                    #                 print(patient_id, eye, date, col_value, date == col_value)
                                    # if date == col_value:
                                    #     treatment_count = columns_of_interest.index(col_name)
                                    #     if patient_id + '_' + eye not in patient :
                                    #         patient[patient_id + '_' + eye] = []
                                    #     patient[patient_id + '_' + eye] .append(treatment_count)
                                    #     print(patient_id, eye, date ,col_name,treatment_count)    
                                    #     if not os.path.exists(path_output + '/' + patient_id + '_' + eye+ '/' + 'masks' ):
                                    #         setFloder(path_output + '/' + patient_id + '_' + eye+ '/' + 'masks' )
                                    #         setFloder(path_output + '/' + patient_id + '_' + eye+ '/' + 'images' )                                          
                                    #     data_post = patient_id + '_'+ eye + '.png'     
                                    #     cv2.imwrite(path_output + '/' + patient_id + '_' + eye + '/' + 'masks' + '/' +  str(date) + '.png', label_post)
                                    #     cv2.imwrite(path_output + '/' + patient_id + '_' + eye + '/' + 'images' + '/' +  str(date) + '.png', image_post)                                         
                
                # else:
                #     print('patient_id + "_" + eye',patient_id + "_" + eye)                                                            

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