import cv2
import glob
import os
import numpy as np
import pandas as pd
from datetime import datetime
import tools.tools as tools
import matplotlib.pyplot as plt
#取得治療前後對應的影像
class getCompareData():
    def __init__(self, PATH_BASE, types,file = '../../Data/打針資料.xlsx',label_list = { 'CC': "4"}):
        self.PATH_BASE = PATH_BASE
        self.types = types
        self.inject(file)
        self.label_list = label_list

    def getData(self, data_class, data_group, layer, data_label): 
        print('getData',data_class, data_group, layer, data_label)
        path = self.PATH_BASE  + 'align/'+ data_group + '/' # input path
        path_output = self.PATH_BASE + 'compare/' # output path
        if not os.path.exists(path_output):
            os.makedirs(path_output)
        
        data_list =  sorted(os.listdir(os.path.join(path, data_label)))
        patient = {}
        eyes = {'R':'OD','L':'OS'}
        patient_eyes = set()
        for data_name in data_list:
            if data_name.endswith(".png"):
                patient_id, eye, date = data_name.split('.png')[0].split('_') # get patient_id, eye, date from align data layer
                
                mapped_eye = eyes.get(eye, eye)
                
                disease_eye = self.inject_df[((self.inject_df['病歷號'] == patient_id) & (self.inject_df['眼睛'] == eyes[eye]) )]
                
                label_post = cv2.imread(path + 'masks' + '/' + data_name)  
                image_post = cv2.imread(path + 'images' + '/' + data_name)  
                # cut 

                # img_scp = cv2.imread(os.path.join(self.PATH_BASE  , 'align', '1', data_name))
                # img_scp = cv2.normalize(img_scp, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
                # img_dcp = cv2.imread(os.path.join(self.PATH_BASE  , 'align', '2', data_name))
                # img_dcp = cv2.normalize(img_dcp, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
                # result = image_post.copy()
                # if  img_scp is not None and img_dcp is not None:
                #     img_scp = cv2.resize(img_scp, (304, 304))
                #     img_dcp = cv2.resize(img_dcp, (304, 304))
                #     result_and = np.zeros_like(image_post)
                #                             # clean
                #     mask = np.zeros_like(image_post, dtype=np.uint8)
                #     mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY) 
                    
                #     for i in range(image_post.shape[0]):
                #         for j in range(image_post.shape[1]):
                #             if np.array_equal(img_scp[i][j], img_dcp[i][j]):
                #                mask[i][j] = 255
                               
                #     result= cv2.inpaint( result , mask, 3, cv2.INPAINT_TELEA)
                #     result = cv2.normalize(result, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
                    
                    # for i in range(image_post.shape[0]):
                    #     for j in range(image_post.shape[1]):
                    #         if np.array_equal(img_scp[i][j], img_dcp[i][j]):
                    #             result_and[i][j] = img_scp[i][j]
                    #             if np.all(result[i][j] >=  img_scp[i][j]):
                    #                 mask = np.zeros_like(image_post)
                    #                 mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
                    #                 mask[i][j] = 255
                    #                 result= cv2.inpaint( result , mask, 3, cv2.INPAINT_TELEA)
                    # result = cv2.normalize(result, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
                    # result = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
                    # result = cv2.bilateralFilter(result, 5, 10, 10)
                    # result = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(8,8)).apply(result)
                    
                    
                    # # show the image
                    # fig, ax = plt.subplots(1, 4, figsize=(15, 15))
                    # ax[0].imshow(img_scp)
                    # ax[0].set_title('SCP')
                    # ax[1].imshow(img_dcp)
                    # ax[1].set_title('DCP')
                    # ax[2].imshow(result)
                    # ax[2].set_title('SCP - DCP')
                    
                    # ax[3].imshow(result_and)
                    # ax[3].set_title('SCP and DCP')
                    # plt.show()
         
                # find the key
                columns_of_interest = ["打針前門診日期","三針後門診"]	
                disease_eye = disease_eye[columns_of_interest]		
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

                                    if not os.path.exists(os.path.join(path_output, patient_id + '_' + eye, data_group,'masks')):
                                        setFloder(os.path.join(path_output, patient_id + '_' + eye, data_group,'masks'))
                                        setFloder(os.path.join(path_output, patient_id + '_' + eye, data_group,'images'))
                                    # if not os.path.exists(os.path.join(path_output, patient_id + '_' + eye, data_group + '_cut','images')):
                                    #     setFloder(os.path.join(path_output, patient_id + '_' + eye, data_group + '_cut','images'))
                                    #     setFloder(os.path.join(path_output, patient_id + '_' + eye, data_group + '_cut','masks'))
                                    # dpath_output + '/' + patient_id + '_' + eye + '/' + 'masks' + '/' +  str(date)+ '_' + data_group+ '.png', label_post
                                    cv2.imwrite(os.path.join(path_output, patient_id + '_' + eye, data_group,'masks', str(date)+ '_' + data_group+ '.png'), label_post)
                                    cv2.imwrite(os.path.join(path_output, patient_id + '_' + eye, data_group,'images', str(date)+ '_' + data_group+ '.png'), image_post)   
                                    
                                    # cv2.imwrite(os.path.join(path_output, patient_id + '_' + eye, data_group + '_cut','images', str(date)+ '_' + data_group+ '.png'), result)
                                    # cv2.imwrite(os.path.join(path_output, patient_id + '_' + eye, data_group + '_cut','masks', str(date)+ '_' + data_group+ '.png'), label_post)                                
                                                                         

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
        
    def inject(self,file = '../../Data/打針資料.xlsx',label = ["診斷","病歷號","眼睛","打針前門診日期","三針後門診"]):
        # self.inject_df = pd.DataFrame()
        print(file)
        self.inject_df = pd.read_excel(file, sheet_name="Focea_collect",na_filter = False, engine='openpyxl')
        # add pd.read_excel(file, sheet_name="20230831",na_filter = False, engine='openpyxl')
        # self.inject_df = self.inject_df.append(pd.read_excel(file, sheet_name="20230831",na_filter = False, engine='openpyxl'))

        self.inject_df['病歷號'] = self.inject_df['病歷號'].apply(lambda x: '{:08d}'.format(x))
        self.inject_df = self.inject_df.sort_values(by=["病歷號","眼睛"])


def setFloder(path):
    os.makedirs(path, exist_ok=True)

def run():
    
    disease   = 'PCV'
    date    = '20240418'
    PATH_BASE    = "../../Data/" + disease + '_' + date + '/'
    data_groups  = ["CC"]
    dict_origin  = {'CC': "4"}
    data_label = "masks"
    data = getCompareData(PATH_BASE,  types = data_groups , file = '../../Data/打針資料.xlsx')
    setFloder('./record/'+disease + '_' + date + '/')
    for data_group in data_groups:
        patient = data.getData(disease, data_group, dict_origin[data_group], data_label)
        json_file = './record/'+ disease + '_' + date + '/'+ data_group + '_Treatment.json'
        tools.write_to_json_file(json_file, patient)


if __name__ == '__main__':
    run()   