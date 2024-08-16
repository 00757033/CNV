import pathlib as pl
from collections import Counter
import pandas as pd
import os
import re
import shutil
import sys
import os
from datetime import datetime


# # 取得 my_project 資料夾的絕對路徑
# my_project_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# # 將 my_project 資料夾的路徑添加到 Python 解釋器的搜索路徑
# sys.path.append(my_project_path)
import tools 

class Analyze():
    def __init__(self,data_path = '../../Data/OCTA',layers = {"3":"OR","4":"CC"},layer_list = ['1','2','3','4','6','7']):
        self.data_path = data_path
        self.layers = layers
        self.layer_list = layer_list

    # read disease data
    def disease(self,file = '../../Data/OCTa預收名單.xlsx',label = ["申請診斷","患者","病歷號","患眼"]):
        self.disease_df = pd.DataFrame()
        data = pd.read_excel(file, sheet_name=None,na_filter = False, engine='openpyxl')
        sheet = pd.ExcelFile(file, engine='openpyxl')
        for s_name in sheet.sheet_names:
            if s_name !='待確認':
                self.disease_df = self.disease_df.append(data[s_name], ignore_index=True)
        # 刪除在 ['病歷號','患眼','申請診斷'] 有空值的 row 只要有一個欄位有空值就刪除
        for column in ['病歷號','患眼','申請診斷']:
            self.disease_df = self.disease_df[self.disease_df[column] != '']
            
        self.disease_df.drop_duplicates(subset=["病歷號","患眼"], keep='last', inplace=True)
        self.disease_df = self.disease_df.reset_index(drop=True)
        self.disease_df['病歷號'] = pd.to_numeric(self.disease_df['病歷號'], errors='coerce')
        self.disease_df['病歷號'] = self.disease_df['病歷號'].fillna(0).astype(int)
        self.disease_df['病歷號'] = self.disease_df['病歷號'].apply(lambda x: '{:08d}'.format(x))
        self.disease_df = self.disease_df.sort_values(by=["病歷號"], ascending=True)
        # 保留 ['申請診斷','醫師','病歷號','患眼','年齡','性別','患眼']

        self.disease_df = self.disease_df [['申請診斷','醫師','病歷號','患眼','年齡','性別','患眼']]
        # save self.disease_df
       
    def save_disease(self,path = './record/disease.csv'):
        self.disease_df.to_csv(path,index = False)
    # which type of disease
    def patient_PCV_AMD(self):
        AMD = dict()
        PCV = dict()
        other = dict()
        patient = dict()
        for index, row in self.disease_df.iterrows():
            diagnosis = str(row['申請診斷'])
            patient_id = row['病歷號']
            eyes = row["患眼"].values
            patient_eye = set()
            for eye in eyes:
                if "OD" in eye:
                    patient_eye.add("OD")
                if "OS" in eye:
                    patient_eye.add("OS")
                if "OU" in eye:
                    patient_eye.add("OD")
                    patient_eye.add("OS")

            category_dict = {
                "AMD": AMD,
                "PCV": PCV,
                "other": other,
                "patient": patient
            }
            for key, value in category_dict.items():
                if key in  diagnosis :
                    if patient_id in value.keys():
                        value[patient_id] = value[patient_id] | patient_eye
                    else:
                        value[patient_id] = patient_eye

        return AMD, PCV, other, patient
    
    def inject(self,file = '../../Data/打針資料.xlsx',label = ["診斷","病歷號","眼睛","打針前門診日期","三針後門診","六針後門診","九針後門診","十二針後門診"]):
        # self.inject_df = pd.DataFrame()
        self.inject_df = pd.read_excel(file, sheet_name="20230830",na_filter = False, engine='openpyxl')
        self.inject_df['病歷號'] = self.inject_df['病歷號'].apply(lambda x: '{:08d}'.format(x))
        # turn to json
        list_date = {'打針前門診日期': 'pre-injection', '三針後門診': '1st-injection', '六針後門診': '2nd-injection', '九針後門診': '3rd-injection', '十二針後門診': '4th-injection'}
        inject_dict = dict()
        for index, row in self.inject_df.iterrows():
            patient_id = row['病歷號']
            if patient_id not in inject_dict.keys():
                inject_dict[patient_id] = dict()
            disease_eyes = row['眼睛'].split(',')
            for eye in disease_eyes:
                if eye not in inject_dict[patient_id].keys():
                    inject_dict[patient_id][eye] = dict()
                # if date is Nat then set '' else turn to yearmonthday
                
                for date , date_name in list_date.items():
                    if pd.isnull(row[date]) or len(str(row[date])) < 8:
                        row[date] = ''
                    else:
                        inject_dict[patient_id][eye][date_name] = pd.to_datetime(row[date]).strftime('%Y%m%d')
        # count inject_dict patient+eye
        count = 0
        for patient_id, eyes in inject_dict.items():
            for eye in eyes:
                if list_date['打針前門診日期'] in inject_dict[patient_id][eye] and  list_date['三針後門診'] in inject_dict[patient_id][eye] :
                    count += 1

        print('inject case',count)
        
        return inject_dict
                        
    def find_missing_records(self,data):
        missing_records = dict()
        for patient_id, eyes in data.items():
            if patient_id in self.inject_df['病歷號'].values:
                for eye in eyes:
                    if eye  in  self.inject_df[self.inject_df['病歷號'] == patient_id]['眼睛'].values:
                        continue
                    else:
                        if patient_id not in missing_records.keys():
                            missing_records[patient_id] = set()
                        missing_records[patient_id].add(eye)
            else:
                if patient_id not in missing_records.keys():
                    missing_records[patient_id] = set()
                missing_records[patient_id] = eyes
        tools.pop_empty_dict(missing_records)
        
        # count missing_records patient+eye
        count = 0
        for patient_id, eyes in missing_records.items():
            count += len(eyes)
        print('missing records',count)
        return missing_records
      
    def check_date(self, patient_in_list,patient_in_folder): # patient_in_list 病患資料表中的病患  patient_in_folder 病歷表中的病患
        # 病歷表中的病患 
        eye_change = {'OD':'R','OS':'L'}
        patient = dict()
        found_patient = dict()
        for patient_id in patient_in_list:
            if patient_id in patient_in_folder.keys():
                print(patient_in_folder[patient_id])
            
        return patient,found_patient

    def read_file(self,path = '../../Data/OCTA'):
        need_collect = dict()
        for patientID in pl.Path(path).iterdir():
            for date in patientID.iterdir():
                for OD_OS in date.iterdir():
                    if not OD_OS.is_dir():
                        os.remove(OD_OS)
                    if OD_OS.is_dir() and len(list(OD_OS.iterdir())) < 14:
                        for layer in self.layer_list:
                            if not os.path.exists(self.data_path + '/' + patientID.name + '/' + date.name + '/' + OD_OS.name + '/' + layer + '.png') and not os.path.exists(self.data_path + '/' + patientID.name + '/' + date.name + '/' + OD_OS.name + '/' + layer + '.jpg'):
                                if patientID.name not in need_collect.keys():
                                    need_collect[patientID.name] = dict()
                                if date.name not in need_collect[patientID.name].keys():
                                    need_collect[patientID.name][date.name] = dict()
                                if OD_OS.name not in need_collect[patientID.name][date.name].keys():
                                    need_collect[patientID.name][date.name][OD_OS.name] = list()
                                need_collect[patientID.name][date.name][OD_OS.name].append(layer)
                                
        tools.pop_empty_dict(need_collect)
        return need_collect
                                                                 
    # Uncollected patients
    def Uncollected_patients(self,patient_in_excell_list,patient_in_folder,inject_dict):
        uncollect_patient = dict()
        collect_patient = dict()
        eyes_change = {'OD':'R','OS':'L'}
        for patient_id in patient_in_excell_list:
            if patient_id not in patient_in_folder.keys(): # add all date
                if patient_id not in uncollect_patient.keys():
                    uncollect_patient[patient_id] = dict()
                if patient_id in inject_dict.keys(): # add all eye
                    for eye in inject_dict[patient_id]:
                        for date in inject_dict[patient_id][eye].values():
                            if date not in uncollect_patient[patient_id].keys():
                                uncollect_patient[patient_id][date] = list()
                            if eye not in uncollect_patient[patient_id][date]:
                                uncollect_patient[patient_id][date].append(eyes_change[eye])
                            
            else :
                collect_patient[patient_id] = dict()
                if patient_id in inject_dict.keys():
                    for eye in inject_dict[patient_id]:
                        for date in inject_dict[patient_id][eye].values():
                            if date not in patient_in_folder[patient_id].keys():
                                if patient_id not in uncollect_patient.keys():
                                    uncollect_patient[patient_id] = dict()
                                if date not in uncollect_patient[patient_id].keys():
                                    uncollect_patient[patient_id][date] = list()
                                if eye not in uncollect_patient[patient_id][date]:
                                    uncollect_patient[patient_id][date].append(eyes_change[eye])
                            else:
                                collect_patient[patient_id][date] = list()

                                if eyes_change[eye] not in patient_in_folder[patient_id][date].keys():
                                    if patient_id not in uncollect_patient.keys():
                                        uncollect_patient[patient_id] = dict()
                                    if date not in uncollect_patient[patient_id].keys():
                                        uncollect_patient[patient_id][date] = list()
                                    if eye not in uncollect_patient[patient_id][date]:
                                        uncollect_patient[patient_id][date].append(eyes_change[eye])
                                else:
                                    # if 3 4 layer not exist
                                    if '3' not in patient_in_folder[patient_id][date][eyes_change[eye]] or '4' not in patient_in_folder[patient_id][date][eyes_change[eye]]:
                                        if patient_id not in uncollect_patient.keys():
                                            uncollect_patient[patient_id] = dict()
                                        if date not in uncollect_patient[patient_id].keys():
                                            uncollect_patient[patient_id][date] = list()
                                        if eye not in uncollect_patient[patient_id][date]:
                                            uncollect_patient[patient_id][date].append(eyes_change[eye])
                                    else:
                                        collect_patient[patient_id][date].append(eyes_change[eye])
                                
                
        tools.pop_empty_dict(uncollect_patient)
        
        # count uncollect_patient patient+date+eye
        count = 0
        for patient_id, dates in uncollect_patient.items():
            for date, eyes in dates.items():
                count += len(eyes)
        print('uncollect patient',count)
        return uncollect_patient,collect_patient
 
    def disease_txt(dict_AMD, dict_PCV,dict_other,file_name = 'AMD_PCV.txt'):
        tools.remove_exist_file(file_name)
        tools.patientID_to_txt_file(file_name,dict_AMD,'AMD',True)
        tools.patientID_to_txt_file(file_name,dict_PCV,'PCV',True)
        tools.patientID_to_txt_file(file_name,dict_other,'other')
        
    # Convert  patient data into a dictionary.
    def file_patient(self,path = '../../Data/OCTA'):
        patient = dict()

        for patientID in pl.Path(path).iterdir():
            patient[patientID.name] = dict()
            for date in patientID.iterdir():
                patient[patientID.name][date.name] = dict()
                for OD_OS in date.iterdir():
                    patient[patientID.name][date.name][OD_OS.name] = list()
                    for layer in OD_OS.iterdir():
                        if layer.is_dir():
                            continue
                        else:
                            patient[patientID.name][date.name][OD_OS.name].append(layer.stem)
        tools.pop_empty_dict(patient)
        print('patient',len(patient))
        return patient

    # check patient data is exist or not
    def check(self,patientID_list,patientID_in_exell):
        patient_in_file = self.file_patient(patientID_list)
        not_exist = list()
        exist = list()
        for patientID in patientID_in_exell:
            if patientID not in patient_in_file:
                not_exist.append(patientID)
            else:
                exist.append(patientID)

        return  exist,not_exist

    # check patient data is exist or not
    def label(self,patient_disease,label_path = '../../Data/20240311_label',label_name = 'label_',label_layer = ['3','4']):
        islabeled = dict()
        unlabel = dict()
        count = 0
        for patientID in patient_disease:
            islabeled[patientID] = dict()
            unlabel[patientID] = dict()
            for date in patient_disease[patientID]:
                islabeled[patientID][date] = dict()
                unlabel[patientID][date] = dict()
                for OD_OS in patient_disease[patientID][date]:
                    islabeled[patientID][date][OD_OS] = list()
                    unlabel[patientID][date][OD_OS] = list()
                    if os.path.exists(label_path + '/' + patientID + '/' + date + '/' + OD_OS):
                        check = True
                        for layer in label_layer:
                            if os.path.exists(label_path + '/' + patientID + '/' + date + '/' + OD_OS + '/' + label_name + layer + '.png'):
                                islabeled[patientID][date][OD_OS].append(layer)
                            else:
                                unlabel[patientID][date][OD_OS].append(layer)
                                check = False
                        if check:
                            count += 1
                            del unlabel[patientID][date][OD_OS]
                    else:
                        unlabel[patientID][date][OD_OS] = label_layer
        tools.pop_empty_dict(islabeled)
        tools.pop_empty_dict(unlabel)
        
        
        print('labeled',count)
       
        return islabeled,unlabel

    # copy need label data to unlabel folder  
    def need_label(self,need_label_dict,label_name,disease_name = 'PCV',unlabel_path = '../../Data/need_label3',label_layer = ['4']):
        for layer in label_layer:
            if not os.path.exists(unlabel_path + '/' + label_name + '/' +layer):
                os.makedirs(unlabel_path + '/' + label_name + '/' +layer)
        for patientID in need_label_dict:
            for date in need_label_dict[patientID]:
                for OD_OS in need_label_dict[patientID][date]:
                    for files in need_label_dict[patientID][date][OD_OS]:
                        print(self.data_path + '/' + patientID + '/' + date + '/' + OD_OS + '/' + files + '.png')
                        print(unlabel_path + '/' + disease_name + '/' +files + '/' + patientID + '_' + OD_OS + '_' + date + '.png')
                        shutil.copyfile(self.data_path + '/' + patientID + '/' + date + '/' + OD_OS + '/' + files + '.png',unlabel_path + '/' + disease_name + '/' +files + '/' + patientID + '_' + OD_OS + '_' + date + '.png')
                    
    # Counting patient visit frequencies in list
    def count_patient_visits(self,patientID_list,patient_dict):
        # count by patient visit
        OD_count = Counter()
        OS_count = Counter()
        total_L = 0
        total_R = 0
        # patientID => Patient's follow-up date  => Folders for Left and Right Eyes => image file
        for patient_id, patient_data in patient_dict.items():
            L_count = 0
            R_count = 0
            for date , patient_data in patient_data.items():
                for eyes , patient_eyes in patient_data.items():
                    if eyes == 'R' and '3' in patient_eyes and '4' in patient_eyes:
                        R_count += 1
                    elif eyes == 'L' and '3' in patient_eyes and '4' in patient_eyes:
                        L_count += 1
            if L_count : 
                OS_count[L_count] += 1
            if R_count : 
                OD_count[R_count] += 1
            total_L += L_count
            total_R += R_count
        return OD_count,OS_count , total_R , total_L

def sort_by_patient_id(item):
    return item[0]                         


if __name__ == '__main__':
    analyze = Analyze()
    # 收集病患資料
    analyze.disease()
    tools.makefolder('./record')
    # save
    analyze.save_disease()
    need_collect = analyze.read_file()
    # 分類病患資料
    AMD, PCV, other, patient = analyze.patient_PCV_AMD()
    print('AMD',len(AMD))
    print('PCV',len(PCV))
    print('other',len(other))
    print('patient',len(patient))
    
    
    
    # 病患資料表中的PCV病患
    tools.remove_exist_file('./record/PCV.txt')
    tools.txt_to_file(PCV,'record','PCV')

    inject = analyze.inject('../../Data/打針資料.xlsx')
    tools.remove_exist_file('./record/inject.json')
    tools.write_to_json_file('./record/inject.json',inject)
    
    # 結合病患資料與打針資料 並找出未紀錄打針資料
    missing_records = analyze.find_missing_records(PCV)
    tools.remove_exist_file('./record/unrecord_inject.txt')
    with open('./record/unrecord_inject.txt', 'w') as f:
        for key, value in missing_records.items():
            f.write('%s:%s\n' % (key, value))
    

    # 全部已收集資料
    patient_collected = analyze.file_patient()

    tools.remove_exist_file('./record/collected.json')
    tools.write_to_json_file('./record/collected.json',patient_collected)
    
    
    # 未收集病患資料&日期
    not_collect,collect_patient = analyze.Uncollected_patients(PCV,patient_collected,inject)
    
    tools.remove_exist_file('./record/uncollect_PCV.txt')
    with open('./record/uncollect_PCV.txt', 'w') as f:
        for key, value in not_collect.items():
            f.write('%s:%s\n' % (key, value))


    tools.remove_exist_file('./record/collect_patient.json')
    tools.write_to_json_file('./record/collect_patient.json',collect_patient)
    
    # need label
    islabeled,unlabel = analyze.label(collect_patient,'../../Data/20240311_label','label_',['4'])
    
    tools.remove_exist_file('./record/islabeled.json')
    tools.write_to_json_file('./record/islabeled.json',islabeled)
    
    tools.remove_exist_file('./record/unlabel.json')
    tools.write_to_json_file('./record/unlabel.json',unlabel)
    
    # # need label
    need_label = analyze.need_label(unlabel,'PCV')
    
    tools.remove_exist_file('./record/need_label.json')
    tools.write_to_json_file('./record/need_label.json',need_label)
    
    
    




