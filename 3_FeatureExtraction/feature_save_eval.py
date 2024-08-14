import pandas as pd
import csv
import matplotlib.pyplot as plt
import os
import json
from collections import OrderedDict
def feature_save_eval(file, label_file, sheet, outputfile, save_classification = True,save_regression = True,ROI = True,cut = True):
    eyes_dict = {'OD': 'R', 'OS': 'L'}
    #label_file = '../../Data/打針資料.xlsx'
    #sheet = 'Focea_collect'
    #outputfile = 'classification.csv'
    label = pd.read_excel(label_file, sheet_name = sheet, engine='openpyxl')
    label['病歷號'] = pd.to_numeric(label['病歷號'], errors='coerce')
    label['病歷號'] = label['病歷號'].astype(int).apply(lambda x: '{:08d}'.format(x))
    
    # eyes_dict = {'OD': 'R', 'OS': 'L'}
    label['眼睛'] = label['眼睛'].map(eyes_dict)
    label['眼睛'] = label['眼睛'].astype(str)
    
    
    feature_relative = json.load(open(file))
    # feature total
    index0 = feature_relative[list(feature_relative.keys())[0]]
    key_count = len(index0)
    print("feature count: ", key_count)
    
    classification_patient = pd.DataFrame()
    count = 0
    good = 0
    bad = 0
    Thick_relative_list = []
    for index, row in label.iterrows():
        patient = row['病歷號']
        eye = row['眼睛']
        if pd.notna(row['針前黃斑部厚度']) and pd.notna(row['三針後黃斑部厚度']):
            # print(row['針前黃斑部厚度'],type(row['針前黃斑部厚度']))
            if 'cf' in str(row['三針後視力']) or 'cf' in str(row['打針前視力']) or 'HM' in str(row['三針後視力']) or 'HM' in str(row['打針前視力']):
                continue
            if patient + '_' + eye in feature_relative:
                Thick_relative = (float(row['三針後黃斑部厚度']) - float(row['針前黃斑部厚度']))/(float(row['針前黃斑部厚度'])-200) * 100
                vision_relative = (float(row['三針後視力']) - float(row['打針前視力']))/abs(float(row['打針前視力'])) * 100
                count += 1
                tmp_dict = OrderedDict()
                tmp_dict['patient'] = patient+'_'+eye
                    
                if save_regression:
                    regression = round(Thick_relative,0)
                    regression = int(regression)
                    tmp_dict['regression'] = Thick_relative
                    tmp_dict['vision'] = vision_relative
                Thick_relative_list.append(Thick_relative)
                if save_classification :
                    if Thick_relative <-22.14:
                        classificate = 1
                        good += 1
                    else:
                        classificate = 0
                        bad += 1
                    tmp_dict['classification'] = classificate
                    
                
                    tmp_dict.update(feature_relative[patient + '_' + eye])
                    
                    # 調整列的順序
                    # patient, classificate, feature1, feature2, feature3, ...
                    tmp_df = pd.DataFrame([tmp_dict])
                    classification_patient = pd.concat([classification_patient, tmp_df])
    
    # print Q3
    Thick_relative_list = pd.Series(Thick_relative_list)
    print('Q3:', Thick_relative_list.quantile(0.75))
    print('Q1:', Thick_relative_list.quantile(0.25))
    print('IQR:', Thick_relative_list.quantile(0.75) - Thick_relative_list.quantile(0.25))
    print('good:', good)
    print('bad:', bad)
    print('count:', count)
    print('good rate:', good/count)
    print('bad rate:', bad/count)
    print('good mean:', Thick_relative_list[Thick_relative_list < -10].mean())
    print('bad mean:', Thick_relative_list[Thick_relative_list >= -10].mean())

    
    # Save the classification_patient to a csv file
    if save_classification:
        classification_patient_save = classification_patient.copy()
        del classification_patient_save['regression']
        file_name = outputfile + 'classification'
            
        file_name = file_name +  '_cut' if cut else file_name
        file_name = file_name + '_ROI.csv' if ROI else file_name + '.csv'
        classification_patient_save.to_csv(file_name, index=False)
        
    if save_regression:
        classification_patient_save = classification_patient.copy()
        del classification_patient_save['classification']
        file_name = outputfile + 'regression'
        file_name = file_name +  '_cut' if cut else file_name
        file_name = file_name + '_ROI.csv' if ROI else file_name + '.csv'
        classification_patient_save.to_csv(file_name, index=False)

    

if __name__ == '__main__':
    disease   = 'PCV'
    date    = '20240418'
    PATH_BASE    = "../../Data/" + disease + '_' + date + '/'
    data_groups  = ["CC"]
    dict_origin  = {'CC': "4"}
    ROI = True
    cut = False
    file_name = 'VesselFeature_relative'
    if cut :
         file_name = file_name + '_cut'
    if ROI:
        file_name = file_name + '_ROI'
    
    file = './record/' + disease + '_' + date +'/' + file_name + '.json'
    
    label_file = '../../Data/打針資料.xlsx'
    sheet = 'Focea_collect'
    
    outputfile = './record/' + disease + '_' + date +'/'
    #feature combination
    feature_save_eval(file, label_file, sheet, outputfile,ROI = ROI,cut = cut)
    




