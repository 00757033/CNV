import pandas as pd
import csv
import matplotlib.pyplot as plt
import os
import json
from collections import OrderedDict
def feature_save_eval(file, label_file, sheet, outputfile):
    eyes_dict = {'OD': 'R', 'OS': 'L'}
    #label_file = '../../Data/打針資料.xlsx'
    #sheet = 'Focea_collect'
    #outputfile = 'classification.csv'
    label = pd.read_excel(label_file, sheet_name = sheet, engine='openpyxl')
    label['病歷號'] = pd.to_numeric(label['病歷號'], errors='coerce')
    label['病歷號'] = label['病歷號'].apply(lambda x: '{:08d}'.format(x))
    
    # eyes_dict = {'OD': 'R', 'OS': 'L'}
    label['眼睛'] = label['眼睛'].map(eyes_dict)
    label['眼睛'] = label['眼睛'].astype(str)
    
    
    feature = json.load(open(file))
    # feature total
    index0 = feature[list(feature.keys())[0]]
    print(index0)
    key_count = len(index0)
    print("feature count: ", key_count)
    
    classification_patient = pd.DataFrame()
    count = 0
    for index, row in label.iterrows():
        patient = row['病歷號']
        eye = row['眼睛']
        if pd.notna(row['針前黃斑部厚度']) and pd.notna(row['三針後黃斑部厚度']):
            Thick_relative = (float(row['三針後黃斑部厚度']) - float(row['針前黃斑部厚度']))/float(row['針前黃斑部厚度']) * 100
            # print(Thick_relative)
            if Thick_relative < -5:
                classificate = 1
            else:
                classificate = 0
                
            if patient + '_' + eye in feature_relative:
                count += 1
                tmp_dict = OrderedDict()
                tmp_dict['patient'] = patient+'_'+eye
                tmp_dict['classification'] = classificate
                tmp_dict.update(feature_relative[patient + '_' + eye])
                
                # 調整列的順序
                # patient, classificate, feature1, feature2, feature3, ...
                tmp_df = pd.DataFrame([tmp_dict])
                classification_patient = pd.concat([classification_patient, tmp_df])
                
    
    print(count)
    
    # Save the classification_patient to a csv file
    
    classification_patient.to_csv(outputfile, index=False)

    

if __name__ == '__main__':
    disease   = 'PCV'
    date    = '20240320'
    PATH_BASE    = "../../Data/" + disease + '_' + date + '/'
    data_groups  = ["CC"]
    dict_origin  = {'CC': "4"}
    
    file = './record/' + disease + '_' + date +'/' + 'VesselFeature.json'
    
    label_file = '../../Data/打針資料.xlsx'
    sheet = 'Focea_collect'
    
    outputfile = './record/' + disease + '_' + date +'/'+'classification.csv'
    if not os.path.exists('./record/' + 'classification/'):
        os.makedirs('./record/' + 'classification/')
    #feature combination
    feature_save_eval(file, label_file, sheet, outputfile)
    
