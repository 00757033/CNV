import pandas as pd
import csv
from csv import reader
import matplotlib.pyplot as plt
date = '0831'
layer = 'OR'

#feature = pd.read_csv(date + '_feature_' + layer + '.csv')
#print(feature[:][0])
GLCM = pd.read_csv('../3_FeatureExtraction/GLCM' + '.csv')
print(GLCM.head())

#GLCM特徵
with open('../3_FeatureExtraction/GLCM.csv', 'r') as csv_file:
    csv_reader = reader(csv_file)
    # Passing the cav_reader object to list() to get a list of lists
    GLCM_list = list(csv_reader)
#Vessel feature
with open('../3_FeatureExtraction/' + date + '_VesselFeature_' + layer + '.csv', 'r') as csv_file:
    csv_reader = reader(csv_file)
    # Passing the cav_reader object to list() to get a list of lists
    vessel_list = list(csv_reader)
    vessel_list = vessel_list[1:]

    for i in range(len(vessel_list)):
        temp = []
        vd  = (float(vessel_list[i][2])-float(vessel_list[i][1]))/float(vessel_list[i][1])
        vld = (float(vessel_list[i][4])-float(vessel_list[i][3]))/float(vessel_list[i][3])
        vdi = (float(vessel_list[i][6])-float(vessel_list[i][5]))/float(vessel_list[i][5])
        vessel_list[i] = [vd, vld, vdi]

#feature combination
total_features = []
GLCM_features   = []
Selection_features   = []
for i in range(len(vessel_list)):
    total_feature = GLCM_list[i] + vessel_list[i]
    GLCM_feature = GLCM_list[i]
    GLCM_features.append(GLCM_feature)
    total_features.append(total_feature)

patient_dict = {}

for patient in GLCM_features:
    print(patient[0])
    patient_dict[patient[0]] = patient[2:]

#biomarker feature
path_biomarker = '../../Data/biomarker_list.csv'
with open(path_biomarker, 'r') as csv_file:
    csv_reader = reader(csv_file)
    biomarkers = list(csv_reader)

fianl_features = []
diff_ratio = []
diff = []
#合併生物標籤跟影像紋理特徵
for biomarker in biomarkers[1:]:
    print(biomarker[0])
    diff_ratio.append((int(biomarker[11])-int(biomarker[10]))/int(biomarker[10]))
    diff.append((int(biomarker[11])-int(biomarker[10])))
    if biomarker[0] in patient_dict:
        feature = patient_dict[biomarker[0]] + biomarker[10:12]
        
        print(feature)
        fianl_features.append(feature)
plt.hist(diff_ratio,bins = 20, edgecolor = 'black', color='grey')
plt.ylabel("Number of patient", fontsize=17, fontweight='bold', fontfamily = 'Times New Roman')
plt.xlabel("Relative change of CMT", fontsize=17, fontweight='bold', fontfamily = 'Times New Roman')
plt.xticks(fontsize=12, fontweight='bold', fontfamily = 'Times New Roman')
plt.yticks(fontsize=12, fontweight='bold', fontfamily = 'Times New Roman')
plt.show()
#將最終特徵輸出feature_final.csv
with open('feature_final.csv', 'w', newline = '') as file:
    writer = csv.writer(file, quoting=csv.QUOTE_ALL, delimiter=',')
    writer.writerows(fianl_features)
