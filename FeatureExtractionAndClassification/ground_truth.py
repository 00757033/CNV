import pandas as pd
import csv
import matplotlib.pyplot as plt
import os
import json
from collections import OrderedDict
from scipy import stats
from bootstrapped import bootstrap as bs
from bootstrapped import stats_functions as bs_stats
import seaborn as sns
def merge_table(df1, df2, outputfile):
 
    df = pd.merge(df1, df2, on=['病歷號', '眼睛'], how='inner')
    df.to_csv(outputfile, index=False,encoding='utf-8-sig')
    return df
def csv_to_pandas(file,sheet):
    df = pd.read_excel(file, sheet_name = sheet, engine='openpyxl')
    return df
   
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
    foundation = 246.1
    Thick_relative_list = []
    for index, row in label.iterrows():
        print('row',row['病歷號'],row['眼睛'])
        patient = row['病歷號']
        eye = row['眼睛']
        if pd.notna(row['針前黃斑部厚度']) and pd.notna(row['三針後黃斑部厚度']):
            # print(row['針前黃斑部厚度'],type(row['針前黃斑部厚度']))
            if 'cf' in str(row['三針後視力']) or 'cf' in str(row['打針前視力']) or 'HM' in str(row['三針後視力']) or 'HM' in str(row['打針前視力']):
                continue
            if patient + '_' + eye in feature_relative:
                smooth = 0.0001
                
                pre_thick = abs(float(row['針前黃斑部厚度']- foundation))
                post_thick = abs(float(row['三針後黃斑部厚度']- foundation))  
                print(pre_thick , post_thick,float(row['針前黃斑部厚度']) -foundation)
               
                Thick_relative = (post_thick - pre_thick) /(pre_thick + smooth) * 100
                print(patient + '_' + eye, Thick_relative)
                vision_relative = (float(row['三針後視力']) - float(row['打針前視力']))/abs(float(row['打針前視力'])) * 100
                count += 1
                tmp_dict = OrderedDict()
                tmp_dict['patient'] = patient+'_'+eye
                    
                
                Thick_relative_list.append(Thick_relative)
    # print Q3
    Thick_relative_list = pd.Series(Thick_relative_list)
    print('Max:', round(Thick_relative_list.max(),3))
    print('Min:', round(Thick_relative_list.min(),3))
    print('Q3:', round(Thick_relative_list.quantile(0.75),3))
    print('Q2:', round(Thick_relative_list.quantile(0.5),3))
    print('Q1:', round(Thick_relative_list.quantile(0.25),3))
    print('IQR:', round(Thick_relative_list.quantile(0.75) - Thick_relative_list.quantile(0.25),3))    
    # 计算均值和标准差
    print('mean:', Thick_relative_list.mean())
    print('std:', Thick_relative_list.std())
    # 中位數
    print('median:', Thick_relative_list.median())
    # 计算众数
    print('mode:', Thick_relative_list.mode())
    # 确定置信水平为 95%
    alpha = 0.997
    # 计算置信区间
    Confidence_Interval = stats.t.interval(alpha, len(Thick_relative_list)-1, Thick_relative_list.mean(), stats.sem(Thick_relative_list))
    print('Confidence_Interval:', Confidence_Interval)
    # Bootstrap方法估計信賴區間 3 代表信心水平為 99.7%
    ci = bs.bootstrap(Thick_relative_list.values, stat_func=bs_stats.mean, alpha=0.95)
    print('Bootstrap:', ci, ci.value, ci.lower_bound, ci.upper_bound)
    Q1 = round(Thick_relative_list.quantile(0.25),2)
    Q3 = round(Thick_relative_list.quantile(0.75),2)
    # # df["col"].skew()
    print('skew:', Thick_relative_list.skew())
    # 绘制直方图
    min_val = round(min(Thick_relative_list), -1)
    max_val = round(max(Thick_relative_list), -1)
    sns.histplot(Thick_relative_list, kde=True, bins= [i for i in range(round(min_val), round(max_val) , 5)], binwidth = 10)
    plt.xlabel('Thick Relative Change (%)')
    plt.xlim(min(Thick_relative_list), max(Thick_relative_list))
    plt.ylim(0, 10)
    plt.title('Thick Relative Change Distribution')
    # show Q1 Q2 Q3 IQR
    plt.axvline(Thick_relative_list.quantile(0.25), color='r', linestyle='--', label=f'Q1: {round(Thick_relative_list.quantile(0.25),2)}')
    plt.axvline(Thick_relative_list.quantile(0.5), color='g', linestyle='--', label= f'Q2: {round(Thick_relative_list.quantile(0.5),2)}')
    plt.axvline(Thick_relative_list.quantile(0.75), color='b', linestyle='--', label= f'Q3: {round(Thick_relative_list.quantile(0.75),2)}')
    plt.legend()
    gound_turth = {'poor':0, 'general':0, 'good':0}
    # # show mean std  mode
    # plt.axvline(Thick_relative_list.mean(), color='y', linestyle='--', label=f'mean: {round(Thick_relative_list.mean(),2)}')
    # plt.axvline(Thick_relative_list.mode()[0], color='m', linestyle='--', label=f'mode: {round(Thick_relative_list.mode()[0],2)}')
    # plt.legend()
    
    plt.show()
    for index, row in label.iterrows():
            patient = row['病歷號']
            eye = row['眼睛']
            if pd.notna(row['針前黃斑部厚度']) and pd.notna(row['三針後黃斑部厚度']):
                # print(row['針前黃斑部厚度'],type(row['針前黃斑部厚度']))
                if 'cf' in str(row['三針後視力']) or 'cf' in str(row['打針前視力']) or 'HM' in str(row['三針後視力']) or 'HM' in str(row['打針前視力']):
                    continue
                if patient + '_' + eye in feature_relative:
                    pre_thick = abs(float(row['針前黃斑部厚度']- foundation))
                    post_thick = abs(float(row['三針後黃斑部厚度']- foundation))  
                        
                    Thick_relative = (post_thick - pre_thick) /(pre_thick + smooth) * 100   
                    
                    vision_relative = (float(row['三針後視力']) - float(row['打針前視力']))/abs(float(row['打針前視力'])) * 100
                    count += 1
                    tmp_dict = OrderedDict()
                    tmp_dict['patient'] = patient+'_'+eye
                        
                    if save_regression:
                        regression = round(Thick_relative,0)
                        regression = int(regression)
                        tmp_dict['regression'] = Thick_relative
                        tmp_dict['vision'] = vision_relative
                    
                    if save_classification :
                        # if Thick_relative < Q1 :
                        # # if Thick_relative < round(Confidence_Interval[0],2):
                        #     classificate = 2
                        #     gound_turth['good'] += 1
                        # elif Thick_relative > Q3:
                        if Thick_relative > Q3 :
                        # elif Thick_relative > round(Confidence_Interval[1],2):
                            classificate = 0
                            gound_turth['poor'] += 1
                        else:
                            classificate = 1
                            gound_turth['general'] += 1
                            
                            
                        tmp_dict['classification'] = classificate
                        
                    
                        tmp_dict.update(feature_relative[patient + '_' + eye])
                        
                        # 調整列的順序
                        # patient, classificate, feature1, feature2, feature3, ...
                        tmp_df = pd.DataFrame([tmp_dict])
                        classification_patient = pd.concat([classification_patient, tmp_df])    
    print('label:', gound_turth)
    print('count:', count)
   
    
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

def feature_save_eval2(file,label, outputfile, save_classification = True,ROI = True,cut = True):
    eyes_dict = {'OD': 'R', 'OS': 'L'}
    #label_file = '../../Data/打針資料.xlsx'
    #sheet = 'Focea_collect'
    #outputfile = 'classification.csv'

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
    gound_turth = {'poor':0, 'general':0, 'good':0}
    classification_patient = pd.DataFrame()
    count = 0
    foundation = 246.1
    Thick_relative_list = []
    for index, row in label.iterrows():
        print('row',row['病歷號'],row['眼睛'])
        patient = row['病歷號']
        eye = row['眼睛']
        if pd.notna(row['Baseline BCVA (logMAR)']) and pd.notna(row['Post BCVA (logMAR)']):
            # print(row['針前黃斑部厚度'],type(row['針前黃斑部厚度']))
            # if 'cf' in str(row['三針後視力']) or 'cf' in str(row['打針前視力']) or 'HM' in str(row['三針後視力']) or 'HM' in str(row['打針前視力']):
            #     continue
            tmp_dict = OrderedDict()
            tmp_dict['patient'] = patient+'_'+eye
            
            if patient + '_' + eye in feature_relative:
                smooth = 0.0001
                if save_classification :
                    if float(row['Post BCVA (logMAR)']) > float(row['Baseline BCVA (logMAR)']) :
                    
                        gound_turth['good'] += 1
                        classificate = 1
                    else:
                        classificate = 0
                        gound_turth['poor'] += 1
                            
                            
                    tmp_dict['classification'] = classificate
                    tmp_dict['Age'] = row['Age']
                    tmp_dict['Gender'] = 1 if row['M / F'] == 'M' else 0
                    # tmp_dict['CMT'] = (row['Baseline CMT'] - row['Post CMT']) / row['Baseline CMT'] * 100
                    # tmp_dict['SFCT'] = (row['Baseline SFCT'] - row['Post SFCT']) / row['Baseline SFCT'] * 100
                    tmp_dict.update(feature_relative[patient + '_' + eye])
                    
                    # 調整列的順序
                    # patient, classificate, feature1, feature2, feature3, ...
                    tmp_df = pd.DataFrame([tmp_dict])
                    classification_patient = pd.concat([classification_patient, tmp_df])    
                    
    print('label:', gound_turth)
    print('count:', count)
   
    
    # Save the classification_patient to a csv file
    if save_classification:
        classification_patient_save = classification_patient.copy()
        file_name = outputfile 
        print(file_name)    

        classification_patient_save.to_csv(file_name, index=False)
        

if __name__ == '__main__':
    disease   = 'PCV'
    date    = '20240524'
    PATH_BASE    = "../../Data/" + disease + '_' + date + '/'
    data_groups  = ["CC"]
    dict_origin  = {'CC': "4"}
    ROI = True
    file_name = 'VesselFeature_relative'
    if ROI:
        file_name = file_name + '_ROI'
    os.makedirs('./record/' + disease + '_' + date, exist_ok=True)
    file = './record/' + disease + '_' + date +'/' + file_name + '.json'
    
    label_file = '../../Data/打針資料.xlsx'
    sheet = 'Focea_collect'
    
    df1 = csv_to_pandas(label_file,sheet)
    df1 = df1[['病歷號', '眼睛', '打針前門診日期','針前黃斑部厚度', '三針後黃斑部厚度', '打針前視力', '三針後門診','三針後視力','三針後黃斑部厚度']]
    label_file2 =  '../../Data/PCV data_0612.xlsx'
    sheet2 = 'Enrolled'
    
    df2 = csv_to_pandas(label_file2,sheet2)
    df2 = df2.drop(columns = ['NO.'] )
    merge_df = merge_table(df1, df2, './record/' + disease + '_' + date +'/merge.csv')
    

    
    outputfile = './record/' + disease + '_' + date +'/' + 'classification' + ROI * '_ROI' + '.csv'

    # feature combination
    feature_save_eval2(file, merge_df, outputfile)


