import matplotlib.pyplot as plt
from sklearn.metrics import jaccard_score
import cv2
import numpy as np
import os
import glob
import matplotlib.image as img
from pathlib import Path
import json
#記錄下每次訓練最佳跟最差的訓練資料的JI跟DC

def calculate_best_worst(folder_name, result_path):
    index = os.listdir(result_path + folder_name + '/images/')
    best = 0
    worst = 1
    count = 0
    best_case = 'Error'
    worst_case = 'Error'
    arr_JI = []
    arr_DC = []
    if os.path.exists(result_path + folder_name +'/result.txt'):
        os.remove(result_path + folder_name +'/result.txt')

    f = open(result_path + folder_name +'/result.txt', 'w')

    for i in index:
        lines = [ i + '\n']
        f.writelines(lines)
        #name = str(count)
        img_true = cv2.imread(result_path + folder_name + '/masks/' + i, 0)

        img_true[img_true < 128] = 0
        img_true[img_true >= 128] = 1
        img_pred = cv2.imread(result_path + folder_name + '/results/' + i, 0)

        img_pred[img_pred < 128] = 0
        img_pred[img_pred >= 128] = 1
        img_true = np.array(img_true).ravel()
        img_pred = np.array(img_pred).ravel()

        score = jaccard_score(img_true, img_pred)
    
        dice_score = np.sum(img_pred[img_true==1])*2.0 / (np.sum(img_pred) + np.sum(img_true))

        if score > best :
            best = score
            best_case = i
        if score < worst:
            worst = score
            worst_case = i
        arr_JI.append(round(score, 3)) # np.std(arr, ddof=1)
        arr_DC.append(round(dice_score,3))
        lines = ['Image : ' + i + ',  JI Score : ' + str(round(score, 3)) + ',  DC Score : ' + str(round(dice_score, 3)) + '\n']
        f.writelines(lines)
        count = count + 1 
        ji_score = round(sum(arr_JI) / len(arr_JI), 3)
        ji_var   = round(np.var(arr_JI), 3)
        dc_score = round(sum(arr_DC) / len(arr_DC), 3)
        dc_var   = round(np.var(arr_DC), 3)
    lines = ['Avg JI Score : ' + str(round(sum(arr_JI)/len(arr_JI), 3)) + '\n',
            'Avg DC Score : ' + str(round(sum(arr_DC)/len(arr_DC), 3)) + '\n',
            'Best Case : ' + best_case + ',  JI Score : ' + str(round(best, 3)) + '\n',
            'Worst Case : ' + worst_case + ',  JI Score : ' + str(round(worst, 3))]
    f.writelines(lines)
    f.close()
    return  str(round(sum(arr_JI)/len(arr_JI), 5)),str(round(np.var(arr_JI), 5)) , str(round(sum(arr_DC)/len(arr_DC), 5)) ,str(round(np.var(arr_DC), 5)),best_case,str(round(best, 5)),worst_case, str(round(worst, 5))

def getResult(PATH):
    path = Path(PATH)
    df = dict()
    for dataset in path.iterdir():
        avg = dict()

        for data in dataset.iterdir():
            for model in data.iterdir():
              if str(data).split('\\')[-1] == 'train':
                Avg_JI , JI_var, Avg_DC, DC_var, best_case, best_score, worst_case, worst_score = calculate_best_worst(model.name,str(data)+'\\')
                name = model.name.split('_')[0]
                batch = model.name.split('_')[2]
                if dataset.name not in df:
                    df[dataset.name] = dict()

                if name not in df[dataset.name]:
                    df[dataset.name][name] = dict()
                    avg[name] = [[],[],[],[]]

                if batch  not in df[dataset.name][name]:
                    df[dataset.name][name][batch] = []

                df[dataset.name][name][batch].append([ Avg_JI, JI_var, Avg_DC, DC_var, best_case, best_score, worst_case, worst_score])
                avg[name][0].append(float(Avg_JI))
                avg[name][1].append(float(JI_var))
                avg[name][2].append(float(Avg_DC))
                avg[name][3].append(float(DC_var))
        avgs = {k: [str(round(sum(v[0])/len(v[0]), 3))+' '+str(round(sum(v[1])/len(v[1]), 3)), str(round(sum(v[2])/len(v[2]), 3))+' '+str(round(sum(v[3])/len(v[3]), 3))] for k, v in avg.items()}
        # df[dataset.name]['avg'] = avgs
    json_data = json.dumps(df, indent=4)           
    with open("sample4.json", "w") as outfile:
        outfile.write(json_data)    
    
