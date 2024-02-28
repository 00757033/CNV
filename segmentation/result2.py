import matplotlib.pyplot as plt
from sklearn.metrics import jaccard_score
import cv2
import numpy as np
import os
import glob
import matplotlib.image as img
from pathlib import Path
import json
import csv
#記錄下每次訓練最佳跟最差的訓練資料的JI跟DC

def calculate_best_worst(folder_name, result_path):
    # print(result_path + folder_name + '/images/')
    index = os.listdir(result_path + folder_name + '/images/')
    best = 0
    worst = 1
    count = 0
    best_case = 'Error'
    worst_case = 'Error'
    arr_JI = []
    arr_DC = []
    if os.path.exists(result_path + folder_name +'/result2.txt'):
        os.remove(result_path + folder_name +'/result2.txt')

    f = open(result_path + folder_name +'/result2.txt', 'w')
    print('result_path folder_name ',result_path + folder_name )
    for i in index:
        lines = [ i + '\n']
        f.writelines(lines)
        #name = str(count)
        img_true = cv2.imread(result_path + folder_name + '/masks/' + i, 0)

        img_true[img_true < 128] = 0
        img_true[img_true >= 128] = 1
        img_pred = cv2.imread(result_path + folder_name + '/predict/' + i, 0)

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
        arr_JI.append(score) # np.std(arr, ddof=1)
        arr_DC.append(dice_score)
        lines = ['Image : ' + i + ',  JI Score : ' + str(round(score, 3)) + ',  DC Score : ' + str(round(dice_score, 3)) + '\n']
        f.writelines(lines)
        count = count + 1 
        # ji_score = round(sum(arr_JI) / len(arr_JI), 2)
        # ji_var   = round(np.std(arr_JI, ddof=1), 2)
        # dc_score = round(sum(arr_DC) / len(arr_DC), 3)
        # dc_var   = round(np.std(arr_DC, ddof=1), 3)
    lines = ['Avg JI Score : ' + str(round(sum(arr_JI) / len(arr_JI),4)) + '\t' + 'JI Variance : ' + str(round(np.std(arr_JI, ddof=1), 4)) + '\n',
            'Avg DC Score : ' + str(round(sum(arr_DC) / len(arr_DC), 4)) + '\t' + 'DC Variance : ' + str(round(np.std(arr_DC, ddof=1), 4)) + '\n',
            'Best Case : ' + best_case + ',  JI Score : ' + str(round(best, 4)) + '\n',
            'Worst Case : ' + worst_case + ',  JI Score : ' + str(round(worst, 4))]
    f.writelines(lines)
    f.close()
    return  str(round(sum(arr_JI) / len(arr_JI),4)),str(round(np.std(arr_JI, ddof=1), 4)) , str(round(sum(arr_DC) / len(arr_DC), 4)) ,str(round(np.std(arr_DC, ddof=1), 4)),best_case,str(round(best, 5)),worst_case, str(round(worst, 5))

def getResult(PATH):
    path = Path(PATH)
    df = dict()
    for dataset in path.iterdir():
        avg = dict()

        for data in dataset.iterdir():
            for model in data.iterdir():
                Avg_JI , JI_var, Avg_DC, DC_var, best_case, best_score, worst_case, worst_score = calculate_best_worst(model.name,str(data)+'\\')
                name = model.name.split('_')[0]
                epoch = model.name.split('_')[1]
                batch = model.name.split('_')[2]
                learning_rate = model.name.split('_')[3]
                if dataset.name not in df:
                    df[dataset.name] = dict()

                if name not in df[dataset.name]:
                    df[dataset.name][name] = dict()

                if epoch  not in df[dataset.name][name]:
                    df[dataset.name][name][epoch] =  dict()

                if batch not in df[dataset.name][name][epoch]:
                    df[dataset.name][name][epoch][batch] =  dict()

                if learning_rate not in df[dataset.name][name][epoch][batch]:
                    df[dataset.name][name][epoch][batch][learning_rate] = []

                df[dataset.name][name][epoch][batch][learning_rate].append([ Avg_JI, JI_var, Avg_DC, DC_var, best_case, best_score, worst_case, worst_score])
                

    json_data = json.dumps(df, indent=4)           
    # Writing to sample.json
    with open("sample5.json", "w") as outfile:
        outfile.write(json_data) 


    with open( 'result.csv', 'w', newline='') as csvf:
        csv_writer = csv.writer(csvf)
        csv_writer.writerow(['Dataset', 'Model', 'Batch','Epoch', 'Learning Rate', 'Avg JI Score', 'JI Variance', 'Avg DC Score', 'DC Variance', 'Best Case', 'Best Score', 'Worst Case', 'Worst Score'])
        for dataset in df:
            print('dataset',dataset)
            for model in df[dataset]:
                for epoch in df[dataset][model]:
                    for batch in df[dataset][model][epoch]:
                        for learning_rate in df[dataset][model][epoch][batch]:
                            for i in df[dataset][model][epoch][batch][learning_rate]:
                                csv_writer.writerow([str(dataset), str(model), str(batch), str(epoch), str(learning_rate), str(i[0]), str(i[1]), str(i[2]), str(i[3]), str(i[4]), str(i[5]), str(i[6]), str(i[7])])


'''
def main():
    data_date = "0319"
    data_class = 'CNV'
    PATH = 'D:/WeiHao/Result/' + data_class  + '_' + data_date
    getResult(PATH)


if __name__ == '__main__':
    main()
'''