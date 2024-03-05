import matplotlib.pyplot as plt
from sklearn.metrics import jaccard_score , dice_score
import cv2
import numpy as np
import os
import glob
import matplotlib.image as img
from pathlib import Path
import json
import csv
#記錄下每次訓練最佳跟最差的訓練資料的JI跟DC
def mean_pixel_accuracy(y_true, y_pred):
    # Count the number of matching pixels
    count_same = np.sum(y_true == y_pred)
    size = y_true.shape[0] * y_true.shape[1]
    return count_same / size



def calculate_best_worst(folder_name, result_path):
    # print(result_path + folder_name + '/images/')
    index = os.listdir(result_path + folder_name + '/images/')
    best = {'case': 0, 'ji': 0, 'dc': 0, 'pa': 0}
    worst = {'case': 0, 'ji': 0, 'dc': 0, 'pa': 0}
    count = 0
    best_case = 'Error'
    worst_case = 'Error'
    arr_JI = []
    arr_DC = []
    arr_PA = []
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

        jaccard_index  = jaccard_score(img_true, img_pred)
    
        dice_coefficient = dice_score(img_true, img_pred)
        
        pixel_accuracy = mean_pixel_accuracy(img_true, img_pred)



        if jaccard_index > best['ji']:
            best['ji'] = jaccard_index
            best['case'] = i
            best['dc'] = dice_coefficient
            best['pa'] = pixel_accuracy
        if jaccard_index < worst['ji']:
            worst['ji'] = jaccard_index
            worst['case'] = i
            worst['dc'] = dice_coefficient
            worst['pa'] = pixel_accuracy

            
        arr_JI.append(jaccard_index) # np.std(arr, ddof=1)
        arr_DC.append(dice_coefficient)
        arr_PA.append(pixel_accuracy)
        lines = ['Image : ' + i + ',  JI Score : ' + str(round(jaccard_index,4)*100) + '%\t' + 'DC Score : ' + str(round(dice_coefficient,4)*100) + '%\t' + 'PA Score : ' + str(round(pixel_accuracy,4)*100) + '%\n']
        count = count + 1 
        # ji_score = round(sum(arr_JI) / len(arr_JI), 2)
        # ji_var   = round(np.std(arr_JI, ddof=1), 2)
        # dc_score = round(sum(arr_DC) / len(arr_DC), 3)
        # dc_var   = round(np.std(arr_DC, ddof=1), 3)
        avg_ji = round(sum(arr_JI) / len(arr_JI), 4) * 100
        ji_var = round(np.std(arr_JI, ddof=1), 4) * 100
        avg_dc = round(sum(arr_DC) / len(arr_DC), 4) * 100
        dc_var = round(np.std(arr_DC, ddof=1), 4) * 100
        avg_pa = round(sum(arr_PA) / len(arr_PA), 4) * 100
        pa_var = round(np.std(arr_PA, ddof=1), 4) * 100
        
    best.update({'ji': round(best['ji'], 4)*100, 'dc': round(best['dc'], 4)*100, 'pa': round(best['pa'], 4)*100})
    worst.update({'ji': round(worst['ji'], 4)*100, 'dc': round(worst['dc'], 4)*100, 'pa': round(worst['pa'], 4)*100})
    lines = ['Avg JI Score : ' + str(avg_ji) + '%\t' + 'JI Variance : ' + str(ji_var) + '%\n',
            'Avg DC Score : ' + str(avg_dc) + '%\t' + 'DC Variance : ' + str(dc_var) + '%\n',
            'Avg PA Score : ' + str(avg_pa) + '%\t' + 'PA Variance : ' + str(pa_var) + '%\n',
            'Best Case : ' + best['case'] + ',  JI Score : ' + str(best['ji']) + '%\t' + 'DC Score : ' + str(best['dc']) + '%\t' + 'PA Score : ' + str(best['pa']) + '%\n',
            'Worst Case : ' +  worst['case'] + ',  JI Score : ' + str(worst['ji']) + '%\t' + 'DC Score : ' + str(worst['dc']) + '%\t' + 'PA Score : ' + str(worst['pa']) + '%\n']
    f.writelines(lines)
    f.close()
    return  avg_ji, ji_var, avg_dc, dc_var, avg_pa, pa_var, best, worst

def getResult(PATH):
    path = Path(PATH)
    df = dict()
    for dataset in path.iterdir():
        avg = dict()

        for data in dataset.iterdir():
            for model in data.iterdir():
                Avg_JI , JI_var, Avg_DC, DC_var, Avg_PA, PA_var, best_case, worst_case = calculate_best_worst(model.name, PATH)
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
                data = { 
                    'Avg JI Score': Avg_JI,
                    'JI Variance': JI_var,
                    'Avg DC Score': Avg_DC,
                    'DC Variance': DC_var,
                    'Avg PA Score': Avg_PA,
                    'PA Variance': PA_var,
                    'Best Case': best_case,
                    'Worst Case': worst_case
                }
                df[dataset.name][name][epoch][batch][learning_rate].append(data)
                

    # json_data = json.dumps(df, indent=4)           
    # # Writing to sample.json
    # with open("sample5.json", "w") as outfile:
    #     outfile.write(json_data) 


    if not os.path.exists('./record'):
        os.mkdir('./record')

    if os.path.exists('./record/' + 'result_all.csv'):
        os.remove('./record/' + 'result_all.csv')
        
    with open( './record/' + 'result_all.csv', 'w', newline='') as csvf:
        csv_writer = csv.writer(csvf)
        csv_writer.writerow(['Dataset', 'Model', 'Batch','Epoch', 'Learning Rate', 'Avg JI Score', 'JI Variance', 'Avg DC Score', 'DC Variance', 'Avg PA Score', 'PA Variance', 'Best Case', 'Best JI Score', 'Best DC Score', 'Best PA Score', 'Worst Case', 'Worst JI Score', 'Worst DC Score', 'Worst PA Score'])
        for dataset in df:
            print('dataset',dataset)
            for model in df[dataset]:
                for epoch in df[dataset][model]:
                    for batch in df[dataset][model][epoch]:
                        for learning_rate in df[dataset][model][epoch][batch]:
                            for i in df[dataset][model][epoch][batch][learning_rate]:
                                csv_writer.writerow([dataset, model, batch, epoch, learning_rate, i['Avg JI Score'], i['JI Variance'], i['Avg DC Score'], i['DC Variance'], i['Avg PA Score'], i['PA Variance'], i['Best Case'], i['Best JI Score'], i['Best DC Score'], i['Best PA Score'], i['Worst Case'], i['Worst JI Score'], i['Worst DC Score'], i['Worst PA Score']])
                                                        


'''
def main():
    data_date = "0319"
    data_class = 'CNV'
    PATH = 'D:/WeiHao/Result/' + data_class  + '_' + data_date
    getResult(PATH)


if __name__ == '__main__':
    main()
'''