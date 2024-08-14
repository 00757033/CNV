import matplotlib.pyplot as plt
from sklearn.metrics import jaccard_score  , confusion_matrix , f1_score, precision_score
import cv2
import numpy as np
import os
import glob
import matplotlib.image as img
from pathlib import Path
import json
import csv
import tools.tools as tools
import tensorflow as tf
from tensorflow.python.ops.numpy_ops import np_config
np_config.enable_numpy_behavior()
#記錄下每次訓練最佳跟最差的訓練資料的JI跟DC
def mean_pixel_accuracy(y_true, y_pred):
    # Count the number of matching pixels
    count_same = np.sum(y_true == y_pred)
    size = y_true.size # Total number of pixels
    return count_same / size


def jaccard_ind(y_true, y_pred):
    smooth = 1e-8
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    intersection = tf.reduce_sum(y_true * y_pred)
    union = tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) - intersection
    jac = (intersection + smooth) / (union + smooth)
    # tf.Tensor to  int
    jac = jac.numpy()
    
    return jac

def dice_coef(y_true, y_pred):
    smooth = 1e-8
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    intersection = tf.reduce_sum(y_true * y_pred)
    union = tf.reduce_sum(y_true) + tf.reduce_sum(y_pred)
    dic = (2. * intersection + smooth) /(union + smooth)
    # tf.Tensor to  int
    dic = dic.numpy()
    return dic
    
def recall_score(y_true, y_pred):
    smooth = 1e-8
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    recall =  tf.reduce_sum(y_true * y_pred) / (tf.reduce_sum(y_true) + smooth)
    recall = recall.numpy()
    return recall


def calculate_best_worst(folder_name, result_path):
    index = os.listdir(os.path.join(result_path, folder_name, 'images'))
    best = {'case': 'Error', 'ji': 0.0 ,'dc': 0.0, 'pa': 0.0, 'sen': 0.0}
    worst = {'case': 'Error', 'ji': 1, 'dc': 1, 'pa': 1, 'sen': 1}
    count = 0

    arr_JI = []
    arr_DC = []
    arr_PA = []
    arr_SEN = []
    if os.path.exists(os.path.join(result_path, folder_name, 'result2.txt')):  
        os.remove(os.path.join(result_path, folder_name, 'result2.txt'))

    f = open(os.path.join(result_path, folder_name, 'result2.txt'), 'a')

    for i in index:
        lines = [ i + '\n']
        f.writelines(lines)
        #name = str(count)
        img_true = cv2.imread(os.path.join(result_path, folder_name, 'masks', i), 0)

        img_true[img_true < 128] = 0
        img_true[img_true >= 128] = 1
        img_pred = cv2.imread(os.path.join(result_path, folder_name, 'predict', i), 0)

        img_pred[img_pred < 128] = 0
        img_pred[img_pred >= 128] = 1
        img_true = np.array(img_true).ravel()
        img_pred = np.array(img_pred).ravel()

        jaccard_index  = jaccard_ind(img_true, img_pred)
    
        dice_coefficient = dice_coef(img_true, img_pred)
        
        pixel_accuracy = mean_pixel_accuracy(img_true, img_pred)
        
        sensivity = recall_score(img_true, img_pred)


        if jaccard_index > best['ji']:
            best['ji'] = jaccard_index
            best['case'] = i
            best['dc'] = dice_coefficient
            best['pa'] = pixel_accuracy
            best['sensivity'] = sensivity
        if jaccard_index < worst['ji']:
            worst['ji'] = jaccard_index
            worst['case'] = i
            worst['dc'] = dice_coefficient
            worst['pa'] = pixel_accuracy
            worst['sensivity'] = sensivity
            
        arr_JI.append(jaccard_index) # np.std(arr, ddof=1)
        arr_DC.append(dice_coefficient)
        arr_PA.append(pixel_accuracy)
        arr_SEN.append(sensivity)
        lines = ['Image : ' + i + ',  JI Score : ' + str(round(jaccard_index,4)*100) + '%' + ',  DC Score : ' + str(round(dice_coefficient,4)*100) + '%' + ',  PA Score : ' + str(round(pixel_accuracy,4)*100) + '%' + ',  Sensivity : ' + str(round(sensivity,4)*100) + '\n']
        f.writelines(lines)
        count = count + 1 

        avg_ji = round(sum(arr_JI) / len(arr_JI), 4) * 100
        ji_var = round(np.std(arr_JI, ddof=1), 4) * 100
        avg_dc = round(sum(arr_DC) / len(arr_DC), 4) * 100
        dc_var = round(np.std(arr_DC, ddof=1), 4) * 100
        avg_pa = round(sum(arr_PA) / len(arr_PA), 4) * 100
        pa_var = round(np.std(arr_PA, ddof=1), 4) * 100
        avg_sen = round(sum(arr_SEN) / len(arr_SEN), 4) * 100
        sen_var = round(np.std(arr_SEN, ddof=1), 4) * 100
    
    best.update({'ji': round(best['ji'], 4)*100, 'dc': round(best['dc'], 4)*100, 'pa': round(best['pa'], 4)*100, 'sen': round(best['sen'], 4)*100})
    worst.update({'ji': round(worst['ji'], 4)*100, 'dc': round(worst['dc'], 4)*100, 'pa': round(worst['pa'], 4)*100, 'sen': round(worst['sen'], 4)*100})
    
    lines = ['Avg JI Score : ' + str(avg_ji) + '\t' + 'JI Variance : ' + str(ji_var) +'\n',
            'Avg DC Score : ' + str(avg_dc) + '\t' + 'DC Variance : ' + str(dc_var) +'\n',
            'Avg PA Score : ' + str(avg_pa) + '\t' + 'PA Variance : ' + str(pa_var) +'\n',
            'Avg Sensivity : ' + str(avg_sen) + '\n', 'Sensivity Variance : ' + str(round(np.std(arr_SEN, ddof=1), 4)*100) + '\n',
            'Best Case : ' + best['case'] + ',  JI Score : ' + str(best['ji'])  + ',  DC Score : ' + str(best['dc']) ,
            ',  PA Score : ' + str(best['pa']) + ',  Sensivity : ' + str(best['sen']) + '\n',
            'Worst Case : ' +  worst['case'] + ',  JI Score : ' + str(worst['ji'])  + ',  DC Score : ' + str(worst['dc'])  ,
            ',  PA Score : ' + str(worst['pa']) +  ',  Sensivity : ' + str(worst['sen']) + '\n']
    f.writelines(lines)
    f.close()
    # print(lines)
    return  avg_ji, ji_var, avg_dc, dc_var, avg_pa, pa_var, avg_sen ,sen_var,best, worst

def getResult(PATH):
    path = Path(PATH)
    print('path',path.stem)
    
    df = dict()
    for dataset in path.iterdir():
        for data in dataset.iterdir():
            print('data',data)
            for model in data.iterdir():
                print(model.name,str(data))
                Avg_JI , JI_var, Avg_DC, DC_var, Avg_PA, PA_var, Avg_SEN, SEN_var, best_case, worst_case = calculate_best_worst(model.name,str(data)+'/')
                print('-'*50)
                name = model.name.split('_')[0]
                epoch = model.name.split('_')[1]
                batch = model.name.split('_')[2]
                learning_rate = model.name.split('_')[3]
                filter = model.name.split('_')[4]
                if dataset.name not in df:
                    df[dataset.name] = dict()

                if name not in df[dataset.name]:
                    df[dataset.name][name] = dict()

                if epoch  not in df[dataset.name][name]:
                    df[dataset.name][name][epoch] =  dict()

                if batch not in df[dataset.name][name][epoch]:
                    df[dataset.name][name][epoch][batch] =  dict()

                if learning_rate not in df[dataset.name][name][epoch][batch]:
                    df[dataset.name][name][epoch][batch][learning_rate] = dict()
                if filter not in df[dataset.name][name][epoch][batch][learning_rate]:
                    df[dataset.name][name][epoch][batch][learning_rate][filter] = list()
                value = { 
                    'Avg JI Score': Avg_JI,
                    'JI Variance': JI_var,
                    'Avg DC Score': Avg_DC,
                    'DC Variance': DC_var,
                    'Avg PA Score': Avg_PA,
                    'PA Variance': PA_var,
                    'Avg SEN Score': Avg_SEN,
                    'SEN Variance': SEN_var,
                    'Best Case': best_case,
                    'Worst Case': worst_case
                }
                df[dataset.name][name][epoch][batch][learning_rate][filter].append(value)
                

    # json_data = json.dumps(df, indent=4)           
    # # Writing to sample.json
    # with open("sample5.json", "w") as outfile:
    #     outfile.write(json_data) 


    if not os.path.exists( os.path.join ('./record','ModelResult')):
        os.mkdir(os.path.join ('./record', 'ModelResult'))

    if os.path.exists(os.path.join('./record', 'ModelResult', path.stem + '.csv')):
        os.remove(os.path.join('./record', 'ModelResult', path.stem + '.csv'))
        
    with open( os.path.join('./record', 'ModelResult', path.stem + '.csv'), 'w', newline='') as csvf:
        csv_writer = csv.writer(csvf)
        csv_writer.writerow(['Dataset', 'Model', 'Batch','Epoch', 'Learning Rate', 'Filter', 'Avg JI Score', 'JI Variance', 'Avg DC Score', 'DC Variance', 'Avg PA Score', 'PA Variance', 'Avg SEN Score', 'SEN Variance', 'Best Case', 'Best JI', 'Best DC', 'Best PA', 'Worst Case', 'Worst JI', 'Worst DC', 'Worst PA'])
        print('dataset',dataset)
        for dataset in df:
            for model in df[dataset]:
                for epoch in df[dataset][model]:
                    for batch in df[dataset][model][epoch]:
                        for learning_rate in df[dataset][model][epoch][batch]:
                            for  filters in df[dataset][model][epoch][batch][learning_rate]:
                                for i in df[dataset][model][epoch][batch][learning_rate][filters]:
                                    csv_writer.writerow([dataset, model, batch, epoch, learning_rate, filters, i['Avg JI Score'], i['JI Variance'], i['Avg DC Score'], i['DC Variance'], i['Avg PA Score'], i['PA Variance'],i['Avg SEN Score'], i['SEN Variance'], i['Best Case']['case'], i['Best Case']['ji'], i['Best Case']['dc'], i['Best Case']['pa'], i['Worst Case']['case'], i['Worst Case']['ji'], i['Worst Case']['dc'], i['Worst Case']['pa']])
'''
def main():
    data_date = "0319"
    data_class = 'CNV'
    PATH = 'D:/WeiHao/Result/' + data_class  + '_' + data_date
    getResult(PATH)


if __name__ == '__main__':
    main()
'''