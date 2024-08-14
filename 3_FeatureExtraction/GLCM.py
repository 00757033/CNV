import os
import math
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA 
from sklearn import naive_bayes, svm, tree
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from skimage.feature import graycomatrix, graycoprops

from sklearn.model_selection import train_test_split
import statistics
from scipy import stats
import csv
import SimpleITK as sitk
import six


#將OCTA影像
def get_frame(address ,image_floder, label_floder, output_floder, flag, show=False):
    area = []
    circularity = []
    center = []
    frame = []
    label = []
    frame_image = []
    frame_mask = []
    images = os.listdir(address + '/' + image_floder)
    masks = os.listdir(address + '/' + label_floder)
    setFloder(address + output_floder)
    for i in range(len(images)):
        img = cv2.imread(address + '/' + image_floder + '/' + images[i])   
        msk = cv2.imread(address + '/' + label_floder + '/' + masks[i])
        #print(image_address + images[i])
        img_g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        msk_g = cv2.cvtColor(msk, cv2.COLOR_BGR2GRAY)
        frame_image.append(img_g)
        contours, hierarchy = cv2.findContours(msk_g, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        if len(contours) != 0:
            # 集合所有contours的點，並找出影像的序號
            points = np.vstack(contours)
            #index = int(images[i].split('.')[0])
            
            # 計算面積
            area.append(cv2.contourArea(points))

            # 計算圓率
            ellipse = cv2.fitEllipse(points)
            if ellipse[1][0] < ellipse[1][1]:
                circularity.append([ellipse[1][0] / ellipse[1][1]])
            else:
                circularity.append([ellipse[1][1] / ellipse[1][0]])

            # 紀錄中心點，用於計算位移
            center.append(ellipse[0])

            # 分割出正中神經的影像
            x, y, w, h = cv2.boundingRect(points)
            frame.append(img_g[y:y+h, x:x+w])
            
            img_mask = img_g
            img_mask[msk_g==0] = 0
            cv2.imwrite(address + '/' + output_floder + '/' + images[i], img_mask[y:y+h, x:x+w])
            frame_mask.append(img_mask[y:y+h, x:x+w])
            
        else:
            area.append([0.0])
            circularity.append([0.0])
            center.append([])
            frame.append([[0]])

        if show:
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), thickness=2)
            cv2.ellipse(img, ellipse, (0, 0, 255), thickness=2)
            cv2.circle(img, (int(ellipse[0][0]), int(ellipse[0][1])), 5, (0, 0, 255), -1)
            cv2.imshow('test', img)
            cv2.waitKey(1000)
    if flag:
        label =np.zeros(len(images), dtype=int)
    else:
        label =np.ones(len(images), dtype=int)
    return area, circularity, center, frame, label, frame_image, frame_mask

#取得GLCM結果
def get_features(address, flag, GLCM_distance, retrain=False):
    train_features = []
    test_features = []
    if len(train_features) == 0 or len(test_features) == 0 or retrain:
        if len(train_features) == 0 or retrain:
            train_area, train_circularity, train_center, train_frame, label, frame_image, frame_mask= get_frame(address, 'image', 'mask' , 'limit', flag)   
            train_features = GLCM(train_frame, GLCM_distance)
            train_features_image  = GLCM(frame_image, GLCM_distance)
            train_features_mask = GLCM(frame_mask, GLCM_distance)
    return train_features, label, train_area, train_features_image, train_features_mask

#計算ENT
def entropy(glcm, level):
    s = 0.0
    for i in range(level):
        for j in range(level):
            if glcm[i, j] == 0:
                s += 0
            else:
                s += glcm[i, j] * np.log(glcm[i, j])
    return -1 * s

#從灰階影像轉換成GLCM，提取6個紋理特徵(ENT、COR、DISSI、HOM、EN、CON)
def GLCM(frame, distance):
    data      = []
    for i in range (0,len(frame)):
        glcm = graycomatrix(frame[i], distances=[distance], angles=[0, np.pi/4, np.pi/2, np.pi*3/4], levels=256, symmetric=True, normed=True)
          = [
            # 熵: 圖像的複雜程度。當共生矩陣中所有值均相等時，隨機性越大
            entropy(glcm[:, :, 0, 0], 256), entropy(glcm[:, :, 0, 1], 256), entropy(glcm[:, :, 0, 2], 256), entropy(glcm[:, :, 0, 3], 256),
            # 相關性
            graycoprops(glcm, 'correlation')[0, 0], graycoprops(glcm, 'correlation')[0, 1], graycoprops(glcm, 'correlation')[0, 2], graycoprops(glcm, 'correlation')[0, 3],
            # 相異性: 紋理規律性較強，值較大
            graycoprops(glcm, 'dissimilarity')[0, 0], graycoprops(glcm, 'dissimilarity')[0, 1], graycoprops(glcm, 'dissimilarity')[0, 2], graycoprops(glcm, 'dissimilarity')[0, 3],
            # 同質性: 測量影像的均勻性，非均勻影像的值較低，均勻影像的值較高
            graycoprops(glcm, 'homogeneity')[0, 0], graycoprops(glcm, 'homogeneity')[0, 1], graycoprops(glcm, 'homogeneity')[0, 2], graycoprops(glcm, 'homogeneity')[0, 3],
            # 角二階矩: 用來描述GLCM的均勻程度，若值都非常接近，則ASM值較小，若值差別較大，則ASM值較大
            graycoprops(glcm, 'ASM')[0, 0], graycoprops(glcm, 'ASM')[0, 1], graycoprops(glcm, 'ASM')[0, 2], graycoprops(glcm, 'ASM')[0, 3],
            # energy
            #graycoprops(glcm, 'energy')[0, 0], graycoprops(glcm, 'energy')[0, 1], graycoprops(glcm, 'energy')[0, 2], graycoprops(glcm, 'energy')[0, 3],
            # 對比度: 紋理越清晰反差越大對比度也就越大
            graycoprops(glcm, 'contrast')[0, 0], graycoprops(glcm, 'contrast')[0, 1], graycoprops(glcm, 'contrast')[0, 2], graycoprops(glcm, 'contrast')[0, 3]]
        data.append(features)
        #features = [f / frame_area_ratio[i] for f in features]
        #data_norm.append(features)
    return data

def setFloder(path):
    if not os.path.isdir(path) : os.mkdir(path)

#畫術前術後比較的barchart
def drawBarCompare(pre_scores, post_scores, pre_std, post_std, number, func):
    score_pre = [pre_scores]
    score_post = [post_scores]
    x = np.array([1.4, 2])
    width = 0.01
    lower_error = 0
    Max = 0
    
    if Max < np.max(post_scores) + np.max(post_std):
        Max = np.max(post_scores) + np.max(post_std)
    if Max < np.max(pre_scores) + np.max(pre_std):
        Max = np.max(pre_scores) + np.max(pre_std)
    plt.subplot(1,4, number+1)
    plt.bar(1.4, score_pre, width, color='dimgrey', label='Pre-injection', yerr=[[lower_error], [pre_std]] , capsize=5, linewidth=1, edgecolor=['black'], zorder=100)
    plt.bar(1.4+width*1.3, score_post, width, color='lightgrey', label='Post-injection', yerr=[[lower_error], [post_std]], capsize=5, linewidth=1, edgecolor=['black'],zorder=100)
    plt.tick_params(
        axis='x',          # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        bottom=False,      # ticks along the bottom edge are off
        top=False,         # ticks along the top edge are off
        labelbottom=False) # labels along the bottom edge are off
    #plt.xlabel('Degree')
    plt.ylabel(func, fontsize=14)
    plt.ylim(0, Max*1.6)
    #plt.title(func)
    plt.legend(loc=1, frameon=False).set_zorder(200)

#畫術前術後比較的barchart
def drawBarall(pre_scores, post_scores, pre_std, post_std, func):
    width = 0.35
    lower_error = [0,0,0,0]
    Max = 0
    label = [func+'(d=1)', func+'(d=2)', func+'(d=4)', func+'(d=8)']
    if Max < np.max(post_scores) + np.max(post_std):
        Max = np.max(post_scores) + np.max(post_std)
    if Max < np.max(pre_scores) + np.max(pre_std):
        Max = np.max(pre_scores) + np.max(pre_std)
    x = np.arange(len(pre_scores)) 
    plt.bar(x - width/2, pre_scores, width, color='dimgrey', label='Pre-treatment', yerr=[lower_error, pre_std] , capsize=5, linewidth=1, edgecolor=['black'], zorder=100)
    plt.bar(x + width/2, post_scores, width, color='lightgrey', label='Post-treatment', yerr=[lower_error, post_std], capsize=5, linewidth=1, edgecolor=['black'],zorder=100)
    plt.xticks(x,label)
    #plt.ylabel(func, fontsize=14)
    plt.ylim(0, Max*1.4)
    #plt.title(func)
    plt.legend(loc=1, frameon=False).set_zorder(200)
    plt.show()

#計算平均跟標準差
def getMeanStd(feature):
    data = []
    data_total =[]
    data_mean = []
    data_std  = []
    for i in range(len(feature[0])):
        for j in range(len(feature)):
            data.append(round(feature[j][i], 3))
        data_total.append(data)
        data = []

    for i in data_total:
        data_mean.append(round(sum(i)/len(i), 3))
        data_std.append(round(statistics.pstdev(i), 3))
    
    return data_total, data_mean, data_std

#計算術前術後的差值
def getDiff(feature1, feature2):
    data = []
    data_diff = []
    data_mean = []
    data_std  = []
    for i in range(len(feature1[0])):
        for j in range(len(feature1)):
            data.append((feature2[j][i]-feature1[j][i])/feature1[j][i])
        data_diff.append(data)
        data = []
    for i in data_diff:
        data_mean.append(round(sum(i)/len(i), 3))
        data_std.append(round(statistics.pstdev(i), 3))

    return data_diff, data_mean, data_std

if __name__ == '__main__':
    path_base = '../../Data/'
    path_date = path_base + 'CNV_0831' #資料夾

    layer = 'OR'
    distance_GLCM = [1, 2, 4, 8] #GLCM參數 d
    data_pre      = []
    data_post     = []
    data_pre_std  = []
    data_post_std = []
    data_pv       = []
    data_CR       = []

    total_4distance_features =[]
    for distance in distance_GLCM :
        path= [ path_date + '/' + layer + '/compare/1/', path_date + '/' + layer + '/compare/2/']
        flag = [True, False]   
        pre_features, pre_y, pre_area , pre_features_image, pre_features_mask = get_features(path[0], flag[0], distance)
        post_features, post_y, post_area, post_features_image, post_features_mask  = get_features(path[1], flag[1], distance)
        #//////////////////////////////////////////////////////////////////////////////////////////////////////
        total_features = []
        for i  in range(len(post_features)): 
            feature = []
            for j  in range(len(post_features[i])):
               feature.append( (post_features[i][j] - pre_features[i][j])/post_features[i][j])
            #total_feature = pre_features[i] + post_features[i] 
            feature_4degree = []
            for k in range(len(feature)):
                if k%4 == 0:
                    feature_4degree.append((feature[k] + feature[k+1] + feature[k+2] + feature[k+3])/4)
            total_features.append(feature_4degree)
        total_4distance_features.append(total_features)


        #///////////////////////////////////////////////////////////////////////////////////////////////////////
        area_mean = sum(pre_area)/len(pre_area)
        pre_data, pre_mean, pre_std = getMeanStd(pre_features_mask)
        post_data, post_mean, post_std = getMeanStd(post_features_mask)
        data_diff, data_diff_mean, data_diff_std = getDiff(pre_features_mask, post_features_mask)
        data_count = len(pre_features)
        count  = 0
        number = 0
        mean_pre      = []
        mean_post     = []
        mean_pre_std  = []
        mean_post_std = []
        mean_pv        = []
        mean_CR       = []
        for func in ['Entropy', 'Correlation', 'Dissimilarity', 'Homogeneity', 'Energy', 'Contrast'] :  
            mean_value_pre = 0
            mean_std_pre = 0 
            mean_value_post = 0
            mean_std_post = 0 
            mean_p = 0
            for degree in ['0', '45', '90', '135']:
                mean_value_pre = mean_value_pre + pre_mean[count]
                mean_std_pre = mean_std_pre + pre_std[count]
                mean_value_post = mean_value_post + post_mean[count]
                mean_std_post = mean_std_post + post_std[count]
                t2, p2 = stats.ttest_ind_from_stats(pre_mean[count], pre_std[count], data_count, post_mean[count],post_std[count], data_count)
                mean_p = mean_p + p2
                count = count + 1
            mean_pre.append(round(mean_value_pre/4, 2))
            mean_post.append(round(mean_value_post/4, 2))
            mean_pre_std.append(round(mean_std_pre/4, 2))
            mean_post_std.append(round(mean_std_post/4, 2))
            mean_pv.append(round(mean_p/4, 2))
            mean_CR.append(100*round((mean_value_post-mean_value_pre)/mean_value_pre, 2))
            print('**************************')
            print(func)
            print('mean pre value : ' + str(round(mean_value_pre/4, 2)) + ' ± ' + str(round(mean_std_pre/4, 2)))
            print('mean post value : ' + str(round(mean_value_post/4, 2)) + ' ± ' + str(round(mean_std_post/4, 2)))
            print('change ratio : ' + str(100*round((mean_value_post-mean_value_pre)/mean_value_pre, 2)))
            print('P value : ' + str(round(mean_p/4, 2)))
            print('**************************')
            print('-----------------------------------------------------------')
            number =number + 1
        data_pre.append(mean_pre)
        data_post.append(mean_post)
        data_pre_std.append(mean_pre_std)
        data_post_std.append(mean_post_std)
        data_CR.append(mean_CR)
        data_pv.append(mean_pv)
    #////////////////////////////////////////////////////////////////////////////////////
    #將GLCM結果寫入GLCM.csv
    id = []
    eye = []
    id_list = os.listdir(path_date + '/' + layer + '/compare/1/image')
    for image_name in id_list:
        print(image_name[3:11], image_name[12])
        id.append([image_name[3:11]])
        if image_name[12] == "L":
            eye.append(["OS"])
        else:
            eye.append(["OD"])
    final_features= []
    for i in range(len(total_4distance_features[0])):
        final_feature = id[i] + eye[i] + total_4distance_features[0][i] + total_4distance_features[1][i] + total_4distance_features[2][i] + total_4distance_features[3][i]
        final_features.append(final_feature)

    with open('GLCM.csv', 'w', newline='') as file:
            writer = csv.writer(file, quoting=csv.QUOTE_ALL,delimiter=',')
            writer.writerows(final_features)
    #////////////////////////////////////////////////////////////////////////////////////
    count = 0 
    #data = [data_pre, data_post, data_pre_std, data_post_std]
    for func in ['ENT', 'COR', 'DISSI', 'HOM', 'EN', 'CON'] :
        print(func)
        pre_score = []
        post_score= []
        pre_score_std = []
        post_score_std = []
        for i in [0, 1, 2, 3]:
            pre_score.append(data_pre[i][count])
            post_score.append(data_post[i][count])
            pre_score_std.append(data_pre_std[i][count])
            post_score_std.append(data_post_std[i][count])
        function        = ['d=1', 'd=2', 'd=4', 'd=8']
        number          = [1, 2, 3, 4]
        data_count = 70
        drawBarall(pre_score, post_score, pre_score_std, post_score_std, func)
        for i in range(0,4):
            t2, p2 = stats.ttest_ind_from_stats(pre_score[i], pre_score_std[i], data_count, post_score[i],post_score_std[i], data_count)
            print(function[i], " pre : ", str(pre_score[i]) + '±' + str(pre_score_std[i]))
            print(function[i], " post : ",  str(post_score[i]) + '±' + str(post_score_std[i]))
            print("P-value : " + str(round(p2, 5)))
            print('change ratio : ' + str(100*round((post_score[i]-pre_score[i])/pre_score[i], 2)))
            #drawBarCompare(pre_score[i], post_score[i], pre_score_std[i], post_score_std[i], i, function[i])
        print('**************************')
        #plt.show()
        count = count + 1
      
        