from msilib.schema import Binary
import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from PIL import Image
from skimage import morphology
import statistics
from scipy import stats
import scipy.misc
from scipy.stats import pearsonr
from skimage.feature import graycomatrix, graycoprops
from skimage import exposure
# from pyct import fdct2
import pywt
import pandas as pd 
import pathlib as pl
import tools.tools as tools
import json
import pyfeats 
import glrlm
import SimpleITK as sitk
import six
from skimage import io, color, measure
import math
from datetime import datetime
from radiomics import featureextractor
from radiomics import firstorder, getTestCase, glcm, glrlm, glszm, imageoperations, shape2D, ngtdm, gldm 
from dateutil import parser
# python -m pip install pyradiomics

class VesselAnalysis():
    def __init__( self, PATH_BASE,disease,compare_path, layers = ['CC']):
        self.PATH_BASE = PATH_BASE
        self.compare_path = compare_path
        self.layers = layers
        self.disease = disease
        self.inject()
    def inject(self,file = '../../Data/打針資料.xlsx',label = ["診斷","病歷號","眼睛","打針前門診日期","三針後門診"]):
        self.inject_df = pd.read_excel(file, sheet_name="Focea_collect",na_filter = False, engine='openpyxl')
        

        self.inject_df['病歷號'] = self.inject_df['病歷號'].apply(lambda x: '{:08d}'.format(x))
        self.inject_df = self.inject_df.sort_values(by=["病歷號","眼睛"])       



    def feature_extract(self,save_file_name = 'VesselFeature', mask_roi = True,cut = False):
        patient_feature = dict()
        cout = 0
        eye_dict = {'L':'OS','R':'OD'}
        
        for patient in pl.Path(os.path.join(self.compare_path) ).iterdir():
            patient_feature[patient.name] = dict()
            date = dict()
            layers = [layer + '_cut' if cut else layer for layer in self.layers]
            for layer in layers: 
                for image in pl.Path(os.path.join(patient, layer , 'images') ).iterdir():
                    patientid, eye = patient.name.split('_')
                    if image.name.endswith(".png"):
                        if self.inject_df[self.inject_df['病歷號'] == patientid].empty:
                            continue
                        patient_record = self.inject_df[self.inject_df['病歷號'] == patientid ]
                        patient_record = patient_record[patient_record['眼睛'] == eye_dict[eye]]
                        # date_ layer all  exist
                        if os.path.exists(os.path.join(patient, layer , 'masks', image.name)) and os.path.exists(os.path.join(patient, layer, 'images', image.name)):
                            image_date = image.name.split('_')[0]
                            if image_date not in date:
                                date[image_date] = 0
                            date[image_date] = 1
                # date 刪除 value < len(self.layers)
                for key in list(date.keys()):
                    if date[key] < len(self.layers):
                        del date[key]     
                if len(date) < 2:
                    del patient_feature[patient.name]
                else :
                    # date 排序
                    date = dict(sorted(date.items(), key=lambda item: item[1], reverse=True)) 
                    feature = dict()   
                    # 將msk 做聯集
                    if 'CC' in layer :
                        msk_CC = np.zeros((304,304), dtype=np.uint8)
                    if 'OR' in layer :
                        msk_OR = np.zeros((304,304), dtype=np.uint8)
                    for item in range(1):# len(date)
                        if 'CC' in layer :
                            CC_post = cv2.imread(os.path.join(patient, layer, 'masks', list(date.keys())[item] + '_CC.png'), cv2.IMREAD_GRAYSCALE)
                            msk_CC = np.where((msk_CC > 0) | (CC_post > 0), 255, 0)
                            
                        if 'OR' in layer :
                            OR_post = cv2.imread(os.path.join(patient, layer, 'masks', list(date.keys())[item] + '_OR.png'), cv2.IMREAD_GRAYSCALE)
                            msk_OR = np.where((msk_OR > 0) | (OR_post > 0), 255, 0)
                    
                    tools.makefolder(os.path.join(patient, layer, 'concat_masks'))
                    
                    # save all_msk 
                    tools.makefolder(os.path.join(patient, layer, 'concat_masks_rm_small'))
                    
                    if 'CC' in layer :
                        msk_CC[msk_CC > 0] = 255
                        msk_CC = msk_CC.astype(np.uint8)
                        cv2.imwrite(os.path.join(patient, layer, 'concat_masks', 'CC.png'), msk_CC)
                        msk_CC_rm_small = remove_small_area(msk_CC, min_area = 50)
                        cv2.imwrite(os.path.join(patient, layer, 'concat_masks_rm_small', 'CC.png'), msk_CC_rm_small)
                        # Dilation 
                        # msk_CC_rm_small = cv2.dilate(msk_CC_rm_small, np.ones((3,3), np.uint8), iterations=1)
                        
                        
                    if 'OR' in layer :
                        msk_OR[msk_OR > 0] = 255
                        msk_OR = msk_OR.astype(np.uint8)
                        cv2.imwrite(os.path.join(patient, layer, 'concat_masks', 'OR.png'), msk_OR)
                        msk_OR_rm_small = remove_small_area(msk_OR, min_area = 50)
                        cv2.imwrite(os.path.join(patient, layer, 'concat_masks_rm_small', 'OR.png'), msk_OR_rm_small)
                        # Dilation
                        # msk_OR_rm_small = cv2.dilate(msk_OR_rm_small, np.ones((4,4), np.uint8), iterations=1)
                    
                    # # Dilation 
                    # msk_CC_rm_small = cv2.dilate(msk_CC_rm_small, np.ones((4,4), np.uint8), iterations=1)
                    # msk_OR_rm_small = cv2.dilate(msk_OR_rm_small, np.ones((4,4), np.uint8), iterations=1)
                    
                    
                    # tools.makefolder(os.path.join(patient, 'all_msk'))
                    # cv2.imwrite(os.path.join(patient, 'all_msk', 'CC.png'), msk_CC_rm_small)
                    # cv2.imwrite(os.path.join(patient, 'all_msk', 'OR.png'), msk_OR_rm_small)
                    
                    for layer in self.layers:
                        cout = 0
                        feature = dict()
                        if 'CC' in layer :
                            msk_rm_small = msk_CC_rm_small
                            
                        else :
                            msk_rm_small = msk_OR_rm_small 
                        if set(msk_rm_small.flatten()) == {0}:
                            continue                       
                        
                        for item in range(len(date)):
                            cout += 1
                            data_time = list(date.keys())[item]
                            img_name = list(date.keys())[item] + '_' + layer + '.png'
                            print('img_name',img_name)
                            if cut :
                                input_layer = layer + '_cut'
                            else :
                                input_layer = layer
                            img = cv2.imread(os.path.join(patient, input_layer, 'images', img_name), cv2.IMREAD_GRAYSCALE)
                            img = cv2.resize(img, (304, 304))
                            img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
                            msk = cv2.imread(os.path.join(patient, input_layer, 'masks', img_name), cv2.IMREAD_GRAYSCALE)
                            if img is None or msk is None:
                                print('No image')
                                return 0, 0, 0, 0, 0
                            # if item == 0:
                            #     msk_rm_small = remove_small_area(msk, min_area = 50)
                                
                                
                            # else :
                            #     msk_pre = cv2.imread(os.path.join(patient, 'masks', list(date.keys())[item-1] + '_' + layer + '.png'), cv2.IMREAD_GRAYSCALE)
                            #     msk = msk_pre + msk
                            #     msk[msk > 0] = 255
                            #     msk = msk.astype(np.uint8)
                                # msk_rm_small = remove_small_area(msk, min_area = 50)

                                
                            # # # show msk_rm_small
                            # plt.imshow(msk_rm_small, cmap = 'gray')
                            # plt.show()
                            
                            data_time = datetime.strptime(data_time, '%Y%m%d').date()
                            pretreatment_date = str(patient_record['打針前門診日期'].values[0])
                            if pretreatment_date == 'nan':
                                continue
                            pretreatment_date = parser.parse(pretreatment_date).date()
                            posttreatment_date = str(patient_record['三針後門診'].values[0])
                            if posttreatment_date == 'nan':
                                continue
                            if pd.isna(posttreatment_date):
                                continue
                            posttreatment_date = parser.parse(posttreatment_date).date()
                           
                            if pretreatment_date == data_time:
                                feature['Pre-treatment'] = dict()
                                if mask_roi:
                                    area, center,VD ,VLD ,VAPR,VLA,VDI , FD = process_image(img = img, msk = msk,all_msk = msk_rm_small,img_name = img_name,patient = patient,layer = input_layer)
                                    feature ['Pre-treatment']['VD_' + layer] = VD
                                    feature ['Pre-treatment']['VLD_' + layer] = VLD
                                    feature ['Pre-treatment']['VDI_' + layer] = VDI
                                    feature ['Pre-treatment']['FD_' + layer] = FD
                                feature ['Pre-treatment']['Date'] = list(date.keys())[item]
                                print('Texture feature')
                                if mask_roi:
                                    texture_features = Texture_features(img, msk_rm_small,layer)
                                else:
                                    msk = np.ones(img.shape, dtype=np.uint8) * 255
                                    texture_features = Texture_features(img, msk,layer)
                                
                                feature['Pre-treatment'] = {**feature['Pre-treatment'], **texture_features}
                            

                            elif posttreatment_date == data_time:
                                print('Post-treatment',patient.name,data_time)
                                feature['Post-treatment'] = dict()
                                if mask_roi:
                                    area, center,VD ,VLD ,VAPR,VLA,VDI , FD = process_image(img = img, msk = msk,all_msk = msk_rm_small,img_name = img_name,patient = patient,layer = input_layer)
                                    feature ['Post-treatment'] = dict()
                                    feature ['Post-treatment']['VD_' + layer] = VD
                                    feature ['Post-treatment']['VLD_' + layer] = VLD
                                    feature ['Post-treatment']['VDI_' + layer] = VDI
                                    feature ['Post-treatment']['FD_' + layer] = FD
                                feature ['Post-treatment']['Date'] = list(date.keys())[item]
                                print('Texture feature')
                                if mask_roi:
                                    texture_features = Texture_features(img, msk_rm_small,layer)
                                else:
                                    
                                    msk = np.ones(img.shape, dtype=np.uint8) * 255
                                    
                                    texture_features = Texture_features(img, msk,layer)
                                feature['Post-treatment'] = {**feature['Post-treatment'], **texture_features}
                            

                                    
                    # print('feature',feature)            
                    patient_feature[patient.name] = feature
        print('patient_feature',len(patient_feature))
        print('cout',cout)
        tools.makefolder(os.path.join( 'record', self.disease))
        file = './record/' + self.disease + '/' + save_file_name + '.json'
        tools.write_to_json_file(file, patient_feature)
           
        return patient_feature
            
    # def get_feature_extract(self):
    #     patient_feature = dict()
    #     cout = 0
    #     for patient in pl.Path(os.path.join(self.compare_path) ).iterdir():
    #         patient_feature[patient.name] = dict()
    #         date = dict()
    #         for image in pl.Path(os.path.join(patient, 'images') ).iterdir():
    #             if image.name.endswith(".png"):
    #                 # date_ layer all  exist
    #                 if os.path.exists(os.path.join(patient, 'masks', image.name)) and os.path.exists(os.path.join(patient, 'images', image.name)):
    #                     image_date = image.name.split('_')[0]
    #                     if image_date not in date:
    #                         date[image_date] = 0
    #                     date[image_date] += 1
    #         # date 刪除 value < len(self.layers)
    #         for key in list(date.keys()):
    #             if date[key] < len(self.layers):
    #                 del date[key]
                    
    #         if len(date) < 2:
    #             del patient_feature[patient.name]
    #         else :
    #             # date 排序
    #             date = dict(sorted(date.items(), key=lambda item: item[1], reverse=True)) 
    #             feature = dict()        
    #             for item in range(len(date)):
    #                 if item == 0:
    #                     continue
    #                 else:
    #                     cout += 1
    #                     post_name = list(date.keys())[item] 
    #                     feature[post_name] = dict()
    #                     treatment_count = item
    #                     for layer in self.layers:
    #                         img_pre_name = list(date.keys())[item-1] + '_' + layer + '.png'
    #                         img_post_name = list(date.keys())[item] + '_' + layer + '.png'
                           
    #                        # Morphological feature 提取血管ROI 影像的形態學特徵
    #                        # VDR 血管半徑變化率
    #                         # VLR 血管長度變化率
    #                         # VBR 血管亮度變化率
    #                         # VAR 血管面積變化率
    #                         print('post_name',post_name)
    #                         VDR, VLR, VBR, VAR = vessel_feature(patient,img_pre_name, img_post_name, min_area = 50)
    #                         feature[post_name][treatment_count] = dict()
    #                         # print(F'VDR : {VDR}, VLR : {VLR}, VBR : {VBR}, VAR : {VAR}')
    #                         feature[post_name][treatment_count]['VDR_' + layer] = VDR
    #                         feature[post_name][treatment_count]['VLR_' + layer] = VLR
    #                         feature[post_name][treatment_count]['VBR_' + layer] = VBR
    #                         feature[post_name][treatment_count]['VAR_' + layer] = VAR
                            
                            
    #                         # Curvelet Transform
    #                         # Wavelets(patient, img_pre_name, img_post_name, layer, treatment_count)
                            
                            
    #                         # 形態學特徵會提取血管ROI影像的面積、半徑、周長、輪廓等

    #                         # morphology_features(pre_name = img_pre_name, post_name = img_post_name, patient = patient)
    #                         # Texture feature
                        
    #                         print('Texture feature')
    #                         # Texture_features_ratio(pre_name = img_pre_name, post_name = img_post_name, patient = patient)
                            
    #             patient_feature[patient.name] = feature
    #     print(cout)
    #     file = './record/' + self.disease + '/' + 'VesselFeature.json'
    #     tools.write_to_json_file(file, patient_feature)
    #     return patient_feature

    def relative_feature_extract(self,input_file_name = 'VesselFeature',save_file_name = 'RelativeVesselFeature', mask_roi = True, cut = False):
        treatment = ['Pre-treatment','Post-treatment']
        json_file = './record/' + self.disease + '/' + input_file_name + '.json'
        print(json_file)
        relative_feature = dict()
        if os.path.exists(json_file):
            patient_feature = json.load(open(json_file))
        else :
            patient_feature = self.feature_extract()
        
        for patient in patient_feature :
            if len(patient_feature[patient]) < 2:
                continue
            relative_feature[patient] = dict()
            for feature in  patient_feature[patient][treatment[0]]:
                if feature in patient_feature[patient][treatment[1]]:
                    if feature == 'Date':
                        continue
                    pre_feature = patient_feature[patient][treatment[0]][feature]
                    post_feature = patient_feature[patient][treatment[1]][feature]
                    if post_feature == 0 and pre_feature == 0:
                        relative_feature[patient][feature] = 0
                    else :
                        relative_feature[patient][feature] = (post_feature - pre_feature) * 100 / abs(pre_feature + 1e-8)
        file = './record/' + self.disease + '/' + save_file_name + '.json'
        tools.write_to_json_file(file, relative_feature)
        return relative_feature
                    
                
            
            

def  HOG(img, msk ,layer = 'CC',bin = 8):
    print('HOG')
    orientations = bin 
    img_roi = img.copy()
    img_roi[msk == 0] = 0
    # 使用HOG提取紋理特徵
    from skimage.feature import hog
    
    fd, hog_image = hog(img_roi, orientations=8, pixels_per_cell=(8, 8),
                        cells_per_block=(1, 1), visualize=True)
    # orientations : 梯度方向的數量 範圍為0~360
    # pixels_per_cell : 每個cell的像素數量 表示每個cell的大小
    # cells_per_block : 每個block的cell數量
    # visualize : 是否返回HOG影像
    # multichannel : 是否為多通道影像
    

    hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10))
    
    # # 可视化
    # fig,  ax = plt.subplots(1, 2, figsize=(12, 6), sharex=True, sharey=True)
    # ax[0].imshow(img, cmap=plt.cm.gray)
    # ax[0].set_title('Input image')
    # ax[0].axis('off')
    # ax[1].imshow(hog_image_rescaled, cmap=plt.cm.gray)
    # ax[1].set_title('Histogram of Oriented Gradients')
    # ax[1].axis('off')
    # plt.show()
    
    
    hist, _ = np.histogram(fd, density=True, bins=bin)
    
    # Normalize the features to have a consistent length
    hist = hist.astype("float")
    hist /= (hist.sum() + 1e-7)
    
    # 從HOG特徵中提取紋理特徵
    features = dict()
    mean_value  = np.mean(fd)
    std_value = np.std(fd)
    Skewness = stats.skew(fd)
    Kurtosis = stats.kurtosis(fd)
    features['HOG Mean_' + layer] = mean_value
    features['HOG Std_' + layer] = std_value
    features['HOG Skewness_' + layer] = Skewness
    features['HOG Kurtosis_' + layer] = Kurtosis
    
    # print('features',features)
    
    return features


def TAS(img, msk,layer = 'CC'): # 閾值鄰接矩陣（TAS）
    print('TAS')
    img_roi = img.copy()
    img_roi[msk == 0] = 0
    feature = dict()
    # Threshold Adjacency Matrix
    features, labels = pyfeats.tas_features(img_roi)
    for key , value in zip(labels,features):
        feature[ key + '_' + layer] = value
        
    return feature
    
def GLDS(img, msk,layer = 'CC'): # 灰階差統計（GLDS）
    print('GLDS')
    img_roi = img.copy()
    img_roi[msk == 0] = 0

    features, labels = pyfeats.glds_features(img_roi,msk,Dx=[0 ,1],Dy=[1,0])
    
    '''
    GLDS特徵
    Homogeneity : 一致性
    Contrast : 對比度
    Energy : 能量
    Mean : 平均值
    
    '''
    
    
    feature = dict()
    for key , value in zip(labels,features):
        key = key.split('_')[1]
        print(key,value)
        feature[ key + '_' + layer] = value
        
    return feature    

# 用局部二值模式（Local Binary Pattern，LBP）提取紋理特徵
def LBP(img, msk,layers = 'CC',bin = 8):
    from skimage.feature import local_binary_pattern
    # settings for LBP
    radius = 1  # LBP算法中範圍半徑的取值
    n_points = bin * radius # 領域像素點數
    
    
    
    lbp = local_binary_pattern(img, n_points, radius)
    
    lbp = lbp * msk
    
    # # 視覺化
    # plt.figure(figsize=(12, 4))
    
    # plt.subplot(1, 2, 1)
    # plt.imshow(img, cmap='gray')
    # plt.title('Original Image')
    
    # plt.subplot(1, 2, 2)
    # plt.imshow(lbp, cmap='gray')
    # plt.title('LBP Image')
    
    # plt.show()
    
    # print('lbp.shape',lbp.shape)
    # print(lbp)
    
    # 計算LBP紋理圖像的直方圖
    
    hist, _ = np.histogram(lbp.ravel(), density=True, bins=n_points)
    
    # Normalize the features to have a consistent length
    hist = hist.astype("float")
    hist /= (hist.sum() + 1e-7)
    
    feature = dict()
    feature['LBP mean' + '_'+ layers] = np.mean(lbp)
    feature['LBP std' + '_'+ layers] = np.std(lbp)
    feature['LBP skewness' + '_'+ layers] = stats.skew(hist)
    feature['LBP kurtosis' + '_'+ layers] = stats.kurtosis(hist)
    # feature['LBP entropy' + '_'+ layers] = stats.entropy(hist)
    # feature['LBP energy' + '_'+ layers] = np.sum(hist ** 2)
    # feature['LBP contrast' + '_'+ layers] = np.sum((np.arange(len(hist)) ** 2) * hist)
    # feature['LBP correlation' + '_'+ layers] = np.sum((np.arange(len(hist)) ** 2) * hist) - np.mean(hist) ** 2
    # feature['LBP homogeneity' + '_'+ layers] = np.sum(hist / (1 + np.abs(np.arange(len(hist)) - np.arange(len(hist)).reshape(-1, 1))))
    # feature['LBP dissimilarity' + '_'+ layers] = np.sum(np.abs(np.arange(len(hist)) - np.arange(len(hist)).reshape(-1, 1)) * hist)
    
    
    # print('hist',hist.shape, hist)
    
    
    return feature
 

# Local Gaussian Difference Extrema Pattern
def LGDEP(img, msk,layers = 'CC',bin = 8):
    
    # Gaussian Difference Extrema Pattern
    img_roi = img.copy()
    img_roi[msk == 0] = 0
    
     # Apply Gaussian blur to the image
    blurred = cv2.GaussianBlur(img_roi, (5, 5), 0)
    # Apply LGDEP
    lgp = cv2.Laplacian(blurred, cv2.CV_64F)
    
    # # 視覺化
    # plt.figure(figsize=(12, 4))
    # plt.subplot(1, 2, 1)
    # plt.imshow(img, cmap='gray')
    # plt.title('Original Image')
    
    # plt.subplot(1, 2, 2)
    # plt.imshow(lgp, cmap='gray')
    # plt.title('LGDEP Image')
    
    # plt.show()
    
    # 計算LGDEP紋理圖像的直方圖
    hist, _ = np.histogram(lgp.ravel(), density=True, bins=bin)
    
    # Normalize the features to have a consistent length
    hist = hist.astype("float")
    hist /= (hist.sum() + 1e-7)
    
    # fe
    feature = dict()
    feature['LGDEP mean '+'_'+ layers] = np.mean(lgp)
    feature['LGDEP std' +'_'+ layers] = np.std(lgp)
    feature['LGDEP skewness' +'_'+ layers] = stats.skew(hist)
    feature['LGDEP kurtosis' +'_'+ layers] = stats.kurtosis(hist)
    
    
    return feature


    
# def Wavelets(patient, img_pre_name, img_post_name, layer, treatment_count):
#     img_pre = cv2.imread(os.path.join(patient, 'images', img_pre_name), cv2.IMREAD_GRAYSCALE)
#     img_post = cv2.imread(os.path.join(patient, 'images', img_post_name), cv2.IMREAD_GRAYSCALE)
#     msk_pre = cv2.imread(os.path.join(patient, 'masks', img_pre_name), cv2.IMREAD_GRAYSCALE)
#     msk_post = cv2.imread(os.path.join(patient, 'masks', img_post_name), cv2.IMREAD_GRAYSCALE)
    
#     if img_pre is None or img_post is None or msk_pre is None or msk_post is None:
#         print('No image')
#         return 0, 0, 0, 0, 0
    
#     HOGL, hog_image = HOG(img_pre)
    
#     msk = msk_pre + msk_post
#     msk[msk > 0] = 255
#     msk = msk.astype(np.uint8)
#     msk_rm_small = remove_small_area(msk, min_area = 50)
    
#     img = img_pre.copy()
#     img[msk_rm_small == 0] = 0
    
    
#     # 使用 Daubechies 小波變換
#     coeffs = pywt.dwt2(img, 'db1')
    
#    # Step 4: Threshold the detail coefficients to remove noise
#     threshold = 0.1  # Adjust this threshold value based on your requirements
#     coeffs = tuple(map(lambda x: pywt.threshold(x, threshold, mode='soft'), coeffs))

#     # Step 5: Reconstruct the denoised image
#     denoised_img = pywt.idwt2(coeffs, 'db1')

#     # 可视化
#     plt.figure(figsize=(12, 4))

#     plt.subplot(1, 2, 1)
#     plt.imshow(img, cmap='gray')
#     plt.title('Original Image')

#     plt.subplot(1, 2, 2)
#     plt.imshow(denoised_img, cmap='gray')
#     plt.title('Denoised Image')

#     plt.show()

#     # # coeffs 包含了逼近系数（LL子图）和细节系数（LH、HL、HH子图）
#     LL, (LH, HL, HH) = coeffs

#     # 可视化
#     plt.figure(figsize=(12, 4))

#     plt.subplot(2, 2, 1)
#     plt.imshow(LL, cmap='gray')
#     plt.title('Approximation (LL)')

#     plt.subplot(2, 2, 2)
#     plt.imshow(LH, cmap='gray')
#     plt.title('Horizontal Detail (LH)')

#     plt.subplot(2, 2, 3)
#     plt.imshow(HL, cmap='gray')
#     plt.title('Vertical Detail (HL)')

#     plt.subplot(2, 2, 4)
#     plt.imshow(HH, cmap='gray')
#     plt.title('Diagonal Detail (HH)')

#     plt.show()
    
#     feature = dict()
#     return feature

# 紋理分析 治療變化

def Texture_features_ratio(pre_name, post_name, patient):
    img_pre = cv2.imread(os.path.join(patient, 'images', pre_name), cv2.IMREAD_GRAYSCALE)
    img_post = cv2.imread(os.path.join(patient, 'images', post_name), cv2.IMREAD_GRAYSCALE)
    msk_pre = cv2.imread(os.path.join(patient, 'masks', pre_name), cv2.IMREAD_GRAYSCALE)
    msk_post = cv2.imread(os.path.join(patient, 'masks', post_name), cv2.IMREAD_GRAYSCALE)
    
    if img_pre is None or img_post is None or msk_pre is None or msk_post is None:
        print('No image')
        return 0, 0, 0, 0, 0
    # msk_pre 聯集 msk_post
    msk  = msk_pre + msk_post 
    msk[msk > 0] = 255
    msk = msk.astype(np.uint8)
    msk_rm_small = remove_small_area(msk, min_area = 50)

    img_pre_roi = img_pre.copy()
    img_pre_roi[msk_rm_small == 0] = 0
    
    img_post_roi = img_post.copy()
    img_post_roi[msk_rm_small == 0] = 0
    
    feature_pre = Texture_features(img_pre_roi, msk_rm_small)
    feature_post = Texture_features(img_post_roi, msk_rm_small)
    
 
def Texture_features(img, msk,layer = 'CC'):
    
    feature = dict()
    distance = [1, 2, 3, 4, 5]
    # https://pyradiomics.readthedocs.io/en/latest/index.html
    # print('GLCM')
    img_roi = img.copy()
    img_roi[msk == 0] = 0
    
    # 只對ROI做GLCM
    
    if img_roi.max() == 0:
        return feature
    
    GLCM_feature = GLCM(img , msk,layer)
    GLCM_feature = {'GLCM_' + key : GLCM_feature[key] for key in GLCM_feature}
    # feature 整合
    feature= {**feature, **GLCM_feature}
    
    # print('GLRLM')
    GLRLM_feature = GLRLM(img , msk ,layer)
    GLRLM_feature = {'GLRLM_' + key : GLRLM_feature[key] for key in GLRLM_feature}
    feature = {**feature, **GLRLM_feature}

    
    # print('GLSZM')
    GLSZM_feature = GLSZM(img , msk,layer)
    GLSZM_feature = {'GLSZM_' + key : GLSZM_feature[key] for key in GLSZM_feature}
    feature = {**feature, **GLSZM_feature}
    
    # print('GLDM')
    GLDM_feature = GLDM(img , msk,layer)
    GLDM_feature = {'GLDM_' + key : GLDM_feature[key] for key in GLDM_feature}
    feature = {**feature, **GLDM_feature}
    
    # print('NGTDM')
    NGTDM_feature = NGTDM(img , msk,layer)
    NGTDM_feature = {'NGTDM_' + key : NGTDM_feature[key] for key in NGTDM_feature}
    feature = {**feature, **NGTDM_feature}
    
    # # # # print('Statistical Feature Matrix')
    # #  SFM 特徵如下：1) 粗糙度，2) 對比度，3) 週期性，4) 粗糙度。
    SFM_feature = SFM(img , msk,layer)
    SFM_feature = {'SFM_' + key : SFM_feature[key] for key in SFM_feature}
    feature = {**feature, **SFM_feature}
    
    # # FDTA
    # print('FDTA')
    FDTA_feature = FDTA(img , msk,layer)
    feature = {**feature, **FDTA_feature}
    
    # FPS
    # print('FPS')
    FPS_feature = FPS(img , msk,layer)
    feature = {**feature, **FPS_feature}
    
    # # HOG
    # print('HOG')
    fd  = HOG(img, msk,layer )
    feature= {**feature, **fd}
    
    
    # GLDS_feature = GLDS(img , msk,layer) # 'GLDS_Homogeneity', 'GLDS_Contrast', 'GLDS_ASM', 'GLDS_Entopy', 'GLDS_Mean'
    # feature = {**feature, **GLDS_feature} 
    
    #     # # print('LBP')
    lbp= LBP(img, msk,layer)
    feature = {**feature, **lbp}
    

    
    # # # # print('FirstOrder')
    firstOrder_feature = FOS(img , msk,layer)
    feature = {**feature, **firstOrder_feature}

    
    
    # # print('DWT')
    EHOG_feature = DWT(img , msk , layer)
    feature = {**feature, **EHOG_feature}
    
    
    
    
    # print('LGDEP')
    lgdep = LGDEP(img, msk,layer)
    feature= {**feature, **lgdep}
    
    # # print('LTEM')
    # ltem = LTEM(img, msk)
    # feature = {**feature, **ltem}
    
    # print('HuMoments')
    hu = HuMoments(img, msk)
    feature = {**feature, **hu}
    
    # # print('TAS')
    # tas = TAS(img, msk)
    # feature = {**feature, **tas}
    
    #print('WaveletPackets')
    # wavelet = WaveletPackets(img, msk)
    # feature = {**feature, **wavelet}
    
    # print('GaborTransform')
    # gabor = GaborTransform(img, msk)
    # feature = {**feature, **gabor}
    
    # print('AMFM')
    # amfm = AMFM(img, msk)
    # feature = {**feature, **amfm}
    
    # print('ZernikesMoments')
    zernikes = ZernikesMoments(img, msk)
    feature = {**feature, **zernikes}
    
        
    return feature

    # EHOG_pre, hog_image_pre = EHOG(img_pre_roi)
   
def Texture_features3D(pre_name, post_name, patient):
    img_pre = cv2.imread(os.path.join(patient, 'images', pre_name), cv2.IMREAD_GRAYSCALE)
    img_post = cv2.imread(os.path.join(patient, 'images', post_name), cv2.IMREAD_GRAYSCALE)
    msk_pre = cv2.imread(os.path.join(patient, 'masks', pre_name), cv2.IMREAD_GRAYSCALE)
    msk_post = cv2.imread(os.path.join(patient, 'masks', post_name), cv2.IMREAD_GRAYSCALE)
    pre_name.replace('CC', 'OR')
    post_name.replace('CC', 'OR')
    img_pre2 = cv2.imread(os.path.join(patient, 'images', pre_name), cv2.IMREAD_GRAYSCALE)
    img_post2 = cv2.imread(os.path.join(patient, 'images', post_name), cv2.IMREAD_GRAYSCALE)
    msk_pre2 = cv2.imread(os.path.join(patient, 'masks', pre_name), cv2.IMREAD_GRAYSCALE)
    msk_post2 = cv2.imread(os.path.join(patient, 'masks', post_name), cv2.IMREAD_GRAYSCALE)
    
    if img_pre is None or img_post is None or msk_pre is None or msk_post is None:
        print('No image')
        return 0, 0, 0, 0, 0
    
    msk = msk_pre + msk_post
    msk[msk > 0] = 255
    msk = msk.astype(np.uint8)
    msk_rm_small = remove_small_area(msk, min_area = 50)
    img_pre_roi = img_pre.copy()
    img_pre_roi[msk_rm_small == 0] = 0
    
    img_post_roi = img_post.copy()
    img_post_roi[msk_rm_small == 0] = 0
    
    msk2 = msk_pre2 + msk_post2
    msk2[msk2 > 0] = 255
    msk2 = msk2.astype(np.uint8)
    msk_rm_small2 = remove_small_area(msk2, min_area = 50)
    img_pre_roi2 = img_pre2.copy()
    img_pre_roi2[msk_rm_small2 == 0] = 0
    
    img_post_roi2 = img_post2.copy()
    img_post_roi2[msk_rm_small2 == 0] = 0
    
    # concat
    img_pre_roi3D = np.stack((img_pre_roi, img_pre_roi2), axis=2)
    img_post_roi3D = np.stack((img_post_roi, img_post_roi2), axis=2)
    
    # OS 特徵如下：1) 平均值、2) 標準差、3) 中位數、4) 眾數、5) skewnewss、6) 峰度、7) 能量、8) 熵、9) 最小灰階、10)最大灰階水平，11) 變異係數，12,13,14,15) 百分位數 (10, 25, 50, 75, 90) 和 16) 直方圖寬度。
    
    features, labels = pyfeats.fos(img_pre, msk_pre)
    print(features)
    
    distance = [1, 2, 3, 4, 5]
    feature = GLCM(img_pre_roi, distance)
    feature = GLCM3D(img_pre_roi)
 
 # DWT co-efficient

def LTEM(img, msk,layer = 'CC'):
    img_roi = img.copy()
    img_roi[msk == 0] = 0
    
    # Threshold Adjacency Matrix
    features, labels = pyfeats.lte_measures(img,msk,l=7)
    
    '''
    Law's Texture Energy Measures (LTEM) 
    LL : Low-Low (LL) texture energy measure 
    EE : Edge-Edge (EE) texture energy measure
    SS : Spot-Spot (SS) texture energy measure
    LS : Low-Spot (LS) texture energy measure
    LE : Low-Edge (LE) texture energy measure
    ES : Edge-Spot (ES) texture energy measure
    
    '''
    
    
    
    feature = dict()
    for key, value in zip(labels, features):
        key = key.split('_')[1]
        feature[key + '_' + layer] = value
        
    return feature

def HuMoments(img, msk,layer = 'CC'):
    img_roi = img.copy()
    img_roi[msk == 0] = 0
    feature = dict()
    # Hu Moments
    features, labels = pyfeats.hu_moments(img_roi)
    for key, value in zip(labels, features):
        feature[key + '_' + layer] = value
        
    return feature


def FPS(img, msk,layer = 'CC'):
    print('FPS') # Fourier Power Spectrum (FPS)
    img_roi = img.copy()
    img_roi[msk == 0] = 0
    
    # FPS
    features, labels = pyfeats.fps(img_roi,msk)
    '''
    Fourier Power Spectrum (FPS)
    RadialSum : Radial sum
    AngularSum : Angular sum
    '''
    
    feature = dict()
    for key, value in zip(labels, features):
        key = key.split('_')[1]
        feature[ key + '_' + layer] = value
        
    return feature
    
   

def FDTA(img, msk,layer = 'CC'):
    print('FDTA')
    img_roi = img.copy()
    img_roi[msk == 0] = 0
    
    # FDTA
    features, labels = pyfeats.fdta(img_roi,msk)
    '''
    Fractal Dimension Texture Analysis (FDTA)
    HurstCoeff : Hurst coefficient 
    HurstCoeff 1 : Hurst coefficient 1
    HurstCoeff 2 : Hurst coefficient 2
    
    
    '''

    feature = dict()
    for key, value in zip(labels, features):
        f = key.split('_')
        key = f[1] + '_' + f[2]
        feature[ key + '_' + layer] = value
        
    return feature

def DWT(img,msk, layer = 'CC'):
    # 特徵提取 - 小波變換
    img_roi = img.copy()
    img_roi[msk == 0] = 0

    feature = dict()
    # 使用 Daubechies 小波變換 僅 對msk內的影像進行小波變換
    coeffs = pywt.dwt2(img_roi, 'db1')
    # coeffs 包含了逼近系数（LL子图）和细节系数（LH、HL、HH子图）
    LL, (LH, HL, HH) = coeffs
    
    # # show LH, HL, HH
    # plt.figure(figsize=(12, 4))
    # plt.subplot(1, 4 ,1)
    # plt.imshow(LL, cmap='gray')
    # plt.title('Approximation (LL)')
    # plt.axis('off')
    # plt.subplot(1, 4, 2)
    # plt.imshow(LH, cmap='gray')
    # plt.title('Horizontal Detail (LH)')
    # plt.axis('off')
    # plt.subplot(1, 4, 3)
    # plt.imshow(HL, cmap='gray')
    # plt.title('Vertical Detail (HL)')
    # plt.axis('off')
    # plt.subplot(1, 4, 4)
    # plt.imshow(HH, cmap='gray')
    # plt.title('Diagonal Detail (HH)')
    # plt.axis('off')
    # plt.show()
    

    non_zero_abs_LH = np.abs(LH[LH != 0])
    non_zero_abs_HL = np.abs(HL[HL != 0])
    non_zero_abs_HH = np.abs(HH[HH != 0])
    # :H=Horizontal, V=Vertical,D=Diagonal
    # avg Dh1 , Dh2,Dv2
    avg_LH = np.mean(non_zero_abs_LH)
    avg_HL = np.mean(non_zero_abs_HL)
   
    
    # std Dh1 , Dh2,Dv2
    std_LH = np.std(non_zero_abs_LH, ddof=1)
    std_HL = np.std(non_zero_abs_HL, ddof=1)
    
    
    
    
    feature['avg_LH_' + layer] = avg_LH
    feature['avg_HL_' + layer] = avg_HL
   
    feature['std_LH_' + layer] = std_LH
    feature['std_HL_' + layer] = std_HL
  
    
    
    
    
    
    


   
    return feature

def SWT(img,msk, layer = 'CC'):
    # 特徵提取 - 小波變換
    img_roi = img.copy()
    img_roi[msk == 0] = 0

    feature = dict()
    # 使用 Daubechies 小波變換 僅 對msk內的影像進行小波變換
    coeffs = pywt.swt2(img_roi, 'db1')
    # coeffs 包含了逼近系数（LL子图）和细节系数（LH、HL、HH子图）
    LL, (LH, HL, HH) = coeffs
    
    # # show LH, HL, HH
    # plt.figure(figsize=(12, 4))
    # plt.subplot(1, 3, 1)
    # plt.imshow(LH, cmap='gray')
    # plt.title('LH')
    # plt.subplot(1, 3, 2)
    # plt.imshow(HL, cmap='gray')
    # plt.title('HL')
    # plt.subplot(1, 3, 3)
    # plt.imshow(HH, cmap='gray')
    # plt.title('HH')
    # plt.show()
    features, labels = pyfeats.dwt_features(img, msk, wavelet='db1')
    for i in range(len(features)):
        feature[labels[i] + '_' + layer] = features[i]
    # LH 水平細節


   
    return feature


def WaveletPackets(img,msk, layer = 'CC'):
    print('WaveletPackets')
    # 特徵提取 - 小波變換
    img_roi = img.copy()
    img_roi[msk == 0] = 0

    feature = dict()

    features, labels = pyfeats.wp_features(img_roi, msk)
    for key, value in zip(labels, features):
        feature[key + '_' + layer] = value
    # LH 水平細節


   
    return feature

def GaborTransform(img, msk, layer = 'CC'):
    print('GaborTransform')
    # 特徵提取 - Gabor變換
    img_roi = img.copy()
    img_roi[msk == 0] = 0

    feature = dict()
    # 使用 Gabor Transform 小波變換 僅 對msk內的影像進行小波變換
    features, labels = pyfeats.gt_features(img, msk)
    for key, value in zip(labels, features):
        feature[key + '_' + layer] = value
    # LH 水平細節

    return feature

def AMFM(img, msk, layer = 'CC'):
    print('AMFM')
    # 特徵提取 - AMFM變換
    img_roi = img.copy()
    img_roi[msk == 0] = 0

    feature = dict()
    # Amplitude Modulation – Frequency Modulation (AM-FM) Transform
    features, labels = pyfeats.amfm_features(img_roi, bins = 8)
    for key, value in zip(labels, features):
        print(key, value)
        feature[key + '_' + layer] = value
    # LH 水平細節

    return feature

def ZernikesMoments(img, msk, layer = 'CC'):
    print('ZernikesMoments')
    # 特徵提取 - ZernikesMoments
    img_roi = img.copy()
    img_roi[msk == 0] = 0

    feature = dict()
    # 使用 Gabor Transform 小波變換 僅 對msk內的影像進行小波變換
    features, labels = pyfeats.zernikes_moments(img_roi)
    for key, value in zip(labels, features):
        feature[key + '_' + layer] = value

    return feature

def GLCM(img, msk,layer = 'CC'):
    print('GLCM')
    layer_name = {'CC':'Choriocapillaris','OR':'Outer retina'}
    img_roi = img.copy()
    img_roi[msk == 0] = 0
    
    features = dict()
    msk [ msk > 0 ] = 1
    # Convert NumPy array to SimpleITK image
    image_sitk = sitk.GetImageFromArray(img_roi)
    mask_sitk = sitk.GetImageFromArray(msk)
    
    glcm_features = glcm.RadiomicsGLCM(image_sitk, mask_sitk, **{'distance': [1], 'symmetric': True, 'normed': True})
    glcm_features.execute()

    '''
    GLCM 灰度共生矩陣
    用於紋理分析的統計特徵
    可以補捉影像中空間鄰近像素之間的關係
    '''
    '''
    
    Autocorrelation  # 自相關 
    Autocorrelation is a measure of the magnitude of the fineness and coarseness of texture. 
    紋理精細度和粗糙度大小的測量
    公式:  Autocorrelation = sum(sum(p(i,j) * i * j))
    
    Joint Average  # 聯合平均值 
    Returns the mean gray level intensity of the idistribution.
    測量了像素值的分散程度
    公式: Joint Average = sum(sum(p(i,j) * i * j))
    
    Cluster Prominence  # 簇降順 
    Cluster Prominence is a measure of the skewness and asymmetry of the GLCM
    A higher values implies more asymmetry about the mean while a lower value indicates a peak near the mean value and less variation about the mean.
    偏度和不對稱性的量測
    公式: Cluster Prominence = sum(sum(p(i,j) * (i - u) * (j - u)))
    
    Cluster Shade  # 簇陰影
    Cluster Shade is a measure of the skewness and uniformity of the GLCM. 
    A higher value indicates a greater asymmetry in the distribution while a lower value indicates a symmetrical distribution.
    偏度和均勻性的量測
    公式: Cluster Shade = sum(sum(p(i,j) * (i + j - 2 * u) ** 3))
    
    Cluster Tendency  # 簇趨勢 
    Cluster Tendency is a measure of groupings of voxels with similar gray-level values.
    A higher value indicates more tendency to form clusters.
    群集中具有相似灰度值的像素的量測
    公式: Cluster Tendency = sum(sum(p(i,j) * (i + j - 2 * u) ** 2))
    
    Contrast  # 對比度 
    Contrast is a measure of the local variations in the gray-level co-occurrence matrix.
    測量了像素值的對比度
    公式: Contrast = sum(sum(p(i,j) * (i - j) ** 2))
    
    Correlation  # 相關性 
    Correlation is a measure of the joint probability occurrence of the specified pixel pairs.
    測量了像素值的相似性 
    公式: Correlation = sum(sum(p(i,j) * (i - u) * (j - u) / (sigma(i) * sigma(j))))
    
    Difference Average  # 差異平均值  => Dissimilarity
    Difference Average is a measure of the difference intensity between the specified pixel pairs.
    測量了像素值的差異
    公式: Difference Average = sum(sum(p(i,j) * abs(i - j)))
    

    Difference Entropy  # 差異熵 
    Difference Entropy is a measure of the randomness/variability in neighborhood intensity value differences.
    測量了像素值的隨機性
    公式: Difference Entropy = - sum(sum(p(i,j) * log2(p(i,j) + eps)))
    
    DifferenceVariance  # 差異變異 
    Difference Variance is a measure of heterogeneity that places higher weights on differing intensity level pairs that deviate more from the mean.
    測量了像素值的差異
    公式: Difference Variance = sum(sum(p(i,j) * (i - j) ** 2))
    
    Joint Energy # 聯合能量
    Energy is a measure of homogeneous patterns in the image. 
    A greater Energy implies that there are more instances of intensity value pairs in the image that neighbor each other at higher frequencies.
    影像中均勻圖案的量測
    公式: Joint Energy = sum(sum(p(i,j) ** 2))
    
    Joint Entropy # 聯合熵
    Joint entropy is a measure of the randomness/variability in neighborhood intensity values.
    鄰域強度值的隨機性/可變性的量測。
    公式: Joint Entropy = - sum(sum(p(i,j) * log2(p(i,j) + eps)))
    
    Informational Measure of Correlation (IMC) 1 
    IMC1 assesses the correlation between the probability distributions of iand j
    (quantifying the complexity of the texture), using mutual information I(x, y)
    沒有互資訊，因此結果將為 0 
    在完全依賴的均勻分佈的情況下，互資訊將等於 log2(Ng)
    
    Informational Measure of Correlation (IMC) 2
    IMC2 also assesses the correlation between the probability distributions of iand j
    (quantifying the complexity of the texture).
    值得注意的是 HXY1=HXY2和 HXY2−HXY≥0代表兩個分佈的互資訊。
    因此，IMC2的範圍=[0, 1)，其中0代表2個獨立分佈（無互資訊）的情況，最大值代表2個完全依賴且均勻分佈的情況（最大互信息，等於 log2(Ng)，接近 1。

    Inverse Difference Moment (IDM)  # ID ：測量了像素值的分散程度 => Homogeneity
    Inverse Difference Moment is a measure of the local homogeneity of an image.
    影像局部同質性的量測 
    公式: Inverse Difference Moment = sum(sum(p(i,j) / (1 + (i - j) ** 2)))
    
    Maximal Correlation Coefficient (MCC)
    The Maximal Correlation Coefficient is a measure of complexity of the texture and 0≤MCC≤1.
    紋理的複雜度的量測
    公式: MCC = (HXY1 - HXY2) / max(HX, HY)
    
    Inverse Difference Moment Normalized (IDMN)
    IDMN (inverse difference moment normalized) is a measure of the local homogeneity of an image.
    IDMN 權重是對比權重的倒數
    公式: IDMN = sum(sum(p(i,j) / (1 + (i - j) ** 2) / Ng ** 2))
    
    Inverse Difference (ID) 
    is another measure of the local homogeneity of an image. With more uniform gray levels, the denominator will remain low, resulting in a higher overall value.
    隨著灰階更加均勻，分母將保持較低，從而產生更高的總體值。
    
    
    Inverse Difference Normalized (IDN) 
    is a measure of the local homogeneity of an image. With more uniform gray levels, the denominator will remain low, resulting in a higher overall value.
    IDN 透過除以離散強度值的總數來標準化相鄰強度值之間的差異。
    
    Inverse Variance
    Inverse Variance is a measure of the local homogeneity of an image.
    
    Maximum Probability
    Maximum Probability is occurrences of the most predominant pair of neighboring intensity values.
    
    Sum Average
    Sum Average measures the relationship between occurrences of pairs with lower intensity values and occurrences of pairs with higher intensity values.
    
    Sum Entropy 
    Sum Entropy is a sum of neighborhood intensity value differences.
    
    Sum of Squares
    Sum of Squares is a measure of the local homogeneity of an image.
    

    '''
    feature_names = {
        'Autocorrelation': 'Autocorrelation',
        'JointAverage': 'Joint Average',
        'ClusterProminence': 'Cluster Prominence',
        'ClusterShade': 'Cluster Shade',
        'ClusterTendency': 'Cluster Tendency',
        'Contrast': 'Contrast',
        'Correlation': 'Correlation',
        'DifferenceAverage': 'Difference Average',
        'DifferenceEntropy': 'Difference Entropy',
        'DifferenceVariance': 'Difference Variance',
        'JointEnergy': 'Joint Energy',
        'JointEntropy': 'Joint Entropy',
        'IMC1': 'Informational Measure of Correlation (IMC) 1',
        'IMC2': 'Informational Measure of Correlation (IMC) 2',
        'IDM': 'Inverse Difference Moment (IDM)',
        'MCC': 'Maximal Correlation Coefficient (MCC)',
        'IDMN': 'Inverse Difference Moment Normalized (IDMN)',
        'ID': 'Inverse Difference (ID)',
        'IDN': 'Inverse Difference Normalized (IDN)',
        'InverseVariance': 'Inverse Variance',
        'MaximumProbability': 'Maximum Probability',
        'SumAverage': 'Sum Average',
        'SumEntropy': 'Sum Entropy',
        'SumofSquares': 'Sum of Squares',
        'SumVariance': 'Sum Variance'  
    }
    rename = {'DifferenceAverage':'Dissimilarity',
              'Idm':'Homogeneity',
              'Idn':'IDN',
              'Idmn':'IDMN',
              'Imc2': 'IMC2',
              'Imc1': 'IMC1',
              'Id':'ID'}
                
    
    for (key, val) in six.iteritems(glcm_features.featureValues):
        # features[feature_names[key] + ' in the Choriocapillaris' + layer] = val
        
        if key in rename.keys():
            features[rename[key] + '_' + layer] = val
        else:
            features[key + '_' + layer] = val
    return features
                          
    
def GLRLM(img, mask, layer = 'CC'):
    print('GLRLM')
    layer_name = {'CC':'Choriocapillaris','OR':'Outer retina'}
    mask [ mask > 0 ] = 1
    # Convert NumPy array to SimpleITK image
    image_sitk = sitk.GetImageFromArray(img)
    mask_sitk = sitk.GetImageFromArray(mask)
    # Ensure the image and mask have the same size
    if image_sitk.GetSize() != mask_sitk.GetSize():
        raise ValueError("Image and mask must have the same size.")


    glrlmFeatures = glrlm.RadiomicsGLRLM(image_sitk, mask_sitk )
    glrlmFeatures.enableAllFeatures()
    glrlmFeatures.execute()
    
    '''
    GLRLM : 灰度長度矩陣 (GLRLM) 特徵是用於表徵影像中特定灰階的大小區域的分佈，這些區域的大小和灰階值是相關的。
    這些特徵可以用於表徵影像中的細微紋理特徵，例如細胞核的大小和形狀。
    GrayLevelNonUniformity # 灰度級非均勻性 ：測量了像素值的分散程度
    GrayLevelNonUniformityNormalized # 正規化灰度級非均勻性 ：測量了像素值的分散程度
    GrayLevelVariance # 灰度級變異 ：測量了像素值的分散程度
    HighGrayLevelRunEmphasis # 高灰度級運行強度 ：測量了像素值的分散程度
    LongRunEmphasis # 長運行強度 ：測量了像素值的分散程度
    LongRunHighGrayLevelEmphasis # 長運行高灰度級強度 ：測量了像素值的分散程度
    LongRunLowGrayLevelEmphasis # 長運行低灰度級強度 ：測量了像素值的分散程度
    LowGrayLevelRunEmphasis # 低灰度級運行強度 ：測量了像素值的分散程度
    RunEntropy # 運行熵 ：測量了像素值的分散程度
    RunLengthNonUniformity # 運行長度非均勻性 ：測量了像素值的分散程度
    RunLengthNonUniformityNormalized # 正規化運行長度非均勻性 ：測量了像素值的分散程度
    RunPercentage # 運行百分比 ：測量了像素值的分散程度
    RunVariance # 運行變異 ：測量了像素值的分散程度
    ShortRunEmphasis # 短運行強度 ：測量了像素值的分散程度
    ShortRunHighGrayLevelEmphasis # 短運行高灰度級強度 ：測量了像素值的分散程度
    ShortRunLowGrayLevelEmphasis # 短運行低灰度級強度 ：測量了像素值的分散程度
    
    '''
    feature_names = {
        'GrayLevelNonUniformity': 'Gray Level Non-Uniformity',
        'GrayLevelNonUniformityNormalized': 'Normalized Gray Level Non-Uniformity',
        'GrayLevelVariance': 'Gray Level Variance',
        'HighGrayLevelRunEmphasis': 'High Gray Level Run Emphasis',
        'LongRunEmphasis': 'Long Run Emphasis',
        'LongRunHighGrayLevelEmphasis': 'Long Run High Gray Level Emphasis',
        'LongRunLowGrayLevelEmphasis': 'Long Run Low Gray Level Emphasis',
        'LowGrayLevelRunEmphasis': 'Low Gray Level Run Emphasis',
        'RunEntropy': 'Run Entropy',
        'RunLengthNonUniformity': 'Run Length Non-Uniformity',
        'RunLengthNonUniformityNormalized': 'Normalized Run Length Non-Uniformity',
        'RunPercentage': 'Run Percentage',
        'RunVariance': 'Run Variance',
        'ShortRunEmphasis': 'Short Run Emphasis',
        'ShortRunHighGrayLevelEmphasis': 'Short Run High Gray Level Emphasis',
        'ShortRunLowGrayLevelEmphasis': 'Short Run Low Gray Level Emphasis'
    }
    
        
    
    
    feature = dict()
    
    for (key, val) in six.iteritems(glrlmFeatures.featureValues):
        # feature[feature_names[key] + ' in the ' + layer_name[layer]] = val
        feature[key + '_' + layer] = val
        
    return feature
        
def GLSZM(img, mask, layer = 'CC'):
    print('GLSZM')
    layer_name = {'CC':'Choriocapillaris','OR':'Outer retina'}
    mask [ mask > 0 ] = 1
    # Convert NumPy array to SimpleITK image
    image_sitk = sitk.GetImageFromArray(img)
    mask_sitk = sitk.GetImageFromArray(mask)

    glszmFeatures = glszm.RadiomicsGLSZM(image_sitk, mask_sitk )
    glszmFeatures.enableAllFeatures()
    glszmFeatures.execute()
    '''
    用於表徵影像中特定灰階的大小區域的分佈，這些區域的大小和灰階值是相關的。
    這些特徵可以用於表徵影像中的細微紋理特徵，例如細胞核的大小和形狀。
    GrayLevelNonUniformity # 灰度級非均勻性 ：測量了像素值的分散程度
    GrayLevelNonUniformityNormalized # 正規化灰度級非均勻性 ：測量了像素值的分散程度
    GrayLevelVariance # 灰度級變異 ：測量了像素值的分散程度 
    HighGrayLevelZoneEmphasis # 高灰度級區域強度 ：測量了像素值的分散程度
    LargeAreaEmphasis # 大面積強度 ：測量了像素值的分散程度
    LargeAreaHighGrayLevelEmphasis # 大面積高灰度級強度 ：測量了像素值的分散程度
    LargeAreaLowGrayLevelEmphasis # 大面積低灰度級強度 ：測量了像素值的分散程度
    LowGrayLevelZoneEmphasis # 低灰度級區域強度 ：測量了像素值的分散程度
    SizeZoneNonUniformity # 尺寸區域非均勻性 ：測量了像素值的分散程度
    SizeZoneNonUniformityNormalized # 正規化尺寸區域非均勻性 ：測量了像素值的分散程度
    SmallAreaEmphasis # 小面積強度 ：測量了像素值的分散程度
    SmallAreaHighGrayLevelEmphasis # 小面積高灰度級強度 ：測量了像素值的分散程度
    SmallAreaLowGrayLevelEmphasis # 小面積低灰度級強度 ：測量了像素值的分散程度
    ZoneEntropy # 區域熵 ：測量了像素值的分散程度
    ZonePercentage # 區域百分比 ：測量了像素值的分散程度
    ZoneVariance # 區域變異 ：測量了像素值的分散程度

    '''

    feature = dict()
    for (key, val) in six.iteritems(glszmFeatures.featureValues):
        feature[key + '_' + layer] = val
    
    return feature

def GLDM(img, mask, layer = 'CC'):
    print('GLDM')
    mask [ mask > 0 ] = 1
    # Convert NumPy array to SimpleITK image
    image_sitk = sitk.GetImageFromArray(img)
    mask_sitk = sitk.GetImageFromArray(mask)

    gldmFeatures = gldm.RadiomicsGLDM(image_sitk, mask_sitk )
    gldmFeatures.enableAllFeatures()
    gldmFeatures.execute()
    
    '''
    GLDM : 灰度差異矩陣 (GLDM) 特徵是用於表徵影像中特定灰階的大小區域的分佈，這些區域的大小和灰階值是相關的。
    這些特徵可以用於表徵影像中的細微紋理特徵，例如細胞核的大小和形狀。
    DependenceEntropy # 依賴熵 ：測量了像素值的分散程度
    DependenceNonUniformity # 依賴非均勻性 ：測量了像素值的分散程度
    DependenceNonUniformityNormalized # 正規化依賴非均勻性 ：測量了像素值的分散程度
    DependenceVariance # 依賴變異 ：測量了像素值的分散程度
    GrayLevelNonUniformity # 灰度級非均勻性 ：測量了像素值的分散程度
    GrayLevelVariance # 灰度級變異 ：測量了像素值的分散程度
    HighGrayLevelEmphasis # 高灰度級強度 ：測量了像素值的分散程度
    LargeDependenceEmphasis # 大依賴強度 ：測量了像素值的分散程度
    LargeDependenceHighGrayLevelEmphasis # 大依賴高灰度級強度 ：測量了像素值的分散程度
    LargeDependenceLowGrayLevelEmphasis # 大依賴低灰度級強度 ：測量了像素值的分散程度
    LowGrayLevelEmphasis # 低灰度級強度 ：測量了像素值的分散程度
    SmallDependenceEmphasis # 小依賴強度 ：測量了像素值的分散程度
    SmallDependenceHighGrayLevelEmphasis # 小依賴高灰度級強度 ：測量了像素值的分散程度
    SmallDependenceLowGrayLevelEmphasis # 小依賴低灰度級強度 ：測量了像素值的分散程度
    
    '''
    
    feature = dict()
    for (key, val) in six.iteritems(gldmFeatures.featureValues):

        feature[key + '_' + layer] = val
    
    return feature
 
def NGTDM(img, mask, layer = 'CC'):
    print('NGTDM')
    mask [ mask > 0 ] = 1
    # Convert NumPy array to SimpleITK image
    image_sitk = sitk.GetImageFromArray(img)
    mask_sitk = sitk.GetImageFromArray(mask)

    ngtdmFeatures = ngtdm.RadiomicsNGTDM(image_sitk, mask_sitk )
    ngtdmFeatures.enableAllFeatures()
    ngtdmFeatures.execute()
    '''
    NGTDM : 灰度差異矩陣 (GLDM) 特徵是用於表徵影像中特定灰階的大小區域的分佈，這些區域的大小和灰階值是相關的。
    這些特徵可以用於表徵影像中的細微紋理特徵，例如細胞核的大小和形狀。
    Coarseness # 粗糙度 ：測量了像素值的分散程度
    Contrast # 對比度 ：測量了像素值的分散程度
    Busyness # 繁忙度 ：測量了像素值的分散程度
    Complexity # 複雜度 ：測量了像素值的分散程度
    Strength # 強度 ：測量了像素值的分散程度
    '''
    feature = dict()
    for (key, val) in six.iteritems(ngtdmFeatures.featureValues):
        feature[key + '_' + layer] = val
    
    
    return feature         

# First Order Features
def FOS(img, mask, layer = 'CC'):
    mask [ mask > 0 ] = 1
    # Convert NumPy array to SimpleITK image
    image_sitk = sitk.GetImageFromArray(img)
    mask_sitk = sitk.GetImageFromArray(mask)
    # Ensure the image and mask have the same size
    if image_sitk.GetSize() != mask_sitk.GetSize():
        raise ValueError("Image and mask must have the same size.")
    # First Order Features
    features, labels = pyfeats.fos(img, mask)
    '''
    FOS : First Order Statistics (FOS) 特徵是用於表徵影像中特定灰階的大小區域的分佈，這些區域的大小和灰階值是相關的。
    這些特徵可以用於表徵影像中的細微紋理特徵，例如細胞核的大小和形狀。
    Mean # 平均值 ：測量了像素值的分散程度
    Variance # 變異 ：測量了像素值的分散程度
    Median # 中位數 ：測量了像素值的分散程度
    Mode # 模式 ：測量了像素值的分散程度
    Skewness # 偏度 ：測量了像素值的分散程度
    Kurtosis # 峰度 ：測量了像素值的分散程度
    Energy # 能量 ：測量了像素值的分散程度
    Entropy # 熵 ：測量了像素值的分散程度
    MinimalGrayLevel # 最小灰度級 ：測量了像素值的分散程度
    MaximalGrayLevel # 最大灰度級 ：測量了像素值的分散程度
    CoefficientOfVariation # 變異係數 ：測量了像素值的分散程度
    10Percentile # 10百分位數 ：測量了像素值的分散程度
    25Percentile # 25百分位數 ：測量了像素值的分散程度
    50Percentile # 50百分位數 ：測量了像素值的分散程度
    75Percentile # 75百分位數 ：測量了像素值的分散程度
    90Percentile # 90百分位數 ：測量了像素值的分散程度
    HistogramEntropy # 直方圖熵 ：測量了像素值的分散程度
    
    
    '''
    
    
    feature = dict()
    for key, value in zip(labels, features):
        feature[key + '_' + layer] = value
        
    return feature
        

def SFM(img, mask, layer = 'CC'):
    print('SFM')
    
    f , l = pyfeats.sfm_features(img, mask)
    
    
    '''
    SFM : Statistical Feature Matrix 測量了影像的統計特徵
    Coarseness # 粗糙度 ：測量了像素值的分散程度
    Contrast # 對比度 ：測量了像素值的分散程度
    Periodicity # 周期性 ：測量了像素值的分散程度
    Roughness # 粗糙度 ：測量了像素值的分散程度
    '''
    feature = dict()
    for i in range(len(f)):
        key = l[i].split('_')[1]
       
        feature[key + '_' + layer] = f[i]
    
    return feature        
              
def read_json_file(file):
    import json
    with open(file, 'r') as f:
        data = json.load(f)
    return data                       
    
def remove_small_area(msk_g, min_area = 50):
    # 刪除小面積
    # 連通域的數目 連通域的圖像 連通域的信息 矩形框的左上角坐標 矩形框的寬高 面積
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(msk_g, connectivity=8)
    # 刪除小面積
    msk_rm_small = msk_g.copy()
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] < min_area:
            msk_rm_small[labels == i] = 0
    return msk_rm_small

def process_image(img, msk,all_msk,img_name , patient ,layer, min_area = 50):
    total_area = img.shape[0] * img.shape[1]
    
    
    roi = msk.copy()
    # roi[all_msk == 0] = 0
    
    
    # # 刪除小面積
    msk_rm_small = remove_small_area(roi, min_area)
    print(os.path.join(patient, layer, 'ROI_rm_small'))
    tools.makefolder(os.path.join(patient, layer, 'ROI_rm_small'))
    tools.makefolder(os.path.join(patient, layer, 'ROI_rm_small'))
    cv2.imwrite(os.path.join(patient, layer, 'ROI_rm_small', img_name), msk_rm_small)
    
    if np.sum(msk_rm_small) == 0:
        return 0, 0, 0, 0, 0 , 0, 0 , 0
    fractaldimension =  fractal_dimension(msk_rm_small) 
    # ax = plt.subplot(1, 2, 1)
    # ax.imshow(msk, cmap='gray')
    # ax = plt.subplot(1, 2, 2)
    # ax.imshow(msk_rm_small, cmap='gray')
    # plt.show()
    
    # img_roi_all = img.copy()
    # img_roi_all[all_msk == 0] = 0


    
    
    # # 畫出all_msk 輪廓 
    # contours, hierarchy = cv2.findContours(all_msk, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_TC89_L1)
    # img_roi_vis = cv2.cvtColor(img_roi_all, cv2.COLOR_GRAY2BGR)
    # # draw contours
    # msk_contours = cv2.drawContours(img_roi_vis, contours, -1, (0, 0, 255), 1)
    # # save ROI
    # tools.makefolder(os.path.join(patient, 'ROI_all'))
    # cv2.imwrite(os.path.join(patient, 'ROI_all', img_name), msk_contours)
    
    # # 真正的血管區域
    # label = all_msk.copy()
    # label[msk == 0] = 0
    # # 刪除小面積
    # msk_rm_small = remove_small_area(label, min_area)
    # # save remove small ROI
    # tools.makefolder(os.path.join(patient, 'ROI_rm_small'))
    # cv2.imwrite(os.path.join(patient, 'ROI_rm_small', img_name), msk_rm_small)
        
    # fractaldimension =  fractal_dimension(msk_rm_small)
    # print('fractal_dimension',fractaldimension)
        
    # fig, ax = plt.subplots(1, 5, figsize=(12, 4))
    # ax[0].imshow(img_roi_all, cmap='gray')
    # ax[0].set_title('Original Mask')
    # ax[1].imshow(all_msk, cmap='gray')
    # ax[1].set_title('all_msk')
    # ax[2].imshow(label, cmap='gray')
    # ax[2].set_title('label')
    # ax[3].imshow(msk_rm_small, cmap='gray')
    # ax[3].set_title('msk_rm_small')
    # ax[4].imshow(msk, cmap='gray')
    # ax[4].set_title('msk')
    # plt.show()

     # 找血管的質心點
    M = cv2.moments(msk_rm_small)
    center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
    
    
    # 計算刪除小面積後的面積
    # msk_rm_small > 0 的數量
    
    area = sum(np.where(msk_rm_small > 0, 1, 0).flatten())
    original_msk_area = sum(np.where(all_msk > 0, 1, 0).flatten())
    # 血管面積變化率
    VAPR = area *100 / original_msk_area
    
    # 血管彎曲度
    # Tortuosity =
    # fractal dimension
    # fractal_dimension = 0
    # if VAPR > 1 :
    #     print('VAPR',VAPR)
    #     img_roi_all_vis = cv2.cvtColor(all_msk, cv2.COLOR_GRAY2BGR)
    #     msk_rm_small_vis = cv2.cvtColor(msk_rm_small, cv2.COLOR_GRAY2BGR)
    #     img_roi_all_vis [:,:,0] = 0
    #     img_roi_all_vis [:,:,2] = 0
    #     concate = cv2.addWeighted (img_roi_all_vis, 0.5, msk_rm_small_vis, 0.5, 0)
    #     cv2.imshow('all_msk',img_roi_all_vis)
    #     cv2.imshow('msk_rm_small',msk_rm_small_vis)
    #     Vis = msk_rm_small - all_msk
    #     count_Vis = sum (sum (Vis))
    #     print(count_Vis)
    #     cv2.imshow('Vis',Vis)
    #     cv2.imshow('concate',concate)
    #     cv2.waitKey(0)
    #     cv2.destroyAllWindows()
    
    # 實際內部血管區域
    
    img_roi = img.copy()
    img_roi[msk_rm_small == 0] = 0
    img_roi = remove_small_area(img_roi, min_area)
    # save ROI
    tools.makefolder(os.path.join(patient, layer, 'ROI'))
    cv2.imwrite(os.path.join(patient, layer, 'ROI', img_name), img_roi)

   
    # 血管亮度指數 : Vessel Luminosity Average (VLA)

    VLA = np.sum(img_roi) / area
    
        
   # 骨架化
    ret, binary_img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    binary_img[msk_rm_small == 0] = 0
    binary_img [binary_img > 0 ] = 1
    skeleton = morphology. skeletonize(binary_img)
    skeleton = skeleton.astype(np.uint8)
    # save skeleton
    tools.makefolder(os.path.join(patient, layer, 'skeleton'))
    cv2.imwrite(os.path.join(patient, layer, 'skeleton', img_name), skeleton * 255)
    
    
    binary_img_area = sum(sum(binary_img)) 
    skeleton_area = sum(sum(skeleton))
    # print('binary_img_area',binary_img_area)
    # print('skeleton_area',skeleton_area)
    # VDI : 血管半徑指數
    VDI = binary_img_area / skeleton_area
    
    # 血管
    VLD = sum(np.where(skeleton > 0, 1, 0).flatten()) * 100 / total_area
    
    # VD  : 血管
    VD = sum (np.where(msk_rm_small > 0, 1, 0).flatten()) * 100/ total_area
    
    
    return area, center, VD ,VLD ,VAPR,VLA,VDI ,fractaldimension


                     

def fractal_dimension(image,box_size = 2):
    
    image = image.copy()
    image[image > 0] = 1
    
    # 算法
    def box_count(image, box_size):
        count = 0
        for i in range(0, image.shape[0], box_size):
            for j in range(0, image.shape[1], box_size):
                if np.sum(image[i:i+box_size, j:j+box_size]) > 0:
                    count += 1
        return count
    
    # Calculate the fractal dimension
    counts = []
    box_sizes = []
    while box_size < image.shape[0] and box_size < image.shape[1]:
        box_sizes.append(box_size)
        counts.append(box_count(image, box_size))
        box_size *= 2
        
    # Linear regression
    counts = np.array(counts)
    box_sizes = np.array(box_sizes)
    log_counts = np.log(counts)
    log_box_sizes = np.log(box_sizes)
    
    # Calculate the fractal dimension
    slope, intercept, r_value, p_value, std_err = stats.linregress(log_box_sizes, log_counts)
    return slope

    

def vessel_feature(patient,img_pre_name, img_post_name, layer, min_area = 50):   
    print("img_pre_name",img_pre_name)
    print("img_post_name",img_post_name)
    img_pre = cv2.imread(os.path.join(patient, layer, 'images', img_pre_name), cv2.IMREAD_GRAYSCALE)
        
    img_post = cv2.imread(os.path.join(patient, layer, 'images', img_post_name), cv2.IMREAD_GRAYSCALE)
    msk_pre = cv2.imread(os.path.join(patient, layer, 'masks', img_pre_name), cv2.IMREAD_GRAYSCALE)
    msk_post = cv2.imread(os.path.join(patient, layer, 'masks', img_post_name), cv2.IMREAD_GRAYSCALE)
    
    if img_pre is None or img_post is None or msk_pre is None or msk_post is None:
        print('No image')
        return 0, 0, 0, 0 
    
    # 處理影像
    VAD_pre, center_pre, VBI_pre, VLD_pre, VDI_pre , fractaldimension_pre = process_image(img_pre, msk_pre,msk_pre,img_pre_name, patient, layer, min_area)
    VAD_post, center_post, VBI_post, VLD_post, VDI_post, fractaldimension_post = process_image(img_post, msk_post,msk_pre,img_post_name, patient, layer, min_area)
    
    if VAD_pre == 0 or VAD_post == 0:
        print('No vessel')
        return 0, 0, 0, 0 
    # 血管半徑變化率
    VDR = (VDI_post - VDI_pre) / VDI_pre
    
    # 計算血管長度變化率
    VLR = (VLD_post - VLD_pre) / VLD_pre
    
    # 計算血管亮度變化率
    VBR = (VBI_post - VBI_pre) / VBI_pre
    
    # 計算血管面積變化率
    VAR = (VAD_post - VAD_pre) / VAD_pre
    
    return VDR, VLR, VBR, VAR 


def otsu(image, kernal_size = (3,3)):
        image = cv2.GaussianBlur(image,kernal_size,0)
        try:
            ret3,th3 = cv2.threshold(image,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        except:
            print('NO')
            th3 = np.zeros((304,304))
        return th3

def setFloder(path):
    if not os.path.isdir(path) : os.mkdir(path)



def getMeanStd(feature):
    data_mean = []
    data_std = []
    for i in range(len(feature)):
        data_mean.append(np.mean(feature[i]))
        data_std.append(np.std(feature[i]))
    return data_mean, data_std
    

def main():
    PATH_BASE = '../../Data/'
    data_class = 'PCV'
    data_date = '20240418'
    PATH_BASE  =  PATH_BASE + data_class + '_' + data_date + '/'
    path = PATH_BASE +  '/compare/'
    vessel_analysis = VesselAnalysis(PATH_BASE,data_class + '_' + data_date,path)
    feature_file_name = 'VesselFeature'
    relative_feature_file_name = 'VesselFeature_relative'
    mask_roi = True
    cut = False
    if mask_roi:
        feature_file_name = feature_file_name + '_ROI'
        relative_feature_file_name = relative_feature_file_name + '_ROI'
    
    patient_feature = vessel_analysis.feature_extract(feature_file_name, mask_roi = mask_roi,cut = cut)
    relative_feature = vessel_analysis.relative_feature_extract(feature_file_name,relative_feature_file_name, mask_roi = mask_roi,cut = cut)

    
if __name__ == '__main__':
    main()
