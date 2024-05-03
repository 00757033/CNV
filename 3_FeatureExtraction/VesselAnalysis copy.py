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

from radiomics import featureextractor
from radiomics import firstorder, getTestCase, glcm, glrlm, glszm, imageoperations, shape2D, ngtdm, gldm 
# python -m pip install pyradiomics

class VesselAnalysis():
    def __init__( self, PATH_BASE,disease,compare_path, layers = ['CC', 'OR']):
        self.PATH_BASE = PATH_BASE
        self.compare_path = compare_path
        self.layers = layers
        self.disease = disease
        

    def feature_extract(self):
        patient_feature = dict()
        cout = 0
        for patient in pl.Path(os.path.join(self.compare_path) ).iterdir():
            print('patient',patient)
            patient_feature[patient.name] = dict()
            date = dict()
            for image in pl.Path(os.path.join(patient, 'images') ).iterdir():
                print('image',image)
                if image.name.endswith(".png"):
                    # date_ layer all  exist
                    if os.path.exists(os.path.join(patient, 'masks', image.name)) and os.path.exists(os.path.join(patient, 'images', image.name)):
                        image_date = image.name.split('_')[0]
                        if image_date not in date:
                            date[image_date] = 0
                        date[image_date] += 1
            # date 刪除 value < len(self.layers)
            for key in list(date.keys()):
                if date[key] < len(self.layers):
                    del date[key]
            print('date',date)        
            if len(date) < 2:
                del patient_feature[patient.name]
            else :
                # date 排序
                date = dict(sorted(date.items(), key=lambda item: item[1], reverse=True)) 
                feature = dict()   
                # 將msk 做聯集
                msk_CC = np.zeros((304,304), dtype=np.uint8)
                msk_OR = np.zeros((304,304), dtype=np.uint8)
                for item in range(1):# len(date)
                    print('item',item)
                    CC_post = cv2.imread(os.path.join(patient, 'masks', list(date.keys())[item] + '_CC.png'), cv2.IMREAD_GRAYSCALE)
                    OR_post = cv2.imread(os.path.join(patient, 'masks', list(date.keys())[item] + '_OR.png'), cv2.IMREAD_GRAYSCALE)
                    msk_CC = np.where((msk_CC > 0) | (CC_post > 0), 255, 0)
                    msk_OR = np.where((msk_OR > 0) | (OR_post > 0), 255, 0)
                
                msk_CC[msk_CC > 0] = 255
                msk_CC = msk_CC.astype(np.uint8)
                msk_OR[msk_OR > 0] = 255
                msk_OR = msk_OR.astype(np.uint8)
                
                tools.makefolder(os.path.join(patient, 'concat_masks'))
                cv2.imwrite(os.path.join(patient, 'concat_masks', 'CC.png'), msk_CC)
                cv2.imwrite(os.path.join(patient, 'concat_masks', 'OR.png'), msk_OR)
                
                msk_CC_rm_small = remove_small_area(msk_CC, min_area = 50)
                msk_OR_rm_small = remove_small_area(msk_OR, min_area = 50)
                # save all_msk 
                tools.makefolder(os.path.join(patient, 'concat_masks_rm_small'))
                cv2.imwrite(os.path.join(patient, 'concat_masks_rm_small', 'CC.png'), msk_CC_rm_small)
                cv2.imwrite(os.path.join(patient, 'concat_masks_rm_small', 'OR.png'), msk_OR_rm_small)
                
                # # Dilation 
                # msk_CC_rm_small = cv2.dilate(msk_CC_rm_small, np.ones((4,4), np.uint8), iterations=1)
                # msk_OR_rm_small = cv2.dilate(msk_OR_rm_small, np.ones((4,4), np.uint8), iterations=1)
                
                
                # tools.makefolder(os.path.join(patient, 'all_msk'))
                # cv2.imwrite(os.path.join(patient, 'all_msk', 'CC.png'), msk_CC_rm_small)
                # cv2.imwrite(os.path.join(patient, 'all_msk', 'OR.png'), msk_OR_rm_small)
                
                
                for item in range(len(date)):
                    cout += 1
                    name = list(date.keys())[item] 
                    feature[item] = dict()
                    for layer in self.layers:
                        img_name = list(date.keys())[item] + '_' + layer + '.png'
                        img = cv2.imread(os.path.join(patient, 'images', img_name), cv2.IMREAD_GRAYSCALE)
                        msk = cv2.imread(os.path.join(patient, 'masks', img_name), cv2.IMREAD_GRAYSCALE)
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
                        if layer == 'CC':
                            msk_rm_small = msk_CC_rm_small
                        else :
                            msk_rm_small = msk_OR_rm_small                           
                        area, center,VD ,VLD ,VAPR,VLA,VDI= process_image(img = img, msk = msk,all_msk = msk_rm_small,img_name = img_name,patient = patient)
                        feature[item]['VD_' + layer] = VD
                        feature[item]['VLD_' + layer] = VLD
                        feature[item]['VAPR_' + layer] = VAPR
                        # feature[item]['VLA_' + layer] = VLA
                        feature[item]['VDI_' + layer] = VDI
                        # feature[item]['center_' + layer] = center
                        # feature[item]['Date'] = list(date.keys())[item]
                        print('Texture feature')
                        

                            
                        # texture_features = Texture_features(img, msk_rm_small,layer)    
                        
                        # feature[item] = {**feature[item], **texture_features}
                        len_feature = len(feature[item])
                            
                                
                # print('feature',feature)            
                patient_feature[patient.name] = feature
        print('patient_feature',len(patient_feature))
        print('cout',cout)
        print('len_feature',len_feature)
        tools.makefolder(os.path.join( 'record', self.disease))
        file = './record/' + self.disease + '/' + 'VesselFeature_Morphology.json'
        tools.write_to_json_file(file, patient_feature)
        return patient_feature
            
    def get_feature_extract(self):
        patient_feature = dict()
        cout = 0
        for patient in pl.Path(os.path.join(self.compare_path) ).iterdir():
            patient_feature[patient.name] = dict()
            date = dict()
            for image in pl.Path(os.path.join(patient, 'images') ).iterdir():
                if image.name.endswith(".png"):
                    # date_ layer all  exist
                    if os.path.exists(os.path.join(patient, 'masks', image.name)) and os.path.exists(os.path.join(patient, 'images', image.name)):
                        image_date = image.name.split('_')[0]
                        if image_date not in date:
                            date[image_date] = 0
                        date[image_date] += 1
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
                for item in range(len(date)):
                    if item == 0:
                        continue
                    else:
                        cout += 1
                        post_name = list(date.keys())[item] 
                        feature[post_name] = dict()
                        treatment_count = item
                        for layer in self.layers:
                            img_pre_name = list(date.keys())[item-1] + '_' + layer + '.png'
                            img_post_name = list(date.keys())[item] + '_' + layer + '.png'
                           
                           # Morphological feature 提取血管ROI 影像的形態學特徵
                           # VDR 血管半徑變化率
                            # VLR 血管長度變化率
                            # VBR 血管亮度變化率
                            # VAR 血管面積變化率
                            VDR, VLR, VBR, VAR = vessel_feature(patient,img_pre_name, img_post_name, min_area = 50)
                            feature[post_name][treatment_count] = dict()
                            # print(F'VDR : {VDR}, VLR : {VLR}, VBR : {VBR}, VAR : {VAR}')
                            feature[post_name][treatment_count]['VDR_' + layer] = VDR
                            feature[post_name][treatment_count]['VLR_' + layer] = VLR
                            feature[post_name][treatment_count]['VBR_' + layer] = VBR
                            feature[post_name][treatment_count]['VAR_' + layer] = VAR
                            
                            
                            # Curvelet Transform
                            # Wavelets(patient, img_pre_name, img_post_name, layer, treatment_count)
                            
                            
                            # 形態學特徵會提取血管ROI影像的面積、半徑、周長、輪廓等

                            # morphology_features(pre_name = img_pre_name, post_name = img_post_name, patient = patient)
                            # Texture feature
                        
                            print('Texture feature')
                            # Texture_features_ratio(pre_name = img_pre_name, post_name = img_post_name, patient = patient)
                            
                patient_feature[patient.name] = feature
        print(cout)
        file = './record/' + self.disease + '/' + 'VesselFeature.json'
        tools.write_to_json_file(file, patient_feature)
        return patient_feature

def  HOG(img, msk , bin = 8):
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
    
    features ,labels = pyfeats.hog_features(img_roi, ppc=8, cpb=3)
    # features, labels = pyfeats.tas_features(img_roi)

    

    
    hist, _ = np.histogram(fd, density=True, bins=bin)
    
    # Normalize the features to have a consistent length
    hist = hist.astype("float")
    hist /= (hist.sum() + 1e-7)
    
    
    # print('hist',hist.shape, hist)
    

    # 回傳特徵
    return hist

def TAS(img, msk,layer = 'CC'):
    img_roi = img.copy()
    img_roi[msk == 0] = 0
    feature = dict()
    # Threshold Adjacency Matrix
    features, labels = pyfeats.tas_features(img_roi)
    for i in range(len(features)):
        feature['TAM_' + str(i) + '_' + layer] = features[i]
        
    return feature
    
def GLDS(img, msk,layer = 'CC'):
    img_roi = img.copy()
    img_roi[msk == 0] = 0
    feature = dict()
    # Threshold Adjacency Matrix
    features, labels = pyfeats.glds_features(img_roi,msk)
   
    for i in range(len(features)):
        key = labels[i].split('_')[1]
        feature[labels[i] + '_' + layer] = features[i]
        
    return feature    

# 用局部二值模式（Local Binary Pattern，LBP）提取紋理特徵
def LBP(img, msk,bin =  8):
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
    
    # print('hist',hist.shape, hist)
    
    return hist
 

    
    
    
    
    
    

# Local Gaussian Difference Extrema Pattern
# def LGDEP(img, msk, bin = 64):
#     # Gaussian Difference Extrema Pattern
#     img_roi = img.copy()
#     img_roi[msk == 0] = 0
    
#      # Apply Gaussian blur to the image
#     blurred = cv2.GaussianBlur(img_roi, (5, 5), 0)
#     # Apply LGDEP
#     lgp = cv2.Laplacian(blurred, cv2.CV_64F)
    
#     # 視覺化
#     plt.figure(figsize=(12, 4))
#     plt.subplot(1, 2, 1)
#     plt.imshow(img, cmap='gray')
#     plt.title('Original Image')
    
#     plt.subplot(1, 2, 2)
#     plt.imshow(lgp, cmap='gray')
#     plt.title('LGDEP Image')
    
#     plt.show()
    
#     # 計算LGDEP紋理圖像的直方圖
#     hist, _ = np.histogram(lgp.ravel(), density=True, bins=bin)
    
#     # Normalize the features to have a consistent length
#     hist = hist.astype("float")
#     hist /= (hist.sum() + 1e-7)
    
#     # print('hist',hist.shape, hist)
    
#     return hist

    
def Wavelets(patient, img_pre_name, img_post_name, layer, treatment_count):
    img_pre = cv2.imread(os.path.join(patient, 'images', img_pre_name), cv2.IMREAD_GRAYSCALE)
    img_post = cv2.imread(os.path.join(patient, 'images', img_post_name), cv2.IMREAD_GRAYSCALE)
    msk_pre = cv2.imread(os.path.join(patient, 'masks', img_pre_name), cv2.IMREAD_GRAYSCALE)
    msk_post = cv2.imread(os.path.join(patient, 'masks', img_post_name), cv2.IMREAD_GRAYSCALE)
    
    if img_pre is None or img_post is None or msk_pre is None or msk_post is None:
        print('No image')
        return 0, 0, 0, 0, 0
    
    HOGL, hog_image = HOG(img_pre)
    
    msk = msk_pre + msk_post
    msk[msk > 0] = 255
    msk = msk.astype(np.uint8)
    msk_rm_small = remove_small_area(msk, min_area = 50)
    
    img = img_pre.copy()
    img[msk_rm_small == 0] = 0
    
    
    # 使用 Daubechies 小波變換
    coeffs = pywt.dwt2(img, 'db1')
    
   # Step 4: Threshold the detail coefficients to remove noise
    threshold = 0.1  # Adjust this threshold value based on your requirements
    coeffs = tuple(map(lambda x: pywt.threshold(x, threshold, mode='soft'), coeffs))

    # Step 5: Reconstruct the denoised image
    denoised_img = pywt.idwt2(coeffs, 'db1')

    # 可视化
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.imshow(img, cmap='gray')
    plt.title('Original Image')

    plt.subplot(1, 2, 2)
    plt.imshow(denoised_img, cmap='gray')
    plt.title('Denoised Image')

    plt.show()

    # # coeffs 包含了逼近系数（LL子图）和细节系数（LH、HL、HH子图）
    LL, (LH, HL, HH) = coeffs

    # 可视化
    plt.figure(figsize=(12, 4))

    plt.subplot(2, 2, 1)
    plt.imshow(LL, cmap='gray')
    plt.title('Approximation (LL)')

    plt.subplot(2, 2, 2)
    plt.imshow(LH, cmap='gray')
    plt.title('Horizontal Detail (LH)')

    plt.subplot(2, 2, 3)
    plt.imshow(HL, cmap='gray')
    plt.title('Vertical Detail (HL)')

    plt.subplot(2, 2, 4)
    plt.imshow(HH, cmap='gray')
    plt.title('Diagonal Detail (HH)')

    plt.show()

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
    
    # feature 整合
    feature= {**feature, **GLCM_feature}
    
    # print('GLRLM')
    GLRLM_feature = GLRLM(img , msk ,layer)
    
    feature = {**feature, **GLRLM_feature}

    
    # print('GLSZM')
    GLSZM_feature = GLSZM(img , msk,layer)
    feature = {**feature, **GLSZM_feature}
    
    # print('GLDM')
    GLDM_feature = GLDM(img , msk,layer)
    feature = {**feature, **GLDM_feature}
    
    # print('NGTDM')
    NGTDM_feature = NGTDM(img , msk,layer)
    feature = {**feature, **NGTDM_feature}
    
    # # # # print('Statistical Feature Matrix')
    # #  SFM 特徵如下：1) 粗糙度，2) 對比度，3) 週期性，4) 粗糙度。
    SFM_feature = SFM(img , msk)
    feature = {**feature, **SFM_feature}
    
    # # FDTA
    # # print('FDTA')
    FDTA_feature = FDTA(img , msk,layer)
    feature = {**feature, **FDTA_feature}
    
    # # HOG
    # print('HOG')
    fd  = HOG(img, msk )
    feature['HOG' + '_' + layer] = fd
    
    
    GLDS_feature = GLDS(img , msk,layer) # 'GLDS_Homogeneity', 'GLDS_Contrast', 'GLDS_ASM', 'GLDS_Entopy', 'GLDS_Mean'
    feature = {**feature, **GLDS_feature} 
    
    #     # # print('LBP')
    # lbp= LBP(img, msk )
    # feature['LBP' + '_' + layer] = lbp
    

    
    # # # print('FirstOrder')
    # firstOrder_feature = firstOrder(img , msk,layer)
    # feature = {**feature, **firstOrder_feature}

    
    
    # # print('DWT')
    EHOG_feature = DWT(img , msk , layer)
    feature = {**feature, **EHOG_feature}
    
    
    
    
    # # print('LGDEP')
    
    
        
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

# def LTEM(img, msk,layer = 'CC'):
#     img_roi = img.copy()
#     img_roi[msk == 0] = 0
#     feature = dict()
#     # Threshold Adjacency Matrix
#     features, labels = pyfeats.lte_measures(img,msk,l=3)
#     print(labels)
#     for i in range(len(features)):
#         print('LTEM_' + str(i) + '_' + layer,features[i])
#         feature['LTEM_' + str(i) + '_' + layer] = features[i]
        
#     return feature

def HuMoments(img, msk,layer = 'CC'):
    img_roi = img.copy()
    img_roi[msk == 0] = 0
    feature = dict()
    # Hu Moments
    features, labels = pyfeats.hu_moments(img_roi)
    for i in range(len(features)):
        print('Hu_' + str(i) + '_' + layer,features[i])
        feature['Hu_' + str(i) + '_' + layer] = features[i]
        
    return feature

def Zernikes(img, msk,layer = 'CC'):
    img_roi = img.copy()
    img_roi[msk == 0] = 0
    feature = dict()
    # Zernikes
    features, labels = pyfeats.zernikes(img_roi)
    for i in range(len(features)):
        feature['Zernikes_' + str(i) + '_' + layer] = features[i]
        
    return feature


def FDTA(img, msk,layer = 'CC'):
    img_roi = img.copy()
    img_roi[msk == 0] = 0
    feature = dict()
    # FDTA
    features, labels = pyfeats.fdta(img_roi,msk)
    for i in range(len(features)):
        feature['FDTA_' + str(i) + '_' + layer] = features[i]
        
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
    std_LH = np.std(non_zero_abs_LH)
    std_HL = np.std(non_zero_abs_HL)
    
    
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
    # 特徵提取 - 小波變換
    img_roi = img.copy()
    img_roi[msk == 0] = 0

    feature = dict()

    features, labels = pyfeats.dwt_features(img, msk, wavelet='db1')
    for i in range(len(features)):
        feature[labels[i] + '_' + layer] = features[i]
    # LH 水平細節


   
    return feature

def GaborTransform(img, msk, layer = 'CC'):
    # 特徵提取 - Gabor變換
    img_roi = img.copy()
    img_roi[msk == 0] = 0

    feature = dict()
    # 使用 Gabor Transform 小波變換 僅 對msk內的影像進行小波變換
    features, labels = pyfeats.dwt_features(img, msk, wavelet='db1')
    for i in range(len(features)):
        feature[labels[i] + '_' + layer] = features[i]
    # LH 水平細節

    return feature

def AMFM(img, msk, layer = 'CC'):
    # 特徵提取 - AMFM變換
    img_roi = img.copy()
    img_roi[msk == 0] = 0

    feature = dict()
    # 使用 Gabor Transform 小波變換 僅 對msk內的影像進行小波變換
    features, labels = pyfeats.amfm_features(img_roi, bins = 32)
    for i in range(len(features)):
        feature[labels[i] + '_' + layer] = features[i]
    # LH 水平細節

    return feature

def GLCM(img, msk,layer = 'CC'):
    img_roi = img.copy()
    img_roi[msk == 0] = 0
    
    features = dict()
    msk [ msk > 0 ] = 1
    # Convert NumPy array to SimpleITK image
    image_sitk = sitk.GetImageFromArray(img_roi)
    mask_sitk = sitk.GetImageFromArray(msk)
    
    glcm_features = glcm.RadiomicsGLCM(image_sitk, mask_sitk)
    glcm_features.execute()
    for (key, val) in six.iteritems(glcm_features.featureValues):
        features[key + '_' + layer] = val
        
    
    glcms = graycomatrix (img, [1], [0, np.pi/4, np.pi/2, 3*np.pi/4], levels=256, symmetric=True, normed=True)
    Homogeneity = graycoprops(glcms, 'homogeneity')
    Dissimilarity = graycoprops(glcms, 'dissimilarity')
    Contrast = graycoprops(glcms, 'contrast')
    Energy = graycoprops(glcms, 'energy')
    Correlation = graycoprops(glcms, 'correlation')
    ASM = graycoprops(glcms, 'ASM')
        
    features['Homogeneity' + '_' + layer] = sum(Homogeneity[0])/len(Homogeneity[0])
    features['Dissimilarity' + '_' + layer] = sum(Dissimilarity[0])/len(Dissimilarity[0])
    
    # for i in range(len(distance)):
    #         glcm = graycomatrix(img, [distance[i]], [0, np.pi/4, np.pi/2, 3*np.pi/4], levels=256, symmetric=True, density=True)
    #         contrast = graycoprops(glcm, 'contrast')
    #         dissimilarity = graycoprops(glcm, 'dissimilarity')
    #         homogeneity = graycoprops(glcm, 'homogeneity')
    #         energy = graycoprops(glcm, 'energy')
    #         correlation = graycoprops(glcm, 'correlation')
    #         ASM = graycoprops(glcm, 'ASM')
    #         features[i] = dict()
    #         features[i]['contrast'] = contrast
    #         features[i]['dissimilarity'] = dissimilarity
    #         features[i]['homogeneity'] = homogeneity
    #         features[i]['energy'] = energy
    #         features[i]['correlation'] = correlation
    #         features[i]['ASM'] = ASM
            
               
    return features
                          
def GLCM3D(img):
    # 3 維矩陣

    features_mean, features_range, labels_mean, labels_range = pyfeats.glcm_features(img, ignore_zeros=True)
    feature = dict()
    for i in range(len(features_mean)):
        feature[labels_mean[i]] = features_mean[i]
    for i in range(len(features_range)):
        feature[labels_range[i]] = features_range[i]
    print(feature)
    return feature
    
def GLRLM(img, mask, layer = 'CC'):
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
    feature = dict()
    
    for (key, val) in six.iteritems(glrlmFeatures.featureValues):
        feature[key + '_' + layer] = val
        
    return feature
        
def GLSZM(img, mask, layer = 'CC'):
    mask [ mask > 0 ] = 1
    # Convert NumPy array to SimpleITK image
    image_sitk = sitk.GetImageFromArray(img)
    mask_sitk = sitk.GetImageFromArray(mask)

    glszmFeatures = glszm.RadiomicsGLSZM(image_sitk, mask_sitk )
    glszmFeatures.enableAllFeatures()
    glszmFeatures.execute()
    
    feature = dict()
    for (key, val) in six.iteritems(glszmFeatures.featureValues):
        feature[key + '_' + layer] = val
    
    return feature

def GLDM(img, mask, layer = 'CC'):
    mask [ mask > 0 ] = 1
    # Convert NumPy array to SimpleITK image
    image_sitk = sitk.GetImageFromArray(img)
    mask_sitk = sitk.GetImageFromArray(mask)

    gldmFeatures = gldm.RadiomicsGLDM(image_sitk, mask_sitk )
    gldmFeatures.enableAllFeatures()
    gldmFeatures.execute()
    feature = dict()
    for (key, val) in six.iteritems(gldmFeatures.featureValues):
        feature[key + '_' + layer] = val
    
    return feature
 
def NGTDM(img, mask, layer = 'CC'):
    mask [ mask > 0 ] = 1
    # Convert NumPy array to SimpleITK image
    image_sitk = sitk.GetImageFromArray(img)
    mask_sitk = sitk.GetImageFromArray(mask)

    ngtdmFeatures = ngtdm.RadiomicsNGTDM(image_sitk, mask_sitk )
    ngtdmFeatures.enableAllFeatures()
    ngtdmFeatures.execute()
    feature = dict()
    for (key, val) in six.iteritems(ngtdmFeatures.featureValues):
        feature[key + '_' + layer] = val
    
    
    return feature         

# First Order Features
def firstOrder(img, mask, layer = 'CC'):
    mask [ mask > 0 ] = 1
    # Convert NumPy array to SimpleITK image
    image_sitk = sitk.GetImageFromArray(img)
    mask_sitk = sitk.GetImageFromArray(mask)

    firstorderFeatures = firstorder.RadiomicsFirstOrder(image_sitk, mask_sitk )
    firstorderFeatures.enableAllFeatures()
    firstorderFeatures.execute()
    feature = dict()
    for (key, val) in six.iteritems(firstorderFeatures.featureValues):
        feature[key + '_' + layer] = val
    
    return feature

def SFM(img, mask, layer = 'CC'):
    feature = dict()
    f , l = pyfeats.sfm_features(img, mask)
    
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
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(msk_g)
    # 刪除小面積
    msk_rm_small = msk_g.copy()
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] < min_area:
            msk_rm_small[labels == i] = 0
    return msk_rm_small

def process_image(img, msk,all_msk,img_name , patient , min_area = 50):
    total_area = img.shape[0] * img.shape[1]
  
    img_roi_all = img.copy()
    img_roi_all[all_msk == 0] = 0


    
    
    # 畫出all_msk 輪廓 
    contours, hierarchy = cv2.findContours(all_msk, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_TC89_L1)
    img_roi_vis = cv2.cvtColor(img_roi_all, cv2.COLOR_GRAY2BGR)
    # draw contours
    msk_contours = cv2.drawContours(img_roi_vis, contours, -1, (0, 0, 255), 1)
    # save ROI
    tools.makefolder(os.path.join(patient, 'ROI_all'))
    cv2.imwrite(os.path.join(patient, 'ROI_all', img_name), msk_contours)
    
    # 真正的血管區域
    label = all_msk.copy()
    label[msk == 0] = 0
    # 刪除小面積
    msk_rm_small = remove_small_area(label, min_area)
    # save remove small ROI
    tools.makefolder(os.path.join(patient, 'ROI_rm_small'))
    cv2.imwrite(os.path.join(patient, 'ROI_rm_small', img_name), msk_rm_small)
        
    # fig, ax = plt.subplots(1, 4, figsize=(12, 4))
    # ax[0].imshow(img_roi_all, cmap='gray')
    # ax[0].set_title('Original Mask')
    # ax[1].imshow(all_msk, cmap='gray')
    # ax[1].set_title('all_msk')
    # ax[2].imshow(label, cmap='gray')
    # ax[2].set_title('label')
    # ax[3].imshow(msk_rm_small, cmap='gray')
    # ax[3].set_title('msk_rm_small')
    # plt.show()
    
    if np.sum(msk_rm_small) == 0:
        return 0, 0, 0, 0, 0 , 0, 0
     # 找血管的質心點
    M = cv2.moments(msk_rm_small)
    center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
    
    
    # 計算刪除小面積後的面積
    # msk_rm_small > 0 的數量
    
    area = sum(np.where(msk_rm_small > 0, 1, 0).flatten())
    original_msk_area = sum(np.where(all_msk > 0, 1, 0).flatten())
    # 血管面積變化率
    VAPR = area *100 / original_msk_area
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
    tools.makefolder(os.path.join(patient, 'ROI'))
    cv2.imwrite(os.path.join(patient, 'ROI', img_name), img_roi)

   
    # 血管亮度指數 : Vessel Luminosity Average (VLA)

    VLA = np.sum(img_roi) / area
    
        
   # 骨架化
    ret, binary_img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    binary_img[msk_rm_small == 0] = 0
    binary_img [binary_img > 0 ] = 1
    skeleton = morphology. skeletonize(binary_img)
    skeleton = skeleton.astype(np.uint8)
    # save skeleton
    tools.makefolder(os.path.join(patient, 'skeleton'))
    cv2.imwrite(os.path.join(patient, 'skeleton', img_name), skeleton * 255)
    
    
    binary_img_area = sum(sum(binary_img)) 
    skeleton_area = sum(sum(skeleton))
    print('binary_img_area',binary_img_area)
    print('skeleton_area',skeleton_area)
    # VDI : 血管半徑指數
    VDI = binary_img_area / skeleton_area
    
    # 血管
    VLD = sum(np.where(skeleton > 0, 1, 0).flatten()) * 100 / total_area
    
    # VD  : 血管
    VD = sum (np.where(msk_rm_small > 0, 1, 0).flatten()) * 100/ total_area
    
    
    return area, center, VD ,VLD ,VAPR,VLA,VDI


def vessel_feature(patient,img_pre_name, img_post_name, min_area = 50):   
    print("img_pre_name",img_pre_name)
    print("img_post_name",img_post_name)
    img_pre = cv2.imread(os.path.join(patient, 'images', img_pre_name), cv2.IMREAD_GRAYSCALE)
        
    img_post = cv2.imread(os.path.join(patient, 'images', img_post_name), cv2.IMREAD_GRAYSCALE)
    msk_pre = cv2.imread(os.path.join(patient, 'masks', img_pre_name), cv2.IMREAD_GRAYSCALE)
    msk_post = cv2.imread(os.path.join(patient, 'masks', img_post_name), cv2.IMREAD_GRAYSCALE)
    
    if img_pre is None or img_post is None or msk_pre is None or msk_post is None:
        print('No image')
        return 0, 0, 0, 0 
    
    # 處理影像
    VAD_pre, center_pre, VBI_pre, VLD_pre, VDI_pre = process_image(img_pre, msk_pre, min_area)
    VAD_post, center_post, VBI_post, VLD_post, VDI_post = process_image(img_post, msk_post, min_area)
    
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
    data_date = '20240311'
    PATH_BASE  =  PATH_BASE + data_class + '_' + data_date + '/'
    path = PATH_BASE +  '/compare/'
    vessel_analysis = VesselAnalysis(PATH_BASE,data_class + '_' + data_date,path)
    vessel_analysis.feature_extract()
    # 計算 P-value

    
if __name__ == '__main__':
    main()
