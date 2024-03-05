import cv2
import os
import numpy as np
import pathlib as pl
import shutil
import pandas as pd
import csv
from sklearn.cluster import DBSCAN
from matplotlib import pyplot as plt
from skimage.metrics import mean_squared_error 
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio
import math
# Entropy
from sklearn.decomposition import PCA
import tools.tools as tools

# pip install opencv-contrib-python
def mean_squared_error_ignore_zeros(img1, img2,img3):
    # 對於兩張影像 計算每個pixel 的差異
    # 找出兩影像中相應像素都為零的位置
    msk1 = img1 == 0
    msk2 = img2 == 0
    msk3 = img3 == 0
    both_zeros_mask = np.logical_and(np.logical_and(msk1, msk2), msk3)
    # 提取非零区域
    # img1_non_zero = img1.copy()
    # img1_non_zero[both_zeros_mask] = 0
    # img2_non_zero = img2.copy()
    # img2_non_zero[both_zeros_mask] = 0
    # img3_non_zero = img3.copy()
    # img3_non_zero[both_zeros_mask] = 0
    
    size = 304 * 304 - np.sum(both_zeros_mask)
    
    # diff = img1_non_zero - img2_non_zero
    # mse = np.sum(diff ** 2) / size
    # print('mse',mse)
    
    
    diff = img1[~both_zeros_mask] - img2[~both_zeros_mask]
    mse =  np.sum(diff ** 2) / size
    # print('mse',mse)

    
    return mse

def psnr_ignore_zeros(img1, img2,img3):
    msk1 = img1 == 0
    msk2 = img2 == 0
    msk3 = img3 == 0
    
    both_zeros_mask = np.logical_and(np.logical_and(msk1, msk2), msk3)
    
    img1_non_zero = img1.copy()
    img1_non_zero[both_zeros_mask] = 0
    img2_non_zero = img2.copy()
    img2_non_zero[both_zeros_mask] = 0
    img3_non_zero = img3.copy()
    img3_non_zero[both_zeros_mask] = 0
    

    
    # # 計算 PSNR
    mse = mean_squared_error_ignore_zeros(img1, img2,img3)
    psnr_value = 10 * np.log10(255 ** 2 / mse)
    # print('psnr',psnr_value)
    
    return psnr_value

def ssim_ignore_zeros(img1, img2,img3):
    
    # 找出兩影像中相應像素都為零的位置
    msk1 = img1 == 0
    msk2 = img2 == 0
    msk3 = img3 == 0
    both_zeros_mask = np.logical_and(np.logical_and(msk1, msk2), msk3)
    # 提取非零区域
    img1_non_zero = img1.copy()
    img1_non_zero[both_zeros_mask] = 0
    img2_non_zero = img2.copy()
    img2_non_zero[both_zeros_mask] = 0
    

    ssim_value, _ = ssim(img1_non_zero, img2_non_zero, data_range=254, full=True, gaussian_weights=True, use_sample_covariance=False)
    # print('ssim_value',ssim_value)
    # # SSIM
    # ssim_index, _ = ssim(img1_non_zero, img2_non_zero,full=True)
    # print('ssim_index',ssim_index)
    return ssim_value
    
 
def ncc_ignore_zeros(img1, img2,img3):
    both_zeros_mask  = (img1 == 0) & (img2 == 0) & (img3 == 0)
    # 提取非零区域
    img1_non_zero = img1[~both_zeros_mask]
    img2_non_zero = img2[~both_zeros_mask]
    # Calculate the mean of the two images
    mean_image1 = np.mean(img1_non_zero)
    mean_image2 = np.mean(img2_non_zero)

    # Calculate the NCC
    ncc = np.sum((img1_non_zero - mean_image1) * (img2_non_zero - mean_image2)) / (np.sqrt(np.sum((img1_non_zero - mean_image1) ** 2)) * np.sqrt(np.sum((img2_non_zero - mean_image2) ** 2)))
    return ncc
   
    
class template_matcher():

        
    def __init__(self,image_path,label_path,output_image_path,output_label_path):
        self.image_path = image_path
        self.label_path = label_path
        self.output_image_path = output_image_path
        self.output_label_path = output_label_path
        self.layer_list = ['1','2', '3', '4', '1_OCT', '2_OCT', '3_OCT', '4_OCT']
        self.label_list = ['CC']
        self.image_size = (304, 304)
        
        setFolder(self.output_image_path)
        setFolder(self.output_label_path)

        self.methods = [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED, cv2.TM_CCORR , cv2.TM_CCORR_NORMED, cv2.TM_CCOEFF, cv2.TM_CCOEFF_NORMED]
        self.method_name = ['TM_SQDIFF', 'TM_SQDIFF_NORMED', 'TM_CCORR' , 'TM_CCORR_NORMED', 'TM_CCOEFF', 'TM_CCOEFF_NORMED']
        # for method in self.method_name:
        #     for layer in self.layer_list:
        #         setFolder(os.path.join(self.output_image_path,'crop_'+ method, layer))
        #     for label in self.label_list:
        #         setFolder(os.path.join(self.output_label_path, 'crop_'+method, label))


    #Template Match Method:找到template在img中最高的相關係數的數值(r)和位置(top_left)
    def getPoints(self,img, template, method=cv2.TM_CCOEFF_NORMED):
        result = cv2.matchTemplate(img, template, method) # 回傳的是相關係數矩陣
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result) # 找到最大值和最小值的位置 也就是左上角的位置 以及右下角的位置
        if self.method_name[method] in ['TM_SQDIFF', 'TM_SQDIFF_NORMED']:
            top_left = min_loc # 要移動的位置
            r = min_val # r=最低的相關係數
        else:
            top_left = max_loc
            r = max_val # r=最高的相關係數
        return top_left, r #top_left:回傳對位左上角的位置, r=最高的相關係數


    def get_element(self,image, template, offset_x=0, offset_y=0, method=cv2.TM_CCOEFF_NORMED):
        elements, r = self.getPoints(image, template, method)
        elements = (elements[0] - offset_x, elements[1] - offset_y)
        return elements, r

    def preprocess(self,patient_id, eye, pre_treatment, post_treatment, image):
        image2 = image.copy()
        image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)
        imageequal = cv2.GaussianBlur(image, (3, 3), 0)
        imageequal = cv2.equalizeHist(imageequal)
        imageequal = cv2.GaussianBlur(imageequal, (5, 5), 0)
        imageequal = cv2.equalizeHist(imageequal)
        imageequal = cv2.GaussianBlur(imageequal, (3, 3), 0)
        imageequal = cv2.equalizeHist(imageequal)
        # # top hat
        # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 40))
        # tophat = cv2.morphologyEx(imageequal, cv2.MORPH_TOPHAT, kernel)
        # # black hat
        # blackhat = cv2.morphologyEx(image, cv2.MORPH_BLACKHAT, kernel)
        # # add and subtract between morphological gradient and image
        # image = tophat
        # image = cv2.add(image, tophat)
        image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)
        # image = cv2.GaussianBlur(image, (3, 3), 0)
        # image = cv2.equalizeHist(image)
        # image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)
        # image = cv2.GaussianBlur(image, (5, 5), 0)
        # image = cv2.equalizeHist(image)
        # fig, ax = plt.subplots(2, 5, figsize=(20, 20))
        # ax[0, 0].imshow(image2, cmap='gray')
        # ax[0, 0].set_title('Original Image')
        # ax[0, 1].imshow(tophat, cmap='gray')
        # ax[0, 1].set_title('Top Hat')
        # ax[0, 2].imshow(blackhat, cmap='gray')
        # ax[0, 2].set_title('blackhat')
        # ax[0, 3].imshow(image, cmap='gray')
        # ax[0, 3].set_title('Add and Subtract')
        # ax[0, 4].imshow(imageequal ,cmap='gray')
        # ax[0, 4].set_title('EqualizeHist')
        
        # ax[1, 0].hist(image2.ravel(), 256, [0, 256])
        # ax[1, 0].set_title('Original Image')
        # ax[1, 1].hist(tophat.ravel(), 256, [0, 256])
        # ax[1, 1].set_title('Top Hat')
        # ax[1, 2].hist(blackhat.ravel(), 256, [0, 256])
        # ax[1, 2].set_title('Black Hat')
        # ax[1, 3].hist(image.ravel(), 256, [0, 256])
        # ax[1, 3].set_title('Add and Subtract')
        # ax[1, 4].hist(imageequal.ravel(), 256, [0, 256])
        # ax[1, 4].set_title('EqualizeHist')
        # plt.show()


        
        # image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)
        # image = cv2.equalizeHist(image)
        # image = cv2.equalizeHist(image)
        # image = cv2.GaussianBlur(image, (3, 3), 0)
        # image = cv2.equalizeHist(image)
        # image = cv2.GaussianBlur(image, (3, 3), 0)
        # image = cv2.equalizeHist(image)
        # image = cv2.GaussianBlur(image, (3, 3), 0)
        # image = cv2.equalizeHist(image)
        # image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)

        return image
    
    def preprocess2(self,patient_id, eye, pre_treatment, post_treatment, image):
        
        image2 = image.copy()
        image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)
        imageequal = cv2.GaussianBlur(image, (3, 3), 0)
        imageequal = cv2.equalizeHist(imageequal)
        # imageequal = cv2.GaussianBlur(imageequal, (3, 3), 0)
        # imageequal = cv2.equalizeHist(imageequal)
        # imageequal = cv2.GaussianBlur(imageequal, (5, 5), 0)
        # imageequal = cv2.equalizeHist(imageequal)
        # top hat
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 40))
        tophat = cv2.morphologyEx(imageequal, cv2.MORPH_TOPHAT, kernel)
        # black hat
        blackhat = cv2.morphologyEx(image, cv2.MORPH_BLACKHAT, kernel)
        # add and subtract between morphological gradient and image
        image = tophat
        image = cv2.add(image, tophat)
        image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)
        # image = cv2.GaussianBlur(image, (3, 3), 0)
        # image = cv2.equalizeHist(image)
        # image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)
        # image = cv2.GaussianBlur(image, (5, 5), 0)
        # image = cv2.equalizeHist(image)
        # fig, ax = plt.subplots(2, 5, figsize=(20, 20))
        # ax[0, 0].imshow(image2, cmap='gray')
        # ax[0, 0].set_title('Original Image')
        # ax[0, 1].imshow(tophat, cmap='gray')
        # ax[0, 1].set_title('Top Hat')
        # ax[0, 2].imshow(blackhat, cmap='gray')
        # ax[0, 2].set_title('blackhat')
        # ax[0, 3].imshow(image, cmap='gray')
        # ax[0, 3].set_title('Add and Subtract')
        # ax[0, 4].imshow(imageequal ,cmap='gray')
        # ax[0, 4].set_title('EqualizeHist')
        
        # ax[1, 0].hist(image2.ravel(), 256, [0, 256])
        # ax[1, 0].set_title('Original Image')
        # ax[1, 1].hist(tophat.ravel(), 256, [0, 256])
        # ax[1, 1].set_title('Top Hat')
        # ax[1, 2].hist(blackhat.ravel(), 256, [0, 256])
        # ax[1, 2].set_title('Black Hat')
        # ax[1, 3].hist(image.ravel(), 256, [0, 256])
        # ax[1, 3].set_title('Add and Subtract')
        # ax[1, 4].hist(imageequal.ravel(), 256, [0, 256])
        # ax[1, 4].set_title('EqualizeHist')
        # plt.show()


        
        # image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)
        # image = cv2.equalizeHist(image)
        # image = cv2.equalizeHist(image)
        # image = cv2.GaussianBlur(image, (3, 3), 0)
        # image = cv2.equalizeHist(image)
        # image = cv2.GaussianBlur(image, (3, 3), 0)
        # image = cv2.equalizeHist(image)
        # image = cv2.GaussianBlur(image, (3, 3), 0)
        # image = cv2.equalizeHist(image)
        # image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)

        return image


    def find_center(self,patient_id, eye, pre_treatment, post_treatment,image):
        image_1 = image.copy()
        image_1 = self.preprocess2(patient_id, eye, pre_treatment, post_treatment,image_1)
        # image_1 = cv2.GaussianBlur(image_1, (5, 5), 0)
        # rst,image_1 = cv2.threshold(image_1, 0, 255,  cv2.THRESH_BINARY  + cv2.THRESH_OTSU)
        
        rst,image_1 = cv2.threshold(image_1, 64, 255,  cv2.THRESH_BINARY)
        # image_1 = cv2.adaptiveThreshold(image_1, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
        image_1 = cv2.bitwise_not(image_1)

        image_1[ image_1 > 0] = 1
        # image_2[ image_2 > 0] = 1
        # print('image_2',np.count_nonzero(image_2))
        # image_add = np.multiply(image_1, image_2) * 255
        # print(image_add)
       
        # cv2.imshow('img',image_add)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()


        # 找到影像中 最白的最大面積的圓
        _, labels, stats, centroids = cv2.connectedComponentsWithStats(image_1, connectivity=4)

        # 找到最大的區域（排除背景區域）
        label_area = dict()
        for i in range(1, labels.max() + 1):
            label_area[i] = stats[i][4]

        # 依照面積從大到小排序
        label_area = sorted(label_area.items(), key=lambda kv: kv[1], reverse=True)

        # 找到最大的面積
        max_label = label_area[0][0]

        # 上色
        draw_image = image.copy()
        draw_image = cv2.cvtColor(draw_image, cv2.COLOR_GRAY2BGR)
        draw_image[labels == max_label] = (0, 0, 255)

        # 找到最大的面積的中心點
        center = centroids[max_label]
        center = (int(center[0]), int(center[1]))
        # 與中心最遠的距離
        # max_radius = max(stats[max_label][2], stats[max_label][3]) // 2
        max_radius = 0
        # for i in 最大面積的所有點:
        for i in  np.argwhere(labels == max_label):
            # 計算與中心的距離
            radius = math.sqrt((i[0] - center[1]) ** 2 + (i[1] - center[0]) ** 2)
            # 找到最遠的距離
            if radius > max_radius:
                max_radius = radius

        # 無條件進位
        max_radius = int(math.ceil(max_radius))

        t = 50
        if int(center[0] - max_radius) < t :
            t = int(center[0] - max_radius)
        if int(center[1] - max_radius) < t :
            t = int(center[1] - max_radius)
        if int(center[0] + max_radius) > len(image[0]) - t :
            t = len(image[0]) - int(center[0] + max_radius)
        if int(center[1] + max_radius) > len(image) - t :
            t = len(image) - int(center[1] + max_radius)        
        # # 畫出黃斑中心
        # # draw the circle
        cv2.rectangle(draw_image, (int(center[0] - max_radius), int(center[1] - max_radius)), (int(center[0] + max_radius), int(center[1] + max_radius)), (255, 0, 255), 3)
        cv2.circle(draw_image, center, 5, (0, 255, 255), -1)
        
        cv2.rectangle(draw_image, (int(center[0] - max_radius-t), int(center[1] - max_radius-t)), (int(center[0] + max_radius+ t), int(center[1] + max_radius+ t)), (255, 255, 0), 3)

        # cv2.imshow('img',draw_image)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        
        setFolder(self.output_image_path + 'crop/')
        cv2.imwrite(self.output_image_path + 'crop/' + patient_id + '_' + eye + '_' + post_treatment + '.png', draw_image)



        crop_img = image[int(center[1] - max_radius)-t:int(center[1] + max_radius)+t, int(center[0] - max_radius)-t:int(center[0] + max_radius)+t]
        # show the image wait 3 seconds and destroy


        return  crop_img , center , max_radius + t
    #將術後的影像切成4個角落跟中央，並分別跟術前影像做getPoints，找到5個template中R最高的位置，並回傳要位移的x跟y
    def pointMatch(self,patient_id, eye, pre_treatment, post_treatment,image, template, method = cv2.TM_CCOEFF_NORMED):
        crop_img ,center,radius = self.find_center(patient_id, eye, pre_treatment, post_treatment,template)
        elements_c, r_c = self.get_element(image, crop_img, center[0] - radius, center[1] - radius, method)
        
        # 選擇最高的相關係數的位置
        e = [elements_c]
        r = [r_c]
        Relation = 0
        relation_min = 1000000000000000000000000000
        shift_x = None
        shift_y = None
        for i in range(0, len(r)):

            if self.method_name[method] in ['TM_SQDIFF', 'TM_SQDIFF_NORMED']:

                if r[i] < relation_min:
                    relation_min = r[i]
                    shift_x = e[i][0] 
                    shift_y = e[i][1]
            elif  self.method_name[method] in ['TM_CCORR']:  
                if r[i] > Relation:
                    Relation = r[i]
                    shift_x = e[i][0]
                    shift_y = e[i][1]
            else:
                if r[i] > Relation:
                    Relation = r[i]
                    shift_x = e[i][0]
                    shift_y = e[i][1]                
        if shift_x != None : 
            return shift_x, shift_y
        else:
            print('No Match')
            return 0,0

    def evaluates(self, pre_treatment_img, post_treatment_img,matching_img):
        cmp_pre_treatment_img = cv2.imread(pre_treatment_img , cv2.IMREAD_GRAYSCALE)
        cmp_post_treatment_img = cv2.imread(post_treatment_img, cv2.IMREAD_GRAYSCALE)
        cmp_matching_img = cv2.imread(matching_img, cv2.IMREAD_GRAYSCALE)
        

        
                          
        
        if cmp_pre_treatment_img is None or cmp_post_treatment_img is None or cmp_matching_img is None:
            return -1 , -1 , -1,-1 , -1 , -1

        if cmp_pre_treatment_img.shape != cmp_post_treatment_img.shape and cmp_post_treatment_img.shape != cmp_matching_img.shape : 
            return -1 , -1 , -1,-1 , -1 , -1
        
        img_white = np.ones(cmp_pre_treatment_img.shape) * 255
        mse = mean_squared_error_ignore_zeros(cmp_pre_treatment_img ,    cmp_post_treatment_img,img_white)
        psnr = psnr_ignore_zeros( cmp_pre_treatment_img ,    cmp_post_treatment_img,img_white)
        ssim = ssim_ignore_zeros(cmp_pre_treatment_img ,    cmp_post_treatment_img,img_white)
        ssim = ssim * 100
        ncc = ncc_ignore_zeros(cmp_pre_treatment_img, cmp_post_treatment_img,cmp_matching_img)
        
        # 僅保留 cmp_matching_img 不為0的部分 進行比較
        cmp_pre_treatment_img[cmp_matching_img == 0] = 0
        cmp_post_treatment_img[cmp_matching_img == 0] = 0
        
        matching_mse = mean_squared_error_ignore_zeros(cmp_pre_treatment_img, cmp_matching_img,cmp_matching_img)
        matching_psnr = psnr_ignore_zeros(cmp_pre_treatment_img, cmp_matching_img,cmp_matching_img)
        matching_ssim = ssim_ignore_zeros(cmp_pre_treatment_img, cmp_matching_img,cmp_matching_img)
        matching_ssim = matching_ssim * 100
        
        
        
        matching_ncc = ncc_ignore_zeros(cmp_pre_treatment_img, cmp_matching_img,cmp_matching_img)
        


        
        return mse,psnr,ssim ,ncc,matching_mse,matching_psnr,matching_ssim,matching_ncc
    
    # def evaluate(self,image ,original_image):
    #     image = cv2.imread(image, cv2.IMREAD_GRAYSCALE)
    #     original_image = cv2.imread(original_image, cv2.IMREAD_GRAYSCALE)
    #     mse = mean_squared_error_ignore_zeros(image, original_image)
    #     psnr = peak_signal_noise_ratio(image, original_image)
    #     ssim = structural_similarity(image, original_image)
    #     ssim = ssim * 100
    #     # Calculate the mean of the two images
    #     mean_image1 = np.mean(original_image)
    #     mean_image2 = np.mean(image)

    #     # Calculate the NCC
    #     ncc = np.sum((original_image - mean_image1) * (image - mean_image2)) / (
    #         np.sqrt(np.sum((original_image - mean_image1)**2) * np.sum((image - mean_image2)**2))
    #     )

    #     return mse,psnr,ssim,ncc

    def get_pre_treatment_evaluation(self,file_path):
        patient_dict = read_patient_list(file_path)
        avg_mse = 0
        avg_psnr = 0
        avg_ssim = 0
        count = 0
        mse_list = []
        psnr_list = []
        ssim_list = []
        ncc_list = []
        for patient, eyes in patient_dict.items():
            for eye,date_list in eyes.items():
                if len(date_list) > 1:
                    path_pre_date = pl.Path(self.output_image_path + patient + '/'+date_list[0] + '/' + eye )
                    if path_pre_date.exists() and path_pre_date.is_dir() :
                        
                        for date in date_list[1:]:
                            path_date = pl.Path(self.image_path + patient + '/'+date + '/' + eye )
                            if path_date.exists() and path_date.is_dir() :
                                img_mse = []
                                img_psnr = []
                                img_ssim = []
                                img_ncc = []
                                img_avg_mse = -1
                                img_avg_psnr = -1
                                img_avg_ssim = -1
                                img_avg_ncc = -1
                                best_case = dict()
                                best_ssim = -1
                                worst_case = dict()
                                worst_ssim = 10000000000000
                                for img in self.layer_list:
                                    pre_treatment =  self.image_path + patient + '/'+date_list[0] + '/' + eye + '/' + img + '.png'
                                    original = self.image_path + patient + '/'+date + '/' + eye + '/' + img + '.png'
                                    mse,psnr,ssim,ncc = self.evaluate(pre_treatment,original)
                                    img_mse.append(mse)
                                    img_psnr.append(psnr)
                                    img_ssim.append(ssim)
                                img_avg_mse  = sum(img_mse)/len(img_mse)
                                img_avg_psnr = sum(img_psnr)/len(img_psnr)
                                img_avg_ssim = sum(img_ssim)/len(img_ssim)
                                if img_avg_ssim > best_ssim :
                                    best_ssim = img_avg_ssim
                                    best_case['patient'] = [ patient,eye,date]
                                    best_case ['psnr'] = img_avg_psnr
                                    best_case ['ssim'] = img_avg_ssim
                                    best_case ['mse'] = img_avg_mse
                                    best_case ['ncc'] = img_avg_ncc
                                    
                                if img_avg_ssim < worst_ssim :
                                    worst_ssim = img_avg_ssim
                                    worst_case['patient'] = [ patient,eye,date]
                                    worst_case ['psnr'] = img_avg_psnr
                                    worst_case ['ssim'] = img_avg_ssim
                                    worst_case ['mse'] = img_avg_mse
                                    worst_case ['ncc'] = img_avg_ncc
                                    
                                mse_list.append(img_avg_mse)
                                psnr_list.append(img_avg_psnr)
                                ssim_list.append(img_avg_ssim)
                                ncc_list.append(img_avg_ncc)


        avg_mse = round(sum(mse_list)/len(mse_list),5)
        avg_psnr = round( sum(psnr_list)/len(psnr_list),5)
        avg_ssim = round( sum(ssim_list)/len(ssim_list),5)
        avg_ncc = round( sum(ncc_list)/len(ncc_list),5)
        mse_std = round(np.std(mse_list, ddof=1),5)
        psnr_std = round(np.std(psnr_list, ddof=1),5)
        ssim_std = round(np.std(ssim_list, ddof=1),5)
        ncc_std = round(np.std(ncc_list, ddof=1),5)


        eval = {}
        eval['avg_mse'] = avg_mse
        eval['mse_std'] = mse_std
        eval['avg_psnr'] = avg_psnr
        eval['psnr_std'] = psnr_std
        eval['avg_ssim'] = avg_ssim
        eval['ssim_std'] = ssim_std
        eval['avg_ncc'] = avg_ncc
        eval['ncc_std'] = ncc_std
        eval['best_case'] = best_case
        eval['worst_case'] = worst_case
        return eval

    
    def avg_evaluate(self):
        eval = {}
        original = {}
        best_method = {}

        for method in pl.Path(self.output_image_path).iterdir():
            print('method',method)
            if method.parts[-1] != 'crop':
                
                if method.is_dir():
                    method_name = method.name

                    eval[method_name] = {}
                    patient = {}
                    image_folder =str(method.joinpath('1')) 
                    filenames = sorted(os.listdir(image_folder))

                    best_case = dict()
                    worst_case = dict()
                    best_differece_ssim = -1
                    worst_differece_ssim = 100000000000000000000
                    avg_mse = 0
                    avg_psnr = 0
                    avg_ssim = 0
                    mse_list = []
                    psnr_list = []
                    ssim_list = []
                    ncc_list = []
                    matching_mse_list = []
                    matching_psnr_list = []
                    matching_ssim_list = []
                    matching_ncc_list = []
                    for filename in filenames: 
                        if filename.endswith(".png"):

                            patient_id, eye, date = filename.split('.png')[0].split('_')
                            if patient_id + '_' + eye not in patient :
                                patient[patient_id + '_' + eye] = {}
                                post_treatment = ''
                                pre_treatment = date
                            else :
                                post_treatment =  date 

                                pre_treatment_img = self.image_path+ '1/' + patient_id + '_' + eye + '_' + pre_treatment + '.png'
                                post_treatment_img = self.image_path + '1/' + patient_id + '_' + eye + '_' + post_treatment + '.png'
                                matching_img = self.output_image_path + method_name + '/1_move/' + patient_id + '_' + eye + '_' + post_treatment + '.png'
                                

                                
                                
                                mse,psnr,ssim,ncc,matching_mse,matching_psnr,matching_ssim,matching_ncc = self.evaluates(pre_treatment_img,post_treatment_img,matching_img)
                                mse_list.append(mse)
                                psnr_list.append(psnr)
                                ssim_list.append(ssim)
                                ncc_list.append(ncc)


                               
                                matching_mse_list.append(matching_mse)
                                matching_psnr_list.append(matching_psnr)
                                matching_ssim_list.append(matching_ssim)
                                matching_ncc_list.append(matching_ncc)

                                if matching_ssim - ssim > best_differece_ssim :
                                    best_differece_ssim = matching_ssim - ssim
                                    best_case['patient'] = [ patient_id,eye,post_treatment]
                                    best_case ['psnr'] = psnr
                                    best_case ['ssim'] = ssim
                                    best_case ['mse'] = mse
                                    best_case ['ncc'] = ncc
                                    best_case ['matching_psnr'] = matching_psnr
                                    best_case ['matching_ssim'] = matching_ssim
                                    best_case ['matching_mse'] = matching_mse
                                    best_case ['matching_ncc'] = matching_ncc

                                if matching_ssim - ssim < worst_differece_ssim :
                                    worst_differece_ssim = matching_ssim - ssim
                                    worst_case['patient'] = [ patient_id,eye,post_treatment]
                                    worst_case ['psnr'] = psnr
                                    worst_case ['ssim'] = ssim
                                    worst_case ['mse'] = mse
                                    worst_case ['ncc'] = ncc
                                    worst_case ['matching_psnr'] = matching_psnr
                                    worst_case ['matching_ssim'] = matching_ssim
                                    worst_case ['matching_mse'] = matching_mse
                                    worst_case ['matching_ncc'] = matching_ncc


                    matching_avg_mse = round(sum(matching_mse_list)/len(matching_mse_list),3)
                    matching_avg_psnr = round(sum(matching_psnr_list)/len(matching_psnr_list),3)
                    matching_avg_ssim = round(sum(matching_ssim_list)/len(matching_ssim_list),3)
                    matching_avg_ncc = round(sum(matching_ncc_list)/len(matching_ncc_list),3)
                    matching_mse_std = round(np.std(matching_mse_list, ddof=1),3)
                    matching_psnr_std = round(np.std(matching_psnr_list, ddof=1),3)
                    matching_ssim_std = round(np.std(matching_ssim_list, ddof=1),3)
                    matching_ncc_std = round(np.std(matching_ncc_list, ddof=1),3)
                    eval[method_name]['mse'] = [matching_avg_mse,matching_mse_std]
                    eval[method_name]['psnr'] = [matching_avg_psnr,matching_psnr_std]
                    eval[method_name]['ssim'] = [matching_avg_ssim,matching_ssim_std]
                    eval[method_name]['ncc'] = [matching_avg_ncc,matching_ncc_std]
                    eval[method_name]['case'] = len(mse_list)

                    if best_case != {} :
                        eval[method_name]['best_case'] = best_case
                    if worst_case != {} :
                        eval[method_name]['worst_case'] = worst_case




                    if original == {} :
                        avg_mse = round(sum(mse_list)/len(mse_list),3)
                        avg_psnr = round(sum(psnr_list)/len(psnr_list),3)
                        avg_ssim = round(sum(ssim_list)/len(ssim_list),3)
                        avg_ncc = round(sum(ncc_list)/len(ncc_list),3)
                        mse_std = round(np.std(mse_list, ddof=1),3)
                        psnr_std = round(np.std(psnr_list, ddof=1),3)
                        ssim_std = round(np.std(ssim_list, ddof=1),3)
                        ncc_std = round(np.std(ncc_list, ddof=1),3)

                        eval['original'] = {}
                        eval['original']['mse'] = [avg_mse,mse_std]
                        eval['original']['psnr'] = [avg_psnr,psnr_std]
                        eval['original']['ssim'] = [avg_ssim,ssim_std]
                        eval['original']['ncc'] = [avg_ncc,ncc_std]


        # find best method
        best_method = {}
        for method in eval:
            if method != 'original':
                if best_method == {} :
                    best_method = method
                else :
                    if eval[method]['ssim'][0] > eval[best_method]['ssim'][0] :
                        best_method = method
                        
        eval['best_method'] = best_method
        

        return eval


    def alignment(self,file_path):
        # patient_dict = read_patient_list(file_path)
        method_dict = {}
        image_folder = self.image_path+ '/1/'
        for method in self.method_name:
            print('method',method)
            method_dict[method] = {}
            filenames = sorted(os.listdir(image_folder))
            for filename in filenames: 
                if filename.endswith(".png"):

                    patient_id, eye, date = filename.split('.png')[0].split('_')
                    if patient_id not in method_dict[method]:
                        post_treatment = ''
                        method_dict[method][patient_id] = {}
                    if eye not in method_dict[method][patient_id]:
                        pre_treatment =  date
                        post_treatment = ''
                        method_dict[method][patient_id][eye] = {}
                    else :
                        post_treatment =  date 
                        method_dict[method][patient_id][eye][date] = {}
                    treatment_patient = patient_id
                    treatment_eye = eye


                    if pre_treatment != '' and post_treatment != '':
                        shift = self.template_matching(treatment_patient, treatment_eye, pre_treatment, post_treatment,method)
                        print(treatment_patient, treatment_eye, pre_treatment, post_treatment,shift)
                        method_dict[method][patient_id][eye][date] = shift[method]

                    
        return method_dict
                

    def template_matching(self, patient_id, eye, pre_treatment, post_treatment,match_par):
        pre_img = cv2.imread(self.image_path + '1/' + patient_id + '_' + eye + '_' + pre_treatment + '.png')
        post_img = cv2.imread(self.image_path + '1/' + patient_id + '_' + eye + '_' + post_treatment + '.png')
        if pre_img is None or post_img is None:
            return
        gray_pre = cv2.cvtColor(pre_img, cv2.COLOR_BGR2GRAY)
        gray_post = cv2.cvtColor(post_img, cv2.COLOR_BGR2GRAY)
        shift = dict()

        method = self.methods[self.method_name.index(match_par)]

        # print method name
        shift[match_par]= []
        shift_x, shift_y = self.pointMatch(patient_id, eye, pre_treatment, post_treatment, gray_pre, gray_post, method)
        
        if shift_x > 304//2 or shift_x < -304//2 or shift_y > 304//2 or shift_y < -304//2:
            shift_x = 0
            shift_y = 0
            
        
        
        shift[match_par] = [shift_x, shift_y]
        # 進行平移變換
        M = np.float32([[1, 0, shift_x], [0, 1, shift_y]]) # 平移矩陣


        for layer in self.layer_list:

            if os.path.exists(self.image_path + layer + '/'  + patient_id + '_' + eye + '_' + pre_treatment + '.png'):
                pre_image = cv2.imread(self.image_path + layer + '/'  + patient_id + '_' + eye + '_' + pre_treatment + '.png')
                pre_image = cv2.resize(pre_image, self.image_size)
                pre_image = cv2.normalize(pre_image, None, 0, 255, cv2.NORM_MINMAX)
                
                output_match_par = 'crop_'+match_par

                cv2.imwrite(self.output_image_path+ output_match_par+ '/' + layer + '/' + patient_id + '_' + eye + '_' + pre_treatment + '.png', pre_image)
                if os.path.exists(self.image_path + layer + '/'  + patient_id + '_' + eye + '_' + post_treatment + '.png'):
                    image = cv2.imread(self.image_path + layer + '/' + patient_id + '_' + eye + '_' + post_treatment + '.png')
                    result = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))
                    
                    if not os.path.exists(self.output_image_path  + output_match_par+ '/' + layer ):
                        os.makedirs(self.output_image_path  + output_match_par+ '/' + layer )
                    if not os.path.exists(self.output_image_path  + output_match_par+ '/' + layer +'_move/'):
                        os.makedirs(self.output_image_path  + output_match_par+ '/' + layer +'_move/')
                        

                    cv2.imwrite(self.output_image_path   + output_match_par+ '/' + layer + '_move/'  + patient_id + '_' + eye + '_' + post_treatment + '.png', result)
                    result[result == 0] = pre_image [result == 0]
                    

                    cv2.imwrite(self.output_image_path  + output_match_par+ '/' + layer + '/' + patient_id + '_' + eye + '_' + post_treatment + '.png', result)

                    vis_img = result.copy()
                    vis_img[:,:,0] = 0
                    vis_img[:,:,2] = 0
                    add = cv2.addWeighted(pre_image, 0.5, vis_img, 1.0, 0)
                    if not os.path.exists(self.output_image_path  + output_match_par+ '/' + layer + '_vis' ):
                        os.makedirs(self.output_image_path  + output_match_par+ '/' + layer + '_vis' )
                    cv2.imwrite(self.output_image_path  + output_match_par+ '/' + layer + '_vis' + '/' + patient_id + '_' + eye + '_' + post_treatment +'_vis.png', add)


        for label in self.label_list:
            output_match_par = 'crop_'+match_par
            if not os.path.exists(self.output_label_path + '/' + output_match_par +  '/' + label ):
                os.makedirs(self.output_label_path + '/' +output_match_par  +  '/' + label )
            if not os.path.exists(self.output_label_path + '/' +output_match_par  +  '/' + label +'_move/'):
                os.makedirs(self.output_label_path + '/' + output_match_par + '/' + label +'_move/')
            mask_path = os.path.join(os.path.dirname(os.path.dirname(self.image_path)),label,'masks')
            
            # label_path = os.path.join(path,'masks',label+ '_'+ )
            if os.path.exists(mask_path + '/' + patient_id + '_' + eye + '_' + pre_treatment + '.png'):
                pre_label = cv2.imread(mask_path + '/' + patient_id + '_' + eye + '_' + pre_treatment + '.png')
                pre_label = cv2.resize(pre_label, self.image_size)
                pre_label = cv2.normalize(pre_label, None, 0, 255, cv2.NORM_MINMAX)
                # print('post_treatment',self.output_label_path+ match_par+ '/' + label + '/' + patient_id + '_' + eye + '_' + pre_treatment + '.png')
                cv2.imwrite(self.output_label_path+ output_match_par+ '/' + label + '/' + patient_id + '_' + eye + '_' + pre_treatment + '.png', pre_label)                    
                
                if os.path.exists(mask_path + '/' + patient_id + '_' + eye + '_' + post_treatment + '.png'):
                    post_label = cv2.imread(mask_path + '/' + patient_id + '_' + eye + '_' + post_treatment + '.png')
                    post_label = cv2.resize(post_label, self.image_size)
                    label_result = cv2.warpAffine(post_label, M, (image.shape[1], image.shape[0]))
                    
                    # print('post_treatment',self.output_label_path  + match_par+ '/' + label + '_move/'  + patient_id + '_' + eye + '_' + post_treatment + '.png',)
                    cv2.imwrite(self.output_label_path  + output_match_par+ '/' + label + '_move/'  + patient_id + '_' + eye + '_' + post_treatment + '.png', label_result)
                    # label_result[label_result == 0] = pre_label [label_result == 0]
                    
                    if not os.path.exists(self.output_label_path  + output_match_par+ '/' + label ):
                        os.makedirs(self.output_label_path  + output_match_par+ '/' + label )
                    if not os.path.exists(self.output_label_path  + output_match_par+ '/' + label +'_move/'):
                        os.makedirs(self.output_label_path  + output_match_par+ '/' + label +'_move/')
                    # print('post_treatment',self.output_label_path+ match_par+ '/' + label + '/' + patient_id + '_' + eye + '_' + post_treatment + '.png')
                    cv2.imwrite(self.output_label_path  + output_match_par+ '/' + label + '/' + patient_id + '_' + eye + '_' + post_treatment + '.png', label_result)

        return shift
                     


def read_patient_list(path):
    f = open(path, 'r')
    patient = dict()
    patient_id = '' 
    for line in f.readlines():
        line = line.strip(' : \n')   
        if 'total' in line:
            break   
        if patient_id !='':
            if patient_id not in patient:
                patient[patient_id] = dict()
        line = line.split('\t')
        if(len(line) > 1):
            eyes = line[1].split(' : ')
            date = eyes[1].split(', ')
            if eyes[0] == 'OD':
                patient[patient_id]['R'] = date
            elif eyes[0] == 'OS':
                patient[patient_id]['L'] = date
        else :
            patient_id = line[0]
                # patient
    f.close()
    return patient

    

def setFolder(path):
    os.makedirs(path, exist_ok=True)




if __name__ == '__main__':
    # pre_treatment = "..\\..\\Data\\PPT\\00294362_20210511_1.png"
    # post_treatment = "..\\..\\Data\\PPT\\00294362_20221222_1.png"
    # matching().template_matching(pre_treatment, post_treatment)
    date = '0305'
    disease = 'PCV'
    PATH_DATA = '../../Data/' 
    PATH_BASE = PATH_DATA  + disease + '_' + date + '/'

    PATH_LABEL = PATH_DATA + 'labeled' + '/'
    PATH_IMAGE = PATH_DATA + 'OCTA/' 
    image_path = PATH_BASE + 'inpaint/'
    PATH_MATCH = image_path + 'MATCH/' 
    PATH_MATCH_LABEL = image_path + 'MATCH_LABEL/' 
    
    
    tools.makefolder('./record/' + disease + '_' + date + '/')
    Match = template_matcher(image_path,PATH_LABEL,PATH_MATCH,PATH_MATCH_LABEL)
    # shift_patient_dict = Match.alignment('PCV_exist.txt')
    # json_file = './record/' + disease + '_' + date + '/' + 'crop_template_matching.json'
    # tools.write_to_json_file(json_file, shift_patient_dict)
    eval = Match.avg_evaluate()
    json_file = './record/' + disease + '_' + date + '/' + 'eval.json'
    tools.write_to_json_file(json_file, eval)
    
    # # eval = Match.all_evaluate(image_path + '
    # aligment_file = './record/'+ disease + '_' + date + '/' + 'evaluations.csv'
    # if not os.path.exists(aligment_file):
    #     with open(aligment_file, 'w') as f:
    #         csv_writer = csv.writer(f)  
    #         csv_writer.writerow(['feature', 'cases','avg_mse', 'avg_psnr', 'avg_ssim','std_mse', 'std_psnr', 'std_ssim','avg_matching_mse', 'avg_matching_psnr', 'avg_matching_ssim','std_matching_mse', 'std_matching_psnr', 'std_matching_ssim'])

    # with open(aligment_file, 'a') as f:
    #     csv_writer = csv.writer(f)
    #     csv_writer.writerow([feature + '_' + matcher + '_' + str(distance),
    #                             cases ,
    #                             eval["avg"]['original']['mse'],
    #                             eval["avg"]['original']['psnr'],
    #                             eval["avg"]['original']['ssim'],
    #                             eval["std"]['original']['mse'],
    #                             eval["std"]['original']['psnr'],
    #                             eval["std"]['original']['ssim'],
    #                             eval["avg"]['matching']['mse'],
    #                             eval["avg"]['matching']['psnr'],
    #                             eval["avg"]['matching']['ssim'],
    #                             eval["std"]['matching']['mse'],
    #                             eval["std"]['matching']['psnr'],
    #                             eval["std"]['matching']['ssim']
    #                             ])
    # evals= Match.avg_evaluate() 
    # for method, eval in evals.items():
    #     print(method,eval)
    # json_file = './' + disease + '_' + date + '_eval.json'
    # tools.write_to_json_file(json_file, evals)

    # to csv
    # with open('./' + disease + '_' + date + '_eval.csv', 'w', newline='') as csvfile:
    #     # 建立 CSV 檔寫入器
    #     writer = csv.writer(csvfile)
    #     writer.writerow(['method','mse','psnr','ssim'])
    #     for method, eval in evals.items():
    #         writer.writerow([method,eval['mse'],eval['psnr'],eval['ssim']])




    # original_eval= Match.get_pre_treatment_evaluation('PCV_exist.txt')
    # json_file = './original_eval.json'
    # tools.write_to_json_file(json_file, original_eval)

    #     path_patient = PATH_IMAGE + patient
    #     patient = patient
    #     Match.alignment(patient)
        # matching().alignment(path_patient, path_output, path_label, path_label_output)
        # break
        