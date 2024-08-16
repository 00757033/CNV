import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import math
import json
import shutil
import time
import csv
import tools.tools as tools
from skimage.feature import corner_harris
from skimage.metrics import mean_squared_error
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from sklearn.metrics import normalized_mutual_info_score
from skimage.filters import unsharp_mask
import pandas as pd
import pathlib as pl
from datetime import datetime
def setFolder(path):
    os.makedirs(path, exist_ok=True)
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
    
    diff = abs(img1[~both_zeros_mask] - img2[~both_zeros_mask])
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
    

def NCC_ignore_zeros(img1, img2,img3):
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
    
    # NCC
    mean1 = np.mean(img1_non_zero.flatten())
    mean2 = np.mean(img2_non_zero.flatten())
    
    ncc = np.sum((img1_non_zero.flatten() - mean1) * (img2_non_zero.flatten() - mean2)) / np.sqrt(np.sum((img1_non_zero.flatten() - mean1) ** 2) * np.sum((img2_non_zero.flatten() - mean2) ** 2))
    # print('ncc',ncc)
    return ncc    

def NMI_ignore_zeros(img1, img2,img3):
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
    # NMI
    nmi = normalized_mutual_info_score(img1_non_zero.flatten(), img2_non_zero.flatten())
    # print('nmi',nmi)
    return nmi

def Correlation_coefficient_ignore_zeros(img1, img2,img3):
    # 對於兩張影像 計算每個pixel 的差異
    # 找出兩影像中相應像素都為零的位置
    msk1 = img1 == 0
    msk2 = img2 == 0
    msk3 = img3 == 0
    both_zeros_mask = np.logical_and(np.logical_and(msk1, msk2), msk3)
    
    corr = np.corrcoef(img1[~both_zeros_mask], img2[~both_zeros_mask])[0, 1]
    # print('corr',corr)
    return corr

            
# 尋找OCTA 影像的黃斑中心
class finding():
    def __init__(self,label_path,image_path,label_list,data_groups,output_label_path,output_image_path,methods,matchers,distance,file_name):
        self.label_path = label_path
        self.image_path = image_path
        self.distances = distance
        self.methods = methods 
        self.matchers = matchers
        self.output_image_path = output_image_path
        self.output_label_path = output_label_path
        self.image_size= (304, 304)
        self.data_list = ['1','2', '3', '4', '1_OCT', '2_OCT', '3_OCT', '4_OCT']
        self.label_list = label_list
        self.data_groups = data_groups
        self.methods_template = [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED, cv2.TM_CCORR , cv2.TM_CCORR_NORMED, cv2.TM_CCOEFF, cv2.TM_CCOEFF_NORMED]
        self.method_template_name = ['TM_SQDIFF', 'TM_SQDIFF_NORMED', 'TM_CCORR' , 'TM_CCORR_NORMED', 'TM_CCOEFF', 'TM_CCOEFF_NORMED']
        self.inject(file = file_name)

    def inject(self,file = '../../Data/打針資料.xlsx',label = ["診斷","病歷號","眼睛","打針前門診日期","三針後門診"]):
        # self.inject_df = pd.DataFrame()
        print(file)
        self.inject_df = pd.read_excel(file, sheet_name="Focea_collect",na_filter = False, engine='openpyxl')
        
        # add pd.read_excel(file, sheet_name="20230831",na_filter = False, engine='openpyxl')
        # self.inject_df = self.inject_df.append(pd.read_excel(file, sheet_name="20230831",na_filter = False, engine='openpyxl'))

        self.inject_df['病歷號'] = self.inject_df['病歷號'].apply(lambda x: '{:08d}'.format(x))
        self.inject_df = self.inject_df.sort_values(by=["病歷號","眼睛"])


    def evaluates(self, pre_treatment_img, post_treatment_img,matching_img):
        cmp_pre_treatment_img = cv2.imread(pre_treatment_img , cv2.IMREAD_GRAYSCALE)
        cmp_post_treatment_img = cv2.imread(post_treatment_img, cv2.IMREAD_GRAYSCALE)
        cmp_matching_img = cv2.imread(matching_img, cv2.IMREAD_GRAYSCALE)
        
        evaluate = dict()
                          
        
        if cmp_pre_treatment_img is None or cmp_post_treatment_img is None or cmp_matching_img is None:
            return evaluate

        if cmp_pre_treatment_img.shape != cmp_post_treatment_img.shape and cmp_post_treatment_img.shape != cmp_matching_img.shape : 
            return evaluate
        


        
        # mse = mean_squared_error_ignore_zeros(cmp_pre_treatment_img ,    cmp_post_treatment_img,cmp_matching_img)
        # psnr = psnr_ignore_zeros( cmp_pre_treatment_img ,    cmp_post_treatment_img,cmp_matching_img)
        # ssim = ssim_ignore_zeros(cmp_pre_treatment_img ,    cmp_post_treatment_img,cmp_matching_img)
        # ssim = ssim * 100
        
        
        
        # mse = mean_squared_error(cmp_pre_treatment_img, cmp_post_treatment_img) 
        # psnr = peak_signal_noise_ratio(cmp_pre_treatment_img, cmp_post_treatment_img)
        # ssim = structural_similarity(cmp_pre_treatment_img, cmp_post_treatment_img)
        # ssim = ssim * 100
        # img_one all 255
        img_white = np.ones(cmp_pre_treatment_img.shape) * 255
        
        mse = mean_squared_error_ignore_zeros(cmp_pre_treatment_img ,cmp_post_treatment_img,img_white)
        psnr = psnr_ignore_zeros( cmp_pre_treatment_img ,    cmp_post_treatment_img,img_white)
        ssim = ssim_ignore_zeros(cmp_pre_treatment_img ,    cmp_post_treatment_img,img_white)
        ssim = ssim * 100
        ncc = NCC_ignore_zeros(cmp_pre_treatment_img ,    cmp_post_treatment_img,img_white)
        nmi = NMI_ignore_zeros(cmp_pre_treatment_img ,    cmp_post_treatment_img,img_white)
        corr = Correlation_coefficient_ignore_zeros(cmp_pre_treatment_img ,    cmp_post_treatment_img,img_white)
        
        cmp_pre_treatment_img[cmp_matching_img == 0] = 0
        cmp_post_treatment_img[cmp_matching_img == 0] = 0
        matching_mse = mean_squared_error_ignore_zeros(cmp_pre_treatment_img, cmp_matching_img,cmp_matching_img)
        matching_psnr = psnr_ignore_zeros(cmp_pre_treatment_img, cmp_matching_img,cmp_matching_img)
        matching_ssim = ssim_ignore_zeros(cmp_pre_treatment_img, cmp_matching_img,cmp_matching_img)
        matching_ssim = matching_ssim * 100
        matching_ncc = NCC_ignore_zeros(cmp_pre_treatment_img, cmp_matching_img,cmp_matching_img)
        matching_nmi = NMI_ignore_zeros(cmp_pre_treatment_img, cmp_matching_img,cmp_matching_img)
        matching_corr = Correlation_coefficient_ignore_zeros(cmp_pre_treatment_img, cmp_matching_img,cmp_matching_img)
        

        evaluate['mse'] = mse
        evaluate['psnr'] = psnr
        evaluate['ssim'] = ssim
        evaluate['ncc'] = ncc
        evaluate['nmi'] = nmi
        evaluate['corr'] = corr
        evaluate['matching_mse'] = matching_mse
        evaluate['matching_psnr'] = matching_psnr
        evaluate['matching_ssim'] = matching_ssim
        evaluate['matching_ncc'] = matching_ncc
        evaluate['matching_nmi'] = matching_nmi
        evaluate['matching_corr'] = matching_corr
        
        return evaluate
        



        
        
        
    # def adjust_brightness(self,img1, img2):
    #     # Convert images to float32 for accurate calculations
    #     img1 = img1.astype(np.float32)
    #     img2 = img2.astype(np.float32)

    #     # Compute the average intensity of each image
    #     avg_intensity_img1 = np.mean(img1)
    #     avg_intensity_img2 = np.mean(img2)

    #     # Compute the scaling factor to make the average intensities equal
    #     scaling_factor = avg_intensity_img1 / avg_intensity_img2

    #     # Scale the second image
    #     img2_scaled = img2 * scaling_factor

    #     # Clip the values to the valid intensity range (0-255)
    #     img2_scaled = np.clip(img2_scaled, 0, 255)

    #     # Convert the images back to uint8 format
    #     img1 = img1.astype(np.uint8)
    #     img2_scaled = img2_scaled.astype(np.uint8)

    #     return img1, img2_scaled

    def preprocess(self, image):
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)
        image_preprocess = image.copy()

        
        image_preprocess = cv2.equalizeHist(image_preprocess)
  
   
        kernel = np.ones((5, 5), np.uint8)

        # # # 開運算
        image_preprocess = cv2.morphologyEx(image_preprocess, cv2.MORPH_OPEN, kernel)
        image_preprocess = cv2.normalize(image_preprocess, None, 0, 255, cv2.NORM_MINMAX)

        # image_preprocess = cv2.equalizeHist(image_preprocess)

        # fig, ax = plt.subplots(1, 3)
        # ax[0].imshow(image, cmap='gray')
        # ax[0].set_title('original')
        # ax[0].axis('off')
        # ax[1].imshow(image_preprocess, cmap='gray')
        # ax[1].set_title('preprocessed')
        # ax[1].axis('off')
        # ax[2].hist(image.ravel(), bins=256, range=(0, 255), alpha=0.4, color='r', label='original')
        # ax[2].hist(image_preprocess.ravel(), bins=256, range=(0, 255), alpha=0.4, color='b', label='preprocessed')
        # ax[2].legend(loc='upper right')
        # plt.show()
        
        return image_preprocess


    def find_center(self,image,image_size = (304,304),save_path = None,save_name = None):
        image2 = image.copy()
        
        image2 = self.preprocess(image2)
        image2 = cv2.resize(image2, image_size)
        # rst,image2 = cv2.threshold(image2, 0, 255,  cv2.THRESH_BINARY  + cv2.THRESH_OTSU)
        
        # 二值化 32 255
        rst,image2 = cv2.threshold(image2, 16, 255,  cv2.THRESH_BINARY)
        
        image2 = cv2.bitwise_not(image2)
        # 找到影像中 最白的最大面積的圓
        _, labels, stats, centroids = cv2.connectedComponentsWithStats(image2, connectivity=4)

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
        if len(draw_image.shape) == 2:
            draw_image = cv2.cvtColor(draw_image, cv2.COLOR_GRAY2BGR)
        draw_image= cv2.resize(draw_image, image_size)
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
        
        
        t = 40
        if int(center[0] - max_radius) < t :
            t = int(center[0] - max_radius)
        if int(center[1] - max_radius) < t :
            t = int(center[1] - max_radius)
        if int(center[0] + max_radius) > len(image[0]) - t :
            t = len(image[0]) - int(center[0] + max_radius)
        if int(center[1] + max_radius) > len(image) - t :
            t = len(image) - int(center[1] + max_radius)
            
        # 畫出黃斑中心
        # draw the circle
        cv2.rectangle(draw_image, (int(center[0] - max_radius), int(center[1] - max_radius)), (int(center[0] + max_radius), int(center[1] + max_radius)), (255, 0, 255), 3)
        cv2.circle(draw_image, center, 5, (0, 255, 255), -1)
        
        cv2.rectangle(draw_image, (int(center[0] - max_radius-t), int(center[1] - max_radius-t)), (int(center[0] + max_radius+ t), int(center[1] + max_radius+ t)), (255, 255, 0), 3)

        # # show the image wait 3 seconds and destroy
        # fig , ax = plt.subplots(1,2,figsize=(10,10))
        # ax[0].imshow(draw_image)

        # ax[0].set_title('center')
        # ax[1].imshow(image)

        # ax[1].set_title('image')
        # plt.show()
        if save_path :
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            cv2.imwrite(save_path + save_name + '.png', draw_image)
            cv2.imwrite(save_path + save_name + '_origin.png', image)
            

        crop_img = image[int(center[1] - max_radius)-t:int(center[1] + max_radius)+t, int(center[0] - max_radius)-t:int(center[0] + max_radius)+t]
        # show the image wait 3 seconds and destroy
        return  crop_img , center , max_radius + t

    def LK(self,img1, img2, distance=0.9, method='KAZE',matcher='BF',N = 20):
        # Initiate SIFT detector
        if method == 'SIFT':
            detector  = cv2.xfeatures2d.SIFT_create()
        elif method == 'KAZE':
            detector  = cv2.KAZE_create()
        elif method == 'AKAZE':
            detector  = cv2.AKAZE_create()
        elif method == 'ORB':
            detector  = cv2.ORB_create()
        elif method == 'BRISK':
            detector  = cv2.BRISK_create()
        elif method == 'BRIEF':
            detector = cv2.FastFeatureDetector_create()
            descriptor  = cv2.xfeatures2d.BriefDescriptorExtractor_create( )

        elif method == 'FREAK':
            detector = cv2.FastFeatureDetector_create()
            descriptor = cv2.xfeatures2d.FREAK_create()
        if method in ['BRIEF', 'FREAK']:
            kp1 = detector.detect(img1)
            kp2 = detector.detect(img2)
            kp1, des1 = descriptor.compute(img1, kp1)
            kp2, des2 = descriptor.compute(img2, kp2)
        
        elif method == 'ORB':
            kp1, des1 = detector.detectAndCompute(img1, None)
            kp2, des2 = detector.detectAndCompute(img2, None)

        else:
            kp1 = detector.detect(img1, None)
            kp2 = detector.detect(img2, None)

            kp1, des1 = detector.compute(img1, kp1)
            kp2, des2 = detector.compute(img2, kp2)


        # img1_draw = cv2.drawKeypoints(img1, kp1, img1, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        # img2_draw = cv2.drawKeypoints(img2, kp2, img2, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        # Convert to numpy arrays
        if des1 is None or des2 is None or len(des1) < 2 or len(des2) < 2 :
            H = np.array(np.eye(3))
            return None


        des1 = des1.astype(np.float32)
        des2 = des2.astype(np.float32)

        # Matching method
        if matcher == 'BF':
                if method == 'ORB':
                    bf = cv2.BFMatcher()
                else:
                    bf = cv2.BFMatcher()
        elif matcher == 'FLANN':
            if method == 'ORB':
                index_params = dict(algorithm=6, table_number=6, key_size=12, multi_probe_level=1)
                search_params = dict(checks=60)
                des1 = des1.astype(np.uint8)
                des2 = des2.astype(np.uint8)
            else:
                index_params = dict(algorithm=1, trees=5)
                search_params = dict(checks=1000)
            bf = cv2.FlannBasedMatcher(index_params, search_params)
   
        # if method == 'SIFT'  or method == 'BRISK' or method == 'BRIEF':
        #     if  matcher == 'BF':
        #         bf = cv2.BFMatcher()
        #         if des1 is None or des2 is None or len(des1) < 2 or len(des2) < 2:
        #             H = np.array(np.eye(3))
        #             return None
        #         matches = bf.knnMatch(des1, des2, k=2)
        #         matches = sorted(matches, key=lambda x: x[0].distance)
        #         matches = matches[:N]

        #     elif matcher == 'FLANN':
        #         FLANN_INDEX_KDTREE = 1
        #         index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        #         search_params = dict(checks=1000)

        #         flann = cv2.FlannBasedMatcher(index_params, search_params)
        #         if des1 is None or des2 is None or len(des1) < 2 or len(des2) < 2:
        #             H = np.array(np.eye(3))
        #             return None
        #         matches = flann.knnMatch(des1, des2, k=2)
        #         matches = sorted(matches, key=lambda x: x[0].distance)
        #         matches = matches[:N]
        # elif method == 'ORB':
        #     if  matcher == 'BF':
        #         bf = cv2.BFMatcher()
        #         if des1 is None or des2 is None or len(des1) < 2 or len(des2) < 2:
        #             H = np.array(np.eye(3))
        #             return None
        #         matches = bf.knnMatch(des1, des2, k=2)
        #         matches = sorted(matches, key=lambda x: x[0].distance)
        #         matches = matches[:N]
        #     elif matcher == 'FLANN':
        #         FLANN_INDEX_LSH = 6
        #         index_params= dict(algorithm = FLANN_INDEX_LSH,
        #                         table_number = 6, # 12
        #                         key_size = 12,     # 20
        #                         multi_probe_level = 1)

        #         search_params = dict(checks=60)
        #         # Make sure descriptors are 2D arrays
        #         des1 = np.array(des1).astype(np.uint8)
        #         des2 = np.array(des2).astype(np.uint8)
        #         if des1 is None or des2 is None or len(des1) < 2 or len(des2) < 2:
        #             H = np.array(np.eye(3))
        #             return None
        #         flann = cv2.FlannBasedMatcher(index_params, search_params)
        #         matches = flann.knnMatch(des1, des2, k=2)
        #         matches = sorted(matches, key=lambda x: x[0].distance)
        #         matches = matches[:N]
        # elif method == 'KAZE' or method == 'AKAZE' or method == 'SURF' or method == 'FREAK':
        #     if  matcher == 'BF':
        #         bf = cv2.BFMatcher( cv2.NORM_L2)
        #         if des1 is None or des2 is None or len(des1) < 2 or len(des2) < 2:
        #             H = np.array(np.eye(3))
        #             return None
        #         matches = bf.knnMatch(des1, des2, k=2)
        #         matches = sorted(matches, key=lambda x: x[0].distance)
        #         matches = matches[:N]
        #     elif matcher == 'FLANN':
        #         FLANN_INDEX_KDTREE = 1
        #         index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        #         search_params = dict(checks=60)
        #         if des1 is None or des2 is None or len(des1) < 2 or len(des2) < 2:
        #             H = np.array(np.eye(3))
        #             return None

        #         flann = cv2.FlannBasedMatcher(index_params, search_params)
        #         matches = flann.knnMatch(des1, des2, k=2)
        #         matches = sorted(matches, key=lambda x: x[0].distance)
        #         matches = matches[:N]
        # if not matches:
        #     return None
        # Match descriptors
        matches = bf.knnMatch(des1, des2, k=2)
        
        matches = [m for m in matches if len(m) == 2]
        if len(matches) > 0:
            matches = sorted(matches, key=lambda x: x[0].distance)
        else:
            return None
        
        # Apply ratio test
        good_matches = []
        pts1 = []
        pts2 = []
        for m, n in matches:
            if m.distance < distance * n.distance:
                good_matches.append([m])
                pts2.append(kp2[m.trainIdx].pt)
                pts1.append(kp1[m.queryIdx].pt)
        print(len(pts1),len(pts2))        
        if len(pts1) < 4 or len(pts2) < 4:
            H = np.array(np.eye(3))
            return None
        
        # # Draw matches
        # img3 = cv2.drawMatchesKnn(img1, kp1, img2, kp2, good_matches, None, flags=2)
        # plt.imshow(img3)
        # plt.axis('off')
        # plt.title(method)
        # plt.show()
        
        
        
        # if matcher == 'BF': 
        #     # Need to draw only good matches, so create a mask
        #     matches = sorted(matches, key=lambda x: x[0].distance)
        #     if matches:
        #         min_dist = matches[0][0].distance

        #         # Apply ratio test
        #         good = []
        #         pts1 = []
        #         pts2 = []
        #         for i, mn in enumerate(matches):
        #             if len(mn)== 2 :
        #                 m,n = mn
        #                 if m.distance < distance* n.distance:
        #                     good.append([m])
        #                     pts2.append(kp2[m.trainIdx].pt)
        #                     pts1.append(kp1[m.queryIdx].pt)

        #                 if m.distance > 1.5 * min_dist:
        #                     break
        #     # Draw 保留的matches
        #     # img3 = cv2.drawMatchesKnn(img1,kp1,img2,kp2,good,None,flags=2)
            
        # elif matcher == 'FLANN':
        #     # Need to draw only good matches, so create a mask
        #     matchesMask = [[0, 0] for i in range(len(matches))]
        #     pts1 = []
        #     pts2 = []
        #     if matches and matches[0]:
        #         # matches = sorted(matches, key=lambda x: x[0].distance)
        #         min_dist = matches[0][0].distance
        #         # ratio test as per Lowe's paper
                
        #         for i, match in enumerate(matches):
        #             if len(match) != 2 or match[0] is None or match[1] is None:
        #                 continue
        #             if len(match) == 2:
        #                 m, n = match
        #             if m.distance < distance * n.distance:
        #                 matchesMask[i] = [1, 0]
        #                 pts2.append(kp2[m.trainIdx].pt)
        #                 pts1.append(kp1[m.queryIdx].pt)
        #         draw_params = dict(matchColor = (0,255,0),
        #             singlePointColor = (255,0,0),
        #             matchesMask = matchesMask,
        #             flags = 0)
        #         img3 = cv2.drawMatchesKnn(img1,pts1,img2,pts2,matches,None,**draw_params)


        # plt.imshow(img3)
        # plt.axis('off')
        # plt.title(method)
        # plt.show()

        # 計算img1 需要旋轉 縮放 平移矩陣 原本img1的實際中心點為center
        pts1 = np.array(pts1)
        pts2 = np.array(pts2)

        H, _ = cv2.findHomography(pts2, pts1, cv2.RANSAC, 5.0)

        return H

    # def evaluate(self,image ,original_image):
    #     image = cv2.imread(image, cv2.IMREAD_GRAYSCALE)
    #     original_image = cv2.imread(original_image, cv2.IMREAD_GRAYSCALE)
    #     if image is None or original_image is None:
    #         return 0 , 0 , 1000000

    #     if image.shape != original_image.shape:
    #         return 0 , 0 , 1000000
            
        
    #     mse = mean_squared_error(image, original_image)
    #     psnr = peak_signal_noise_ratio(image, original_image)
    #     ssim = structural_similarity(image, original_image)
    #     ssim = ssim * 100

    #     return mse,psnr,ssim

    def match_histogram(self,source, reference):
        # Convert images to grayscale
        source_gray = cv2.cvtColor(source, cv2.COLOR_BGR2GRAY)
        reference_gray = cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY)

        # Calculate histograms
        hist_source = cv2.calcHist([source_gray], [0], None, [256], [0, 256])
        hist_reference = cv2.calcHist([reference_gray], [0], None, [256], [0, 256])

        # Normalize histograms
        hist_source /= hist_source.sum()
        hist_reference /= hist_reference.sum()

        # Calculate cumulative distribution functions (CDF)
        cdf_source = hist_source.cumsum()
        cdf_reference = hist_reference.cumsum()

        # Create a mapping function
        lut = np.interp(cdf_source, cdf_reference, range(256))

        # Apply the mapping to the source image
        matched_image = cv2.LUT(source_gray, lut.astype('uint8'))

        # Convert back to color if the input images were color images
        if len(source.shape) == 3:
            matched_image = cv2.cvtColor(matched_image, cv2.COLOR_GRAY2BGR)

        return matched_image
        
    def check_H_range(self,H):
        if H is not None:
            translation = (H[0, 2], H[1, 2])
            rotation_rad = math.atan2(H[1, 0], H[0, 0])
            rotation_angle= np.arctan2(H[1, 0], H[0, 0]) * 180 / np.pi
            scale_x =  np.linalg.norm(H[:, 0])
            scale_y = np.linalg.norm(H[:, 1])
            # print('translation',translation)
            # print('rotation_rad',rotation_rad)
            # print('rotation_angle',rotation_angle)
            # print('scale x',scale_x)
            # print('scale y',scale_y)
            
            
            if translation[0] > 304 // 2 or translation[0] < -304 // 2 or translation[1] > 304 // 2 or translation[1] < -304 // 2:
                # print('translation out of range')
                return False
            if rotation_angle > 30 or rotation_angle < -30:
                # print('rotation out of range')
                return False
            if scale_x > 1.5 or scale_x < 0.5 or scale_y > 1.5 or scale_y < 0.5:
                # print('scale out of range')
                return False
            return True
        else:
            return False

    def getPoints(self,img, template, method=cv2.TM_CCOEFF_NORMED):
        # print('shape',img.shape,template.shape)
        result = cv2.matchTemplate(img, template, method) # 回傳的是相關係數矩陣
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result) # 找到最大值和最小值的位置 也就是左上角的位置 以及右下角的位置
        if self.method_template_name[method] in ['TM_SQDIFF', 'TM_SQDIFF_NORMED']:
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

    def pointMatch(self,pre_img,crop_img,center,radius, method = cv2.TM_CCOEFF_NORMED):
        elements_c, r_c = self.get_element(pre_img, crop_img, center[0] - radius, center[1] - radius, method)
        # 選擇最高的相關係數的位置
        e = [elements_c]
        r = [r_c]
        Relation = 0
        relation_min = 1000000000000000000000000000
        shift_x = None
        shift_y = None
        for i in range(0, len(r)):

            if self.method_template_name[method] in ['TM_SQDIFF', 'TM_SQDIFF_NORMED']:

                if r[i] < relation_min:
                    relation_min = r[i]
                    shift_x = e[i][0] 
                    shift_y = e[i][1]
            elif  self.method_template_name[method] in ['TM_CCORR']:  
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
    
    def feature(self, feature, matcher, distance,data_groups,mask_folder_name = 'predict'):
        patient_dict = {}
        eyes = {'R':'OD','L':'OS'}
        image_folder = self.output_image_path+ '/1/'
        filenames = sorted(os.listdir(image_folder))
        self.output_path  = self.image_path + '/' + 'crop_'+ feature + '_' + matcher + '_' + str(distance) + '/'
        for data in self.data_list:
            if not os.path.exists(os.path.join(self.output_path, data)):
                os.makedirs(os.path.join(self.output_path, data))
            # if not os.path.exists(os.path.join(self.output_path, data+ '_vis/')):
            #     os.makedirs(os.path.join(self.output_path, data+ '_vis/'))
            if not os.path.exists(os.path.join(self.output_path, data+ '_move/')):
                os.makedirs(os.path.join(self.output_path, data+ '_move/'))
        print(feature, matcher, distance)
        print('------------------')
        pre_treatment_file = ''
        post_treatment_file = ''
        for filename in filenames:
            if filename.endswith('.png'):

                patient_id, eye, date = filename.split('.png')[0].split('_')
                # patient_id = '07838199'
                # eye = 'L'
                print(patient_id, eye, date)
                if patient_id + '_' + eye not in patient_dict:   
                    if patient_id in self.inject_df['病歷號'].values:
                        if eyes[eye] not in self.inject_df[(self.inject_df['病歷號'] == patient_id)]['眼睛'].values:
                            continue
                        predate = str(self.inject_df[(self.inject_df['病歷號'] == patient_id) & (self.inject_df['眼睛'] == eyes[eye])]['打針前門診日期'].values[0])
                        if predate != 'nan' and predate != 'NaT':
                            if 'T' in  predate:
                                predate = predate.split('T')[0]
                            if ' '  in predate:
                                predate = predate.split(' ')[0]
                            formatted_date_str = datetime.strptime(predate, '%Y-%m-%d').strftime('%Y%m%d')
                            if date < formatted_date_str:
                                continue
                    else:
                        continue
                    patient_dict[patient_id + '_' + eye] = {}
                    pre_treatment_file = filename
                    pre_treatment = date
                    # pre_treatment = '20211013'   
                else:
                    if patient_id in self.inject_df['病歷號'].values:
                        if eyes[eye] not in self.inject_df[(self.inject_df['病歷號'] == patient_id)]['眼睛'].values:
                                continue
                        postdate = str(self.inject_df[(self.inject_df['病歷號'] == patient_id) & (self.inject_df['眼睛'] == eyes[eye])]['三針後門診'].values[0])
                        if postdate != 'nan' and postdate != 'NaT':
                            if 'T' in  postdate:
                                postdate = postdate.split('T')[0]
                            if ' '  in postdate:
                                postdate = postdate.split(' ')[0]
                            formatted_date_str = datetime.strptime(postdate, '%Y-%m-%d').strftime('%Y%m%d')
                            if date != formatted_date_str:
                                continue
                            
                    post_treatment_file = filename
                    post_treatment = date
                    # post_treatment = '20220328'
                    patient_dict[patient_id + '_' + eye][post_treatment] = {}
                    if pre_treatment_file != '' and post_treatment_file != '':
                        pre_image = cv2.imread(self.output_image_path + '1/' + patient_id + '_' + eye + '_' + str(pre_treatment) + '.png')
                        post_image = cv2.imread(self.output_image_path + '1/' + patient_id + '_' + eye + '_' + str(post_treatment) + '.png')
                        
                        pre_image = cv2.resize(pre_image, (304, 304))
                        post_image = cv2.resize(post_image, (304, 304))
                        
                        pre_image_gray = cv2.cvtColor(pre_image, cv2.COLOR_BGR2GRAY)
                        post_image_gray = cv2.cvtColor(post_image, cv2.COLOR_BGR2GRAY)
                        
                        ##############################################
                        # pre_image_gray = cv2.bilateralFilter(pre_image_gray, 3, 10, 10)
                        # pre_image_gray = cv2.createCLAHE(clipLimit=0.7, tileGridSize=(8, 8)).apply(pre_image_gray)
                        # pre_image_gray = cv2.normalize(pre_image_gray, None, 0, 255, cv2.NORM_MINMAX)
                        
                        # post_image_gray = cv2.bilateralFilter(post_image_gray, 3, 10, 10)
                        # post_image_gray = cv2.createCLAHE(clipLimit=0.7, tileGridSize=(8, 8)).apply(post_image_gray)
                        # post_image_gray = cv2.normalize(post_image_gray, None, 0, 255, cv2.NORM_MINMAX)
                        ##############################################
                        
                        # show histogram
                        magnification = 1
                        pre_crop ,pre_center ,pre_radius = self.find_center(pre_image_gray,save_path = self.output_path + 'cut' + '/',save_name = patient_id + '_' + eye + '_' + post_treatment + 'pre')
                        post_crop ,post_center ,post_radius = self.find_center(post_image_gray,save_path = self.output_path + 'cut' + '/',save_name = patient_id + '_' + eye + '_' + post_treatment + 'post')
                        
                        # pre_image_gray = cv2.cvtColor(pre_image, cv2.COLOR_BGR2GRAY)
                        # post_image_gray = cv2.cvtColor(post_image, cv2.COLOR_BGR2GRAY)
                        # crop_img_gray = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
                        
                        # print(center,radius)
                        # print(center[1] - radius,center[1] + radius,center[0] - radius,center[0] + radius)

                        # pre_image2 = cv2.bilateralFilter(pre_image2, 5, 75, 75)
                        # crop_img = cv2.bilateralFilter(crop_img, 5, 75, 75)


                        # pre_image2 = cv2.equalizeHist(pre_image2)
                        # crop_img = cv2.equalizeHist(crop_img)
                        if post_crop is None:
                            continue
                        if len(post_crop.shape) == 2:
                            h , w  = post_crop.shape
                        else:
                            h , w ,c = post_crop.shape
                        pre_image2 = pre_image.copy()
                        
                        pre_image2 = cv2.resize(pre_image2, (int(304 * magnification), int(304 * magnification)))
                        H = self.LK(pre_crop,post_crop, distance=distance, method=feature, matcher=matcher)
                        # print(H)

                        if H is not None:
                            # 平移
                            H[0,2] -= (pre_center[0] - post_center[0])
                            H[1,2]-=(pre_center[1] - post_center[1])
                            translation = (H[0, 2], H[1, 2])
                            check = self.check_H_range(H)
                            
                        if H is  None or not check:
                            post_crop = cv2.resize(post_crop, (h, w ))
                            post_radius = post_radius // magnification
                            shift_x, shift_y = self.pointMatch(pre_image_gray,post_crop,post_center,post_radius, method = cv2.TM_CCOEFF_NORMED)
                            if shift_x > 304 // 2 or shift_x < -304 // 2 or shift_y > 304 // 2 or shift_y < -304 // 2:
                                H = None
                            else:
                                # H 3x3
                                H = np.array([[1.0, 0.0, shift_x], [0, 1.0, shift_y], [0.0, 0.0, 1.0]]).astype(np.float32)
                            
                        if H is  None or not self.check_H_range(H):   
                            H = np.array(np.eye(3))
                    
                        post_image = cv2.warpPerspective(post_image, H, (post_image.shape[1], post_image.shape[0]))
                        # post_image2= post_image.copy()
                        # post_image2[: : , : , 0] = 0
                        # post_image2[: : , : , 2] = 0
                        # visimg1= cv2.addWeighted(pre_image, 0.5, post_image2, 0.5, 0)
                        # visimg2 = cv2.addWeighted(post_image_original, 0.5, post_image2, 0.5, 0)
                    

                        # fig , ax =  plt.subplots(1,6,figsize=(20,20))
                        # ax[0].imshow(pre_image)
                        # ax[0].set_title('pre_image')
                        # ax[1].imshow(post_image_original)
                        # ax[1].set_title('post_image_original')
                        # ax[2].imshow(crop_img)
                        # ax[2].set_title('crop_img')
                        # ax[3].imshow(post_image)
                        # ax[3].set_title('post_image')
                        # ax[4].imshow(visimg1)
                        # ax[4].set_title('visimg1')
                        # ax[5].imshow(visimg2)
                        # ax[5].set_title('visimg2')
                        # plt.show()
                        
                        translation = (H[0, 2], H[1, 2])
                        rotation_rad = math.atan2(H[1, 0], H[0, 0])
                        rotation_angle= np.arctan2(H[1, 0], H[0, 0]) * 180 / np.pi
                        scale_x =  np.linalg.norm(H[:, 0])
                        scale_y = np.linalg.norm(H[:, 1])
                        scale_x = np.float64(scale_x)
                        scale_y = np.float64(scale_y)

                        for data in self.data_list:


                            if os.path.exists(self.output_image_path + data + '/' + patient_id + '_' + eye + '_' + pre_treatment + '.png'):
                                pre_image = cv2.imread(self.output_image_path + data + '/' + patient_id + '_' + eye + '_' + pre_treatment + '.png', cv2.IMREAD_GRAYSCALE)
                                pre_image = cv2.resize(pre_image, (304, 304))
                                
                                ##############################################
                                # pre_image = cv2.bilateralFilter(pre_image, 3, 10, 10)
                                # pre_image = cv2.createCLAHE(clipLimit=0.7, tileGridSize=(8, 8)).apply(pre_image)
                                ##############################################
                                pre_image = cv2.normalize(pre_image, None, 0, 255, cv2.NORM_MINMAX)

                        
                                cv2.imwrite(self.output_path + data + '/' + patient_id + '_' + eye + '_' + pre_treatment + '.png', pre_image)
                                if os.path.exists(self.output_image_path + data + '/' + patient_id + '_' + eye + '_' + post_treatment + '.png'):
                                    post_image = cv2.imread(self.output_image_path + data + '/' + patient_id + '_' + eye + '_' + post_treatment + '.png', cv2.IMREAD_GRAYSCALE)
                                    post_image = cv2.resize(post_image, (304, 304))
                                    
                                    ##############################################
                                    # post_image = cv2.bilateralFilter(post_image, 3, 10, 10)
                                    # post_image = cv2.createCLAHE(clipLimit=0.7, tileGridSize=(8, 8)).apply(post_image)
                                    ##############################################
                                    
                                    post_image = cv2.normalize(post_image, None, 0, 255, cv2.NORM_MINMAX)

                                    # print(self.output_path + data + '/' + patient_id + '_' + eye + '_' + post_treatment + '.png')
                                    cv2.imwrite(self.output_path + data + '/' + patient_id + '_' + eye + '_' + post_treatment + '.png', post_image)

                                    if H is not None:
                                        result = cv2.warpPerspective(post_image, H, (pre_image.shape[1], pre_image.shape[0]))
                                        
                                    else:
                                        result = post_image.copy()
                                        H = np.array(np.eye(3))
                                        
                                        # translation = (H[0, 2], H[1, 2])
                                        # rotation_rad = math.atan2(H[1, 0], H[0, 0])
                                        # rotation_angle= np.arctan2(H[1, 0], H[0, 0]) * 180 / np.pi
                                        # scale_x =  np.linalg.norm(H[:, 0]).astype(np.float32)
                                        # scale_y = np.linalg.norm(H[:, 1]).astype(np.float32)
                                        
                                        
                                        
                                    patient_dict[patient_id + '_' + eye][post_treatment]['translation'] = [translation[0], translation[1]]
                                    patient_dict[patient_id + '_' + eye][post_treatment]['rotation_rad'] = rotation_rad
                                    patient_dict[patient_id + '_' + eye][post_treatment]['rotation_angle'] = rotation_angle
                                    patient_dict[patient_id + '_' + eye][post_treatment]['scale'] = [scale_x, scale_y]
                                    # patient_dict[patient_id + '_' + eye][post_treatment]['center'] = [center[0], center[1]]
                                    # patient_dict[patient_id + '_' + eye][post_treatment]['radius'] = radius


                                    filled = result.copy()
                                    cv2.imwrite(self.output_path + data + '_move/' + patient_id + '_' + eye + '_' + post_treatment + '.png', filled)

                                    filled[filled == 0] = pre_image [filled == 0]

                                    # print(self.output_path + data + '/' + patient_id + '_' + eye + '_' + post_treatment + '.png')
                                    filled = cv2.resize(filled, (304, 304))

                                    cv2.imwrite(self.output_path + data + '/' + patient_id + '_' + eye + '_' + post_treatment + '.png', filled)

                                    # # add the two images together
                                    # vis_img = filled.copy()
                                    # vis_img[:,:,0] = 0
                                    # vis_img[:,:,2] = 0
                                    # vis = cv2.addWeighted(pre_image, 0.5, vis_img, 0.5, 0)
                                    # cv2.rectangle(vis, (int(center[0] - radius), int(center[1] - radius)), (int(center[0] + radius), int(center[1] + radius)), (255, 0, 255), 3)
                                
                                    # cv2.imwrite(self.output_path + data + '_vis/' + patient_id + '_' + eye + '_' + post_treatment + '.png', vis)
                        match_par = 'crop_'+ feature + '_' + matcher + '_' + str(distance) 
                        
                        
                        for layer in data_groups:
                            if 'CC' in layer:
                                output_label = 'CC'
                            else:
                                output_label = 'OR'
                                
                                    
                            if not os.path.exists(self.output_label_path + '/' + match_par +  '/' + output_label ):
                                os.makedirs(self.output_label_path + '/' +match_par  +  '/' + output_label )
                            if not os.path.exists(self.output_label_path + '/' +match_par  +  '/' + output_label +'_move/'):
                                os.makedirs(self.output_label_path + '/' + match_par + '/' + output_label +'_move/')
                                
                            mask_path = os.path.join(self.label_path + '_' + layer,mask_folder_name)

                            if os.path.exists(mask_path + '/' + patient_id + '_' + eye + '_' + pre_treatment + '.png'):
                                pre_label = cv2.imread(mask_path + '/' + patient_id + '_' + eye + '_' + pre_treatment + '.png')
                                pre_label = cv2.resize(pre_label, self.image_size)
                                # pre_label = cv2.normalize(pre_label, None, 0, 255, cv2.NORM_MINMAX)
                                
                                # print('pre_label',self.output_label_path+ match_par+ '/' + label + '/' + patient_id + '_' + eye + '_' + pre_treatment + '.png')
                                cv2.imwrite(self.output_label_path+ match_par+ '/' + output_label + '/' + patient_id + '_' + eye + '_' + pre_treatment + '.png', pre_label)
                                
                                if os.path.exists(mask_path + '/' + patient_id + '_' + eye + '_' + post_treatment + '.png'):
                                    post_label = cv2.imread(mask_path + '/' + patient_id + '_' + eye + '_' + post_treatment + '.png')
                                    post_label = cv2.resize(post_label, self.image_size)
                                    # post_label = cv2.normalize(post_label, None, 0, 255, cv2.NORM_MINMAX)
                                    height, width, channels = post_label.shape
                                    if H is None :
                                        result = post_label
                                    else:
                                        # print(H)
                                        translation = (H[0, 2], H[1, 2])
                                        rotation = math.atan2(H[1, 0], H[0, 0])
                                        rotation_angle= np.arctan2(H[1, 0], H[0, 0]) * 180 / np.pi
                                        scale = H[0, 0] / np.cos(rotation_angle)



                                        if scale < 0.5 or scale > 2:
                                            H = np.array(np.eye(3))
                                            result = post_label
                                        elif translation[0] < -304//2 or translation[0] > 304//2 or translation[1] < -304//2 or translation[1] > 304//2:
                                            H = np.array(np.eye(3))
                                            result = post_label

                                        elif rotation_angle < -60 or rotation_angle > 60:
                                            H = np.array(np.eye(3))
                                            result = post_label
                                        elif np.isnan(H).any():
                                            H = np.array(np.eye(3))
                                            result = post_label

                                        else:
                                            result = cv2.warpPerspective(post_label, H, (width, height))

                                    # print('post_treatment',self.output_label_path+ match_par+ '/' + label + '_move/' + patient_id + '_' + eye + '_' + pre_treatment + '.png')     
                                    cv2.imwrite(self.output_label_path+ match_par+ '/' + output_label + '_move/' + patient_id + '_' + eye + '_' + post_treatment + '.png', result)
                                    # result[result == 0] = pre_label[result == 0]
                                    # print('result',result.shape)
                                    # print('post_treatment',self.output_label_path+ match_par+ '/' + label + '/' + patient_id + '_' + eye + '_' + pre_treatment + '.png')     
                                    cv2.imwrite(self.output_label_path+ match_par+ '/' + output_label + '/' + patient_id + '_' + eye + '_' + post_treatment + '.png', result)

                        
                        
                        
                                    
                        # if  translation[0] !=0 or translation[1]!=0 or rotation_angle !=0 or scale_x != 1 or scale_y != 1:
                        #     pre_treatment_img = self.output_image_path+ '1/' + patient_id + '_' + eye + '_' + pre_treatment + '.png'
                        #     post_treatment_img = self.output_image_path + '1/' + patient_id + '_' + eye + '_' + post_treatment + '.png'
                        #     matching_img = self.output_path  + '/1_move/' + patient_id + '_' + eye + '_' + post_treatment + '.png'                              
                        #     evaluate = self.evaluates(pre_treatment_img,post_treatment_img,matching_img)

                        #     if evaluate is None:
                        #         continue
                        #     if evaluate['psnr'] == float('inf') or evaluate['matching_psnr'] == float('inf') or evaluate['psnr'] < 0 or evaluate['matching_psnr'] < 0:
                        #         continue
                        #     if evaluate['ssim'] == float('inf') or evaluate['matching_ssim'] == float('inf') or evaluate['ssim'] < 0 or evaluate['matching_ssim'] < 0:
                        #         continue
                        #     if evaluate['mse'] == float('inf') or evaluate['matching_mse'] == float('inf') or evaluate['mse'] < 0 or evaluate['matching_mse'] < 0:
                        #         continue
   
                           
                                
                                # patient_dict[patient_id + '_' + eye][post_treatment]['original'] = [mse,psnr,ssim]
                                # patient_dict[patient_id + '_' + eye][post_treatment]['matching'] = [matching_mse,matching_psnr,matching_ssim]                                 
        # delete empty patient
        for key in list(patient_dict.keys()):
            if len(patient_dict[key]) == 0:
                del patient_dict[key]
                
                        
                
        print('------------------')
        print(len(patient_dict))       
        return patient_dict

    def all_evaluate(self,match_path):

        patient = {}
        image_folder = match_path+ '/1/'
        filenames = sorted(os.listdir(image_folder))

        best_case = dict()
        worst_case = dict()
        best_differece_ssim = 0
        best_differece_psnr = 0
        worst_differece_ssim = 100000000000000000000
        worst_differece_psnr = 100000000000000000000
        avg_mse = 0
        avg_psnr = 0
        avg_ssim = 0
        mse_list = []
        psnr_list = []
        ssim_list = []
        ncc_list = []
        nmi_list = []
        corr_list = []
        matching_mse_list = []
        matching_psnr_list = []
        matching_ssim_list = []
        matching_ncc_list = []
        matching_nmi_list = []
        matching_corr_list = []
        pre_treatment_file = ''
        post_treatment_file = ''
        
        for filename in filenames: 
            if filename.endswith(".png"):
                patient_id, eye, date = filename.split('.png')[0].split('_')
                if patient_id + '_' + eye not in patient :
                    patient[patient_id + '_' + eye] = {}
                    post_treatment = ''
                    pre_treatment = date
                    pre_treatment_file = filename
                    # patient[patient_id + '_' + eye]['pre_treatment'] = date
                else :
                    patient[patient_id + '_' + eye][date] = {}
                    post_treatment =  date 
                    post_treatment_file = filename

                    if pre_treatment_file != '' and post_treatment_file != '':
                        patient[patient_id + '_' + eye][date]['pre_treatment'] = pre_treatment
                        pre_treatment_img = self.output_image_path+ '1/' + patient_id + '_' + eye + '_' + pre_treatment + '.png'
                        post_treatment_img = self.output_image_path + '1/' + patient_id + '_' + eye + '_' + post_treatment + '.png'
                        matching_img = match_path  + '/1_move/' + patient_id + '_' + eye + '_' + post_treatment + '.png'
                        evaluate = self.evaluates(pre_treatment_img,post_treatment_img,matching_img)

                        if evaluate['psnr'] == float('inf') or evaluate['matching_psnr'] == float('inf') or evaluate['psnr'] < 0 or evaluate['matching_psnr'] < 0:
                            continue
                        if evaluate['ssim'] == float('inf') or evaluate['matching_ssim'] == float('inf') or evaluate['ssim'] < 0 or evaluate['matching_ssim'] < 0:
                            continue
                        if evaluate['mse'] == float('inf') or evaluate['matching_mse'] == float('inf') or evaluate['mse'] < 0 or evaluate['matching_mse'] < 0:
                            continue
                        mse = evaluate['mse']
                        psnr = evaluate['psnr']
                        ssim = evaluate['ssim']
                        ncc = evaluate['ncc']
                        nmi = evaluate['nmi']
                        corr = evaluate['corr']
                        
                        matching_mse = evaluate['matching_mse']
                        matching_psnr = evaluate['matching_psnr']
                        matching_ssim = evaluate['matching_ssim']
                        matching_ncc = evaluate['matching_ncc']
                        matching_nmi = evaluate['matching_nmi']
                        matching_corr = evaluate['matching_corr']
                        
                        patient[patient_id + '_' + eye][date]['original'] = dict()
                        patient[patient_id + '_' + eye][date]['matching'] = dict()  
                        
                        patient[patient_id + '_' + eye][date]['original']['mse'] = mse
                        patient[patient_id + '_' + eye][date]['original']['psnr'] = psnr
                        patient[patient_id + '_' + eye][date]['original']['ssim'] = ssim
                        patient[patient_id + '_' + eye][date]['original']['ncc'] = ncc
                        patient[patient_id + '_' + eye][date]['original']['nmi'] = nmi
                        patient[patient_id + '_' + eye][date]['original']['corr'] = corr
                        
                        patient[patient_id + '_' + eye][date]['matching']['mse'] = matching_mse
                        patient[patient_id + '_' + eye][date]['matching']['psnr'] = matching_psnr
                        patient[patient_id + '_' + eye][date]['matching']['ssim'] = matching_ssim
                        patient[patient_id + '_' + eye][date]['matching']['ncc'] = matching_ncc
                        patient[patient_id + '_' + eye][date]['matching']['nmi'] = matching_nmi
                        patient[patient_id + '_' + eye][date]['matching']['corr'] = matching_corr
                        
                        
                        mse_list.append(mse)
                        psnr_list.append(psnr)
                        ssim_list.append(ssim)
                        ncc_list.append(ncc)
                        nmi_list.append(nmi)
                        corr_list.append(corr)
                        matching_mse_list.append(matching_mse)
                        matching_psnr_list.append(matching_psnr)
                        matching_ssim_list.append(matching_ssim)
                        matching_ncc_list.append(matching_ncc)
                        matching_nmi_list.append(matching_nmi)
                        matching_corr_list.append(matching_corr)


                        # if matching_ssim > ssim :
                        #     patient[patient_id + '_' + eye][date]['ssim'] = 'better'
                        # elif matching_ssim == ssim :
                        #     patient[patient_id + '_' + eye][date]['ssim'] = 'same'
                        # else :
                        #     patient[patient_id + '_' + eye][date]['ssim'] = 'worse'

                        # if matching_psnr > psnr :
                        #     patient[patient_id + '_' + eye][date]['psnr'] = 'better'
                        # elif matching_psnr == psnr :
                        #     patient[patient_id + '_' + eye][date]['psnr'] = 'same'
                        # else :
                        #     patient[patient_id + '_' + eye][date]['psnr'] = 'worse'

                        # if matching_mse < mse :
                        #     patient[patient_id + '_' + eye][date]['mse'] = 'better'
                        # elif matching_mse == mse :
                        #     patient[patient_id + '_' + eye][date]['mse'] = 'same'
                        # else :
                        #     patient[patient_id + '_' + eye][date]['mse'] = 'worse'
                        
                        

                        if matching_ssim - ssim > best_differece_ssim :
                            best_differece_ssim = matching_ssim - ssim
                            best_case['patient'] = [ patient_id,eye,post_treatment]
                            best_case ['psnr'] = psnr
                            best_case ['ssim'] = ssim
                            best_case ['mse'] = mse
                            best_case ['matching_psnr'] = matching_psnr
                            best_case ['matching_ssim'] = matching_ssim
                            best_case ['matching_mse'] = matching_mse
                            
                        if matching_psnr - psnr > best_differece_psnr :
                            best_differece_psnr = matching_psnr - psnr
                            best_case['patient'] = [ patient_id,eye,post_treatment]
                            best_case ['psnr'] = psnr
                            best_case ['ssim'] = ssim
                            best_case ['mse'] = mse
                            best_case ['matching_psnr'] = matching_psnr
                            best_case ['matching_ssim'] = matching_ssim
                            best_case ['matching_mse'] = matching_mse
                            

                        if matching_ssim - ssim < worst_differece_ssim :
                            worst_differece_ssim = matching_ssim - ssim
                            worst_case['patient'] = [ patient_id,eye,post_treatment]
                            worst_case ['psnr'] = psnr
                            worst_case ['ssim'] = ssim
                            worst_case ['mse'] = mse
                            worst_case ['matching_psnr'] = matching_psnr
                            worst_case ['matching_ssim'] = matching_ssim
                            worst_case ['matching_mse'] = matching_mse
                            
                        if matching_psnr - psnr < worst_differece_psnr :
                            worst_differece_psnr = matching_psnr - psnr
                            worst_case['patient'] = [ patient_id,eye,post_treatment]
                            worst_case ['psnr'] = psnr
                            worst_case ['ssim'] = ssim
                            worst_case ['mse'] = mse
                            worst_case ['matching_psnr'] = matching_psnr
                            worst_case ['matching_ssim'] = matching_ssim
                            worst_case ['matching_mse'] = matching_mse
                            
        matching_avg_mse = round(sum(matching_mse_list)/len(matching_mse_list),2)
        matching_avg_psnr = round(sum(matching_psnr_list)/len(matching_psnr_list),2)
        matching_avg_ssim = round(sum(matching_ssim_list)/len(matching_ssim_list),2)
        matching_avg_ncc = round(sum(matching_ncc_list)/len(matching_ncc_list),2)
        matching_avg_nmi = round(sum(matching_nmi_list)/len(matching_nmi_list),2)
        matching_avg_corr = round(sum(matching_corr_list)/len(matching_corr_list),2)
        matching_mse_std = round(np.std(matching_mse_list, ddof=1),2)
        matching_psnr_std = round(np.std(matching_psnr_list, ddof=1),2)
        matching_ssim_std = round(np.std(matching_ssim_list, ddof=1),2)
        matching_ncc_std = round(np.std(matching_ncc_list, ddof=1),2)
        matching_nmi_std = round(np.std(matching_nmi_list, ddof=1),2)
        matching_corr_std = round(np.std(matching_corr_list, ddof=1),2)

        avg_mse = round(sum(mse_list)/len(mse_list),2)
        avg_psnr = round(sum(psnr_list)/len(psnr_list),2)
        avg_ssim = round(sum(ssim_list)/len(ssim_list),2)
        avg_ncc = round(sum(ncc_list)/len(ncc_list),2)
        avg_nmi = round(sum(nmi_list)/len(nmi_list),2)
        avg_corr = round(sum(corr_list)/len(corr_list),2)
        mse_std = round(np.std(mse_list, ddof=1),2)
        psnr_std = round(np.std(psnr_list, ddof=1),2)
        ssim_std = round(np.std(ssim_list, ddof=1),2)
        ncc_std = round(np.std(ncc_list, ddof=1),2)
        nmi_std = round(np.std(nmi_list, ddof=1),2)
        corr_std = round(np.std(corr_list, ddof=1),2)
            
        patient['avg'] = {}
        patient['avg']['original'] = {}
        patient['avg']['original']['mse'] =avg_mse
        patient['avg']['original']['psnr'] = avg_psnr
        patient['avg']['original']['ssim'] = avg_ssim
        patient['avg']['original']['ncc'] = avg_ncc
        patient['avg']['original']['nmi'] = avg_nmi
        patient['avg']['original']['corr'] = avg_corr
        patient['avg']['matching'] = {}
        patient['avg']['matching']['mse'] = matching_avg_mse
        patient['avg']['matching']['psnr'] = matching_avg_psnr
        patient['avg']['matching']['ssim'] = matching_avg_ssim
        patient['avg']['matching']['ncc'] = matching_avg_ncc
        patient['avg']['matching']['nmi'] = matching_avg_nmi
        patient['avg']['matching']['corr'] = matching_avg_corr
        
        patient['std'] = {}
        patient['std']['original'] = {}
        patient['std']['original']['mse'] = mse_std
        patient['std']['original']['psnr'] = psnr_std
        patient['std']['original']['ssim'] = ssim_std
        patient['std']['original']['ncc'] = ncc_std
        patient['std']['original']['nmi'] = nmi_std
        patient['std']['original']['corr'] = corr_std
        
        patient['std']['matching'] = {}
        patient['std']['matching']['mse'] = matching_mse_std
        patient['std']['matching']['psnr'] = matching_psnr_std
        patient['std']['matching']['ssim'] = matching_ssim_std
        patient['std']['matching']['ncc'] = matching_ncc_std
        patient['std']['matching']['nmi'] = matching_nmi_std
        patient['std']['matching']['corr'] = matching_corr_std
        
        

        if best_case != {} :
            patient['best_case'] = best_case

        if worst_case != {} :
            patient['worst_case'] = worst_case

        return patient

    def relabel(self,img,mask,mathod = 'connectedComponent',min_area = 50):
        if img.shape.__len__() == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
        if mask.shape.__len__() == 3:
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
            

        label = mask.copy()  
        # if mathod == 'threshold':
        #     threshold_label = self.otsuthreshold(img,mask) 
        if mathod == 'connectedComponent':
            ret, binary_image = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
            
            label[binary_image == 0] = 0   
            # 刪除小面積
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(label, connectivity=8)
            for i in range(1, num_labels):
                if stats[i, 4] < min_area:
                    label[labels == i] = 0
            return label 
        
        return label    

# Convert float32 to float recursively
def convert_float32_to_float(obj):
    if isinstance(obj, np.float32):
        return float(obj)
    elif isinstance(obj, dict):
        return {key: convert_float32_to_float(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_float32_to_float(item) for item in obj]
    else:
        return obj

            


def get_data_from_txt_file(filename):
    # get txt
    data = []
    with open(filename, 'r') as file:
        for line in file:
            data.append(line.strip())
    return data






if __name__ == '__main__':
    # pre_treatment_file = "..\\..\\Data\\PCV_1120\\ALL\\1\\08707452_L_20161130.png"
    # post_treatment_file = "..\\..\\Data\\PCV_1120\\ALL\\1\\08707452_L_20170118.png"
    
    date = '20240524'
    disease = 'PCV'
    PATH_DATA = '../../Data/' 
    PATH_BASE = PATH_DATA  + disease + '_' + date + '/'
    data_groups = ['CC']
    label_path = PATH_BASE  + '/' + disease + '_' + date + '_connectedComponent_bil31010_clah0712_concate34OCT'
    PATH_IMAGE = PATH_DATA + 'OCTA/' 
    output_image_path = PATH_BASE + 'ALL/'
    image_path = PATH_BASE + 'ALL/MATCH/' 
    label_path_name = [disease + '_' + date + '_connectedComponent']
    output_label_path = output_image_path + 'MATCH_LABEL/' 
    distances = [0.8]
    features = ['KAZE']#,'KAZE','AKAZE','ORB','BRISK' ,'FREAK','BRIEF'
    matchers = ['BF']# ,'FLANN'
    # patient_list = get_data_from_txt_file('PCV.txt')
    setFolder('./record/'+ disease + '_' + date + '/') 
    
    for distance in distances:
        for feature in features:
            for matcher in matchers:
                # print(image_path + 'crop' + '_' +  feature + '_' + matcher + '_' + str(distance))

                find = finding(label_path,image_path,label_path_name,data_groups,output_label_path,output_image_path,features,matchers,distance,file_name='../../Data/打針資料.xlsx')
                
                find_dict = find.feature(feature,matcher,distance,data_groups)
                find_dict = convert_float32_to_float(find_dict)
                json_file = './record/'+ disease + '_' + date + '/'+ 'crop_'+feature + '_' + matcher + '_' + str(distance) + '_align.json'
                tools.write_to_json_file(json_file, find_dict)
                
                # eval = find.all_evaluate(image_path + 'crop' + '_' +  feature + '_' + matcher + '_' + str(distance))
                # json_file2 = './record/'+ disease + '_' + date + '/'+ 'crop_'+feature + '_' + matcher + '_' + str(distance) + '_evals.json'
                # tools.write_to_json_file(json_file2, eval)
                eval = find.all_evaluate(image_path + 'crop' + '_' +  feature + '_' + matcher + '_' + str(distance))
                csv_file = './record/'+ disease + '_' + date + '/'+ feature + '_' + matcher + '_' + str(distance) + '_evals.csv'
                # tools.write_to_json_file(json_file2, eval)
                cases = 0
                # save evaluation to csv
                # print('eval',eval)
                #[mse,psnr,ssim,ncc,nmi,corr]
                with open(csv_file, 'w' , newline='') as f:
                    writer = csv.writer(f)  
                    writer.writerow(['patient', 'eye','post_treatment', 'pre_treatment', 'mse', 'psnr', 'ssim','ncc','nmi','corr','matching_mse', 'matching_psnr', 'matching_ssim','matching_ncc','matching_nmi','matching_corr'])
                    for patient_eye in eval:
                        if "best_case" not in patient_eye and "worst_case" not in patient_eye and "avg" not in patient_eye and "std" not in patient_eye:
                            patient, eye = patient_eye.split('_')
                            
                            for post_treatment in eval[patient_eye]:
                                if 'pre_treatment' in eval[patient_eye][post_treatment]:
                                    cases += 1
                                    
                                    writer.writerow([patient,eye, 
                                                    post_treatment,
                                                    eval[patient_eye][post_treatment]['pre_treatment'],
                                                     eval[patient_eye][post_treatment]['original']['mse'],
                                                     eval[patient_eye][post_treatment]['original']['psnr'],
                                                     eval[patient_eye][post_treatment]['original']['ssim'],
                                                     eval[patient_eye][post_treatment]['original']['ncc'],
                                                     eval[patient_eye][post_treatment]['original']['nmi'],
                                                     eval[patient_eye][post_treatment]['original']['corr'],
                                                     eval[patient_eye][post_treatment]['matching']['mse'],
                                                     eval[patient_eye][post_treatment]['matching']['psnr'],
                                                     eval[patient_eye][post_treatment]['matching']['ssim'],
                                                     eval[patient_eye][post_treatment]['matching']['ncc'],
                                                     eval[patient_eye][post_treatment]['matching']['nmi'],
                                                     eval[patient_eye][post_treatment]['matching']['corr']])
                                                     
                                        

                                    
                    
                        
                        
                        
                        
               
                        
                aligment_file = './record/'+ disease + '_' + date + '/' + 'evaluations.csv'
                if not os.path.exists(aligment_file):
                    with open(aligment_file, 'w', newline='') as f:
                        csv_writer = csv.writer(f)  
                        csv_writer.writerow(['feature', 'cases','avg_mse','std_mse', 'avg_psnr', 'std_psnr', 'avg_ssim', 'std_ssim','avg_ncc','std_ncc','avg_nmi','std_nmi','avg_corr','std_corr','avg_matching_mse','std_matching_mse', 'avg_matching_psnr', 'std_matching_psnr', 'avg_matching_ssim', 'std_matching_ssim','avg_matching_ncc','std_matching_ncc','avg_matching_nmi','std_matching_nmi','avg_matching_corr','std_matching_corr'])
                with open(aligment_file, 'a', newline='') as f:
                    csv_writer = csv.writer(f)
                    csv_writer.writerow([feature + '_' + matcher + '_' + str(distance),
                                            cases ,
                                            eval["avg"]['original']['mse'],
                                            eval["std"]['original']['mse'],
                                            eval["avg"]['original']['psnr'],
                                            eval["std"]['original']['psnr'],
                                            eval["avg"]['original']['ssim'],
                                            eval["std"]['original']['ssim'],
                                            eval["avg"]['original']['ncc'],
                                            eval["std"]['original']['ncc'],
                                            eval["avg"]['original']['nmi'],
                                            eval["std"]['original']['nmi'],
                                            eval["avg"]['original']['corr'],
                                            eval["std"]['original']['corr'],
                                            eval["avg"]['matching']['mse'],
                                            eval["std"]['matching']['mse'],
                                            eval["avg"]['matching']['psnr'],
                                            eval["std"]['matching']['psnr'],
                                            eval["avg"]['matching']['ssim'],
                                            eval["std"]['matching']['ssim'],
                                            eval["avg"]['matching']['ncc'],
                                            eval["std"]['matching']['ncc'],
                                            eval["avg"]['matching']['nmi'],
                                            eval["std"]['matching']['nmi'],
                                            eval["avg"]['matching']['corr'],
                                            eval["std"]['matching']['corr']
                                            ])
                    


