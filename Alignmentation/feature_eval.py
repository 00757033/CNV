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
import pathlib as pl
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
    
    
            
# 尋找OCTA 影像的黃斑中心
class finding():
    def __init__(self,label_path,image_path,output_label_path,output_image_path,methods,matchers,distance):
        self.label_path = label_path
        self.image_path = image_path
        self.distances = distance
        self.methods = methods 
        self.matchers = matchers
        self.output_image_path = output_image_path
        self.output_label_path = output_label_path
        self.image_size= (304, 304)
        self.data_list = ['1','2', '3', '4', '1_OCT', '2_OCT', '3_OCT', '4_OCT']
        self.label_list = ['CC']
        self.methods_template = [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED, cv2.TM_CCORR , cv2.TM_CCORR_NORMED, cv2.TM_CCOEFF, cv2.TM_CCOEFF_NORMED]
        self.method_template_name = ['TM_SQDIFF', 'TM_SQDIFF_NORMED', 'TM_CCORR' , 'TM_CCORR_NORMED', 'TM_CCOEFF', 'TM_CCOEFF_NORMED']


    def evaluates(self, pre_treatment_img, post_treatment_img,matching_img):
        cmp_pre_treatment_img = cv2.imread(pre_treatment_img , cv2.IMREAD_GRAYSCALE)
        cmp_post_treatment_img = cv2.imread(post_treatment_img, cv2.IMREAD_GRAYSCALE)
        cmp_matching_img = cv2.imread(matching_img, cv2.IMREAD_GRAYSCALE)
        
                          
        
        if cmp_pre_treatment_img is None or cmp_post_treatment_img is None or cmp_matching_img is None:
            return -1 , -1 , -1,-1 , -1 , -1

        if cmp_pre_treatment_img.shape != cmp_post_treatment_img.shape and cmp_post_treatment_img.shape != cmp_matching_img.shape : 
            return -1 , -1 , -1,-1 , -1 , -1
        
        
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
        
        cmp_pre_treatment_img[cmp_matching_img == 0] = 0
        cmp_post_treatment_img[cmp_matching_img == 0] = 0
        matching_mse = mean_squared_error_ignore_zeros(cmp_pre_treatment_img, cmp_matching_img,cmp_matching_img)
        matching_psnr = psnr_ignore_zeros(cmp_pre_treatment_img, cmp_matching_img,cmp_matching_img)
        matching_ssim = ssim_ignore_zeros(cmp_pre_treatment_img, cmp_matching_img,cmp_matching_img)
        matching_ssim = matching_ssim * 100


        
        return mse,psnr,ssim ,matching_mse,matching_psnr,matching_ssim



        
        
        
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
        
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)
        
        image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)
        imageequal = cv2.GaussianBlur(image, (3, 3), 0)
        imageequal = cv2.equalizeHist(imageequal)

        # top hat
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 40))
        tophat = cv2.morphologyEx(imageequal, cv2.MORPH_TOPHAT, kernel)
        # black hat
        blackhat = cv2.morphologyEx(image, cv2.MORPH_BLACKHAT, kernel)
        # add and subtract between morphological gradient and image
        image = tophat
        image = cv2.add(image, tophat)
        image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)

        return image


    def find_center(self,image):
        image2 = image.copy()
        image2 = self.preprocess(image2)
        
        # rst,image2 = cv2.threshold(image2, 0, 255,  cv2.THRESH_BINARY  + cv2.THRESH_OTSU)
        
        # 二值化 32 255
        rst,image2 = cv2.threshold(image2, 64, 255,  cv2.THRESH_BINARY)
        
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
        

        crop_img = image[int(center[1] - max_radius)-t:int(center[1] + max_radius)+t, int(center[0] - max_radius)-t:int(center[0] + max_radius)+t]
        # show the image wait 3 seconds and destroy
        return  crop_img , center , max_radius + t

    def LK(self,img1, img2,center,radius,  distance=0.9, method='KAZE',matcher='BF'):
        # Initiate SIFT detector
        if method == 'SIFT':
            sift = cv2.SIFT_create()
        elif method == 'KAZE':
            sift = cv2.KAZE_create()
        elif method == 'AKAZE':
            sift = cv2.AKAZE_create()
        elif method == 'ORB':
            sift = cv2.ORB_create()
        elif method == 'BRISK':
            sift = cv2.BRISK_create()
        elif method == 'BRIEF':
            detector = cv2.FastFeatureDetector_create()
            brief = cv2.xfeatures2d.BriefDescriptorExtractor_create( )

        elif method == 'FREAK':
            detector = cv2.FastFeatureDetector_create()
            freak = cv2.xfeatures2d.FREAK_create()

        if method == 'FREAK':
            kp1 = detector.detect(img1)
            kp2 = detector.detect(img2)
            kp1, des1 = freak.compute(img1, kp1)
            kp2, des2 = freak.compute(img2, kp2)
        elif method == 'BRIEF':
            kp1 = detector.detect(img1)
            kp2 = detector.detect(img2)
            kp1, des1 = brief.compute(img1, kp1)
            kp2, des2 = brief.compute(img2, kp2)
        elif method == 'ORB':
            kp1, des1 = sift.detectAndCompute(img1, None)
            kp2, des2 = sift.detectAndCompute(img2, None)

        else:
            kp1 = sift.detect(img1, None)
            kp2 = sift.detect(img2, None)

            kp1, des1 = sift.compute(img1, kp1)
            kp2, des2 = sift.compute(img2, kp2)


        # img1_draw = cv2.drawKeypoints(img1, kp1, img1, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        # img2_draw = cv2.drawKeypoints(img2, kp2, img2, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        # Convert to numpy arrays
        if des1 is None or des2 is None:
            H = np.array(np.eye(3))
            return None


        des1 = des1.astype(np.float32)
        des2 = des2.astype(np.float32)

        # Matching method
        if method == 'SIFT'  or method == 'BRISK' or method == 'BRIEF':
            if  matcher == 'BF':
                bf = cv2.BFMatcher( cv2.NORM_L2)
                matches = bf.knnMatch(des1, des2, k=2)

            elif matcher == 'FLANN':
                FLANN_INDEX_KDTREE = 1
                index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
                search_params = dict(checks=1000)

                flann = cv2.FlannBasedMatcher(index_params, search_params)
                matches = flann.knnMatch(des1, des2, k=2)

        elif method == 'ORB':
            if  matcher == 'BF':
                bf = cv2.BFMatcher()
                matches = bf.knnMatch(des1, des2, k=2)

            elif matcher == 'FLANN':
                FLANN_INDEX_LSH = 6
                index_params= dict(algorithm = FLANN_INDEX_LSH,
                                table_number = 6, # 12
                                key_size = 12,     # 20
                                multi_probe_level = 1)

                search_params = dict(checks=60)
                # Make sure descriptors are 2D arrays
                des1 = np.array(des1).astype(np.uint8)
                des2 = np.array(des2).astype(np.uint8)


                flann = cv2.FlannBasedMatcher(index_params, search_params)
                matches = flann.knnMatch(des1, des2, k=2)

        elif method == 'KAZE' or method == 'AKAZE' or method == 'SURF' or method == 'FREAK':
            if  matcher == 'BF':
                bf = cv2.BFMatcher( cv2.NORM_L2)
                matches = bf.knnMatch(des1, des2, k=2)

            elif matcher == 'FLANN':
                FLANN_INDEX_KDTREE = 1
                index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
                search_params = dict(checks=60)

                flann = cv2.FlannBasedMatcher(index_params, search_params)
                matches = flann.knnMatch(des1, des2, k=2)
        if not matches:

            return None
        if matcher == 'BF': 
            # Need to draw only good matches, so create a mask
            matches = sorted(matches, key=lambda x: x[0].distance)
            if matches:
                min_dist = matches[0][0].distance

                # Apply ratio test
                good = []
                pts1 = []
                pts2 = []
                for i, mn in enumerate(matches):
                    if len(mn)== 2 :
                        m,n = mn
                        if m.distance < distance* n.distance:
                            good.append([m])
                            pts2.append(kp2[m.trainIdx].pt)
                            pts1.append(kp1[m.queryIdx].pt)

                        if m.distance > 1.5 * min_dist:
                            break
            # Draw matches
            img3 = cv2.drawMatchesKnn(img1 , kp1, img2, kp2, good, None, flags=2)
            
        elif matcher == 'FLANN':
            # Need to draw only good matches, so create a mask
            matchesMask = [[0, 0] for i in range(len(matches))]
            pts1 = []
            pts2 = []
            if matches and matches[0]:
                # matches = sorted(matches, key=lambda x: x[0].distance)
                min_dist = matches[0][0].distance
                # ratio test as per Lowe's paper
                
                for i, match in enumerate(matches):
                    if len(match) != 2 or match[0] is None or match[1] is None:
                        continue
                    if len(match) == 2:
                        m, n = match
                    if m.distance < distance * n.distance:
                        matchesMask[i] = [1, 0]
                        pts2.append(kp2[m.trainIdx].pt)
                        pts1.append(kp1[m.queryIdx].pt)
                draw_params = dict(matchColor = (0,255,0),
                    singlePointColor = (255,0,0),
                    matchesMask = matchesMask,
                    flags = 0)
                img3 = cv2.drawMatchesKnn(img1,kp1,img2,kp2,matches,None,**draw_params)


        # plt.imshow(img3)
        # plt.axis('off')
        # plt.title(method)
        # plt.show()

        # 計算img1 需要旋轉 縮放 平移矩陣 原本img1的實際中心點為center
        pts1 = np.array(pts1)
        pts2 = np.array(pts2)

        if len(pts1) == 0 or len(pts2) == 0:
            H = np.array(np.eye(3))

            return  None
            # Find homography
        if len(pts1) < 4 or len(pts2) < 4:
            H = np.array(np.eye(3))
            return None

        H, _ = cv2.findHomography(pts2, pts1, cv2.RANSAC, 5.0)

        return H

    def evaluate(self,image ,original_image):
        image = cv2.imread(image, cv2.IMREAD_GRAYSCALE)
        original_image = cv2.imread(original_image, cv2.IMREAD_GRAYSCALE)
        if image is None or original_image is None:
            return 0 , 0 , 1000000

        if image.shape != original_image.shape:
            return 0 , 0 , 1000000
            
        
        mse = mean_squared_error(image, original_image)
        psnr = peak_signal_noise_ratio(image, original_image)
        ssim = structural_similarity(image, original_image)
        ssim = ssim * 100

        return mse,psnr,ssim

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
                print('translation out of range')
                return False
            if rotation_angle > 30 or rotation_angle < -30:
                print('rotation out of range')
                return False
            if scale_x > 1.5 or scale_x < 0.5 or scale_y > 1.5 or scale_y < 0.5:
                print('scale out of range')
                return False
            return True
        else:
            return False

    def getPoints(self,img, template, method=cv2.TM_CCOEFF_NORMED):
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
    
    def feature(self, feature, matcher, distance):
        patient_dict = {}
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

                if patient_id + '_' + eye not in patient_dict :
                    patient_dict[patient_id + '_' + eye] = {}
                    pre_treatment_file = filename
                    pre_treatment = date
                    # pre_treatment = '20211013'

                else:
                    post_treatment_file = filename
                    post_treatment = date
                    # post_treatment = '20220328'
                    patient_dict[patient_id + '_' + eye][post_treatment] = {}
                    if pre_treatment_file != '' and post_treatment_file != '':
                        pre_image = cv2.imread(self.output_image_path + '1/' + patient_id + '_' + eye + '_' + str(pre_treatment) + '.png')
                        post_image = cv2.imread(self.output_image_path + '1/' + patient_id + '_' + eye + '_' + str(post_treatment) + '.png')
                        

                        
                        pre_image = cv2.resize(pre_image, (304, 304))
                        post_image = cv2.resize(post_image, (304, 304))

                        pre_image = cv2.normalize(pre_image, None, 0, 255, cv2.NORM_MINMAX)
                        post_image = cv2.normalize(post_image, None, 0, 255, cv2.NORM_MINMAX)
                        
                        pre_image_original = pre_image.copy()
                        post_image_original = post_image.copy()
                        # show histogram
                        
                        crop_img ,center,radius = self.find_center(post_image)
                        
                        
                        # pre_image_gray = cv2.cvtColor(pre_image, cv2.COLOR_BGR2GRAY)
                        # post_image_gray = cv2.cvtColor(post_image, cv2.COLOR_BGR2GRAY)
                        # crop_img_gray = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
                        
                        # print(center,radius)
                        # print(center[1] - radius,center[1] + radius,center[0] - radius,center[0] + radius)

                        # pre_image2 = cv2.bilateralFilter(pre_image2, 5, 75, 75)
                        # crop_img = cv2.bilateralFilter(crop_img, 5, 75, 75)


                        # pre_image2 = cv2.equalizeHist(pre_image2)
                        # crop_img = cv2.equalizeHist(crop_img)
                        

                        H = self.LK(pre_image,crop_img,center=(center[0],center[1]), radius = int(radius), distance=distance, method=feature, matcher=matcher)
                        

                        if H is not None:
                            # 平移
                            H[0,2] -= (center[0] - int(radius))
                            H[1,2]-=(center[1] - int(radius))
                            translation = (H[0, 2], H[1, 2])
                            
                        if H is  None or not self.check_H_range(H):
                            shift_x, shift_y = self.pointMatch(pre_image,crop_img,center,radius, method = cv2.TM_CCOEFF_NORMED)
                            if shift_x > 304 // 2 or shift_x < -304 // 2 or shift_y > 304 // 2 or shift_y < -304 // 2:
                                H = None
                            else:
                                # H 3x3
                                H = np.array([[1.0, 0.0, shift_x], [0, 1.0, shift_y], [0.0, 0.0, 1.0]]).astype(np.float32)
                            
                        if H is  None or not self.check_H_range(H):   
                            H = np.array(np.eye(3))
                    
                        # post_image = cv2.warpPerspective(post_image, H, (post_image.shape[1], post_image.shape[0]))
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
                                pre_image = cv2.imread(self.output_image_path + data + '/' + patient_id + '_' + eye + '_' + pre_treatment + '.png')
                                pre_image = cv2.resize(pre_image, (304, 304))
                                pre_image = cv2.normalize(pre_image, None, 0, 255, cv2.NORM_MINMAX)

                        
                                cv2.imwrite(self.output_path + data + '/' + patient_id + '_' + eye + '_' + pre_treatment + '.png', pre_image)
                                if os.path.exists(self.output_image_path + data + '/' + patient_id + '_' + eye + '_' + post_treatment + '.png'):
                                    post_image = cv2.imread(self.output_image_path + data + '/' + patient_id + '_' + eye + '_' + post_treatment + '.png')
                                    post_image = cv2.resize(post_image, (304, 304))
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
                                    patient_dict[patient_id + '_' + eye][post_treatment]['center'] = [center[0], center[1]]
                                    patient_dict[patient_id + '_' + eye][post_treatment]['radius'] = radius


                                    filled = result.copy()
                                    cv2.imwrite(self.output_path + data + '_move/' + patient_id + '_' + eye + '_' + post_treatment + '.png', filled)

                                    filled[filled == 0] = pre_image [filled == 0]

                                    # print(self.output_path + data + '/' + patient_id + '_' + eye + '_' + post_treatment + '.png')
                                    filled = cv2.resize(filled, (304, 304))

                                    cv2.imwrite(self.output_path + data + '/' + patient_id + '_' + eye + '_' + post_treatment + '.png', filled)

                                    # add the two images together
                                    vis_img = filled.copy()
                                    vis_img[:,:,0] = 0
                                    vis_img[:,:,2] = 0
                                    # vis = cv2.addWeighted(pre_image, 0.5, vis_img, 0.5, 0)
                                    # cv2.rectangle(vis, (int(center[0] - radius), int(center[1] - radius)), (int(center[0] + radius), int(center[1] + radius)), (255, 0, 255), 3)
                                
                                    # cv2.imwrite(self.output_path + data + '_vis/' + patient_id + '_' + eye + '_' + post_treatment + '.png', vis)
                        match_par = 'crop_'+ feature + '_' + matcher + '_' + str(distance) 
                        for label in self.label_list:
                                    
                                    
                            if not os.path.exists(self.output_label_path + '/' + match_par +  '/' + label ):
                                os.makedirs(self.output_label_path + '/' +match_par  +  '/' + label )
                            if not os.path.exists(self.output_label_path + '/' +match_par  +  '/' + label +'_move/'):
                                os.makedirs(self.output_label_path + '/' + match_par + '/' + label +'_move/')
                                
                            mask_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(self.image_path))),label,'masks')
                            # print('------',mask_path + '/'+ patient_id + '_' + eye + '_' + pre_treatment + '.png')
                            if os.path.exists(mask_path + '/' + patient_id + '_' + eye + '_' + pre_treatment + '.png'):
                                # print('------',mask_path + '/'+ patient_id + '_' + eye + '_' + pre_treatment + '.png')
                                pre_label = cv2.imread(mask_path + '/' + patient_id + '_' + eye + '_' + pre_treatment + '.png')
                                pre_label = cv2.resize(pre_label, self.image_size)
                                pre_label = cv2.normalize(pre_label, None, 0, 255, cv2.NORM_MINMAX)
                                
                                # print('pre_label',self.output_label_path+ match_par+ '/' + label + '/' + patient_id + '_' + eye + '_' + pre_treatment + '.png')
                                cv2.imwrite(self.output_label_path+ match_par+ '/' + label + '/' + patient_id + '_' + eye + '_' + pre_treatment + '.png', pre_label)
                                
                                if os.path.exists(mask_path + '/' + patient_id + '_' + eye + '_' + post_treatment + '.png'):
                                    post_label = cv2.imread(mask_path + '/' + patient_id + '_' + eye + '_' + post_treatment + '.png')
                                    post_label = cv2.resize(post_label, self.image_size)
                                    post_label = cv2.normalize(post_label, None, 0, 255, cv2.NORM_MINMAX)
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
                                    cv2.imwrite(self.output_label_path+ match_par+ '/' + label + '_move/' + patient_id + '_' + eye + '_' + post_treatment + '.png', result)
                                    result[result == 255] = pre_label[result == 255]

                                    # print('post_treatment',self.output_label_path+ match_par+ '/' + label + '/' + patient_id + '_' + eye + '_' + pre_treatment + '.png')     
                                    cv2.imwrite(self.output_label_path+ match_par+ '/' + label + '/' + patient_id + '_' + eye + '_' + post_treatment + '.png', result)

                        
                        
                        
                                    
                        if  translation[0] !=0 or translation[1]!=0 or rotation_angle !=0 or scale != 1:
                            pre_treatment_img = self.output_image_path+ '1/' + patient_id + '_' + eye + '_' + pre_treatment + '.png'
                            post_treatment_img = self.output_image_path + '1/' + patient_id + '_' + eye + '_' + post_treatment + '.png'
                            matching_img = self.output_path  + '/1_move/' + patient_id + '_' + eye + '_' + post_treatment + '.png'                              
                            mse,psnr,ssim, matching_mse,matching_psnr,matching_ssim = self.evaluates(pre_treatment_img,post_treatment_img,matching_img)
                        
                            if psnr == float('inf') :
                                continue
                            if ssim == float('inf') : 
                                continue
                            if matching_psnr == float('inf') :
                                continue
                            if matching_ssim == float('inf')  :
                                continue
                            if ssim  < 0 or psnr < 0 or mse <0 or matching_ssim  < 0 or  matching_psnr < 0 or matching_mse <0 :
                                continue
                                
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
        matching_mse_list = []
        matching_psnr_list = []
        matching_ssim_list = []
        for filename in filenames: 
            if filename.endswith(".png"):
                patient_id, eye, date = filename.split('.png')[0].split('_')
                if patient_id + '_' + eye not in patient :
                    patient[patient_id + '_' + eye] = {}
                    post_treatment = ''
                    pre_treatment = date
                    # patient[patient_id + '_' + eye]['pre_treatment'] = date
                else :
                    patient[patient_id + '_' + eye][date] = {}
                    post_treatment =  date 

                    if pre_treatment_file != '' and post_treatment_file != '':
                        patient[patient_id + '_' + eye][date]['pre_treatment'] = pre_treatment
                        pre_treatment_img = self.output_image_path+ '1/' + patient_id + '_' + eye + '_' + pre_treatment + '.png'
                        post_treatment_img = self.output_image_path + '1/' + patient_id + '_' + eye + '_' + post_treatment + '.png'
                        matching_img = match_path  + '/1_move/' + patient_id + '_' + eye + '_' + post_treatment + '.png'
                        mse,psnr,ssim, matching_mse,matching_psnr,matching_ssim = self.evaluates(pre_treatment_img,post_treatment_img,matching_img)

                        if psnr == float('inf') :
                            continue
                        if ssim == float('inf') : 
                            continue
                        if matching_psnr == float('inf') :
                            continue
                        if matching_ssim == float('inf')  :
                            continue
                        if ssim  < 0 or psnr < 0 or mse <0 or matching_ssim  < 0 or  matching_psnr < 0 or matching_mse <0 :
                            continue
                        
                        patient[patient_id + '_' + eye][date]['original'] = [mse,psnr,ssim]
                        patient[patient_id + '_' + eye][date]['matching'] = [matching_mse,matching_psnr,matching_ssim]
                        mse_list.append(mse)
                        psnr_list.append(psnr)
                        ssim_list.append(ssim)
                        matching_mse_list.append(matching_mse)
                        matching_psnr_list.append(matching_psnr)
                        matching_ssim_list.append(matching_ssim)

                        

                        if matching_ssim > ssim :
                            patient[patient_id + '_' + eye][date]['ssim'] = 'better'
                        elif matching_ssim == ssim :
                            patient[patient_id + '_' + eye][date]['ssim'] = 'same'
                        else :
                            patient[patient_id + '_' + eye][date]['ssim'] = 'worse'

                        if matching_psnr > psnr :
                            patient[patient_id + '_' + eye][date]['psnr'] = 'better'
                        elif matching_psnr == psnr :
                            patient[patient_id + '_' + eye][date]['psnr'] = 'same'
                        else :
                            patient[patient_id + '_' + eye][date]['psnr'] = 'worse'

                        if matching_mse < mse :
                            patient[patient_id + '_' + eye][date]['mse'] = 'better'
                        elif matching_mse == mse :
                            patient[patient_id + '_' + eye][date]['mse'] = 'same'
                        else :
                            patient[patient_id + '_' + eye][date]['mse'] = 'worse'

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
        matching_mse_std = round(np.std(matching_mse_list, ddof=1),2)
        matching_psnr_std = round(np.std(matching_psnr_list, ddof=1),2)
        matching_ssim_std = round(np.std(matching_ssim_list, ddof=1),2)

        avg_mse = round(sum(mse_list)/len(mse_list),2)
        avg_psnr = round(sum(psnr_list)/len(psnr_list),2)
        avg_ssim = round(sum(ssim_list)/len(ssim_list),2)
        mse_std = round(np.std(mse_list, ddof=1),2)
        psnr_std = round(np.std(psnr_list, ddof=1),2)
        ssim_std = round(np.std(ssim_list, ddof=1),2)
            
        patient['avg'] = {}
        patient['avg']['original'] = {}
        patient['avg']['original']['mse'] =avg_mse
        patient['avg']['original']['psnr'] = avg_psnr
        patient['avg']['original']['ssim'] = avg_ssim
        patient['avg']['matching'] = {}
        patient['avg']['matching']['mse'] = matching_avg_mse
        patient['avg']['matching']['psnr'] = matching_avg_psnr
        patient['avg']['matching']['ssim'] = matching_avg_ssim
        
        patient['std'] = {}
        patient['std']['original'] = {}
        patient['std']['original']['mse'] = mse_std
        patient['std']['original']['psnr'] = psnr_std
        patient['std']['original']['ssim'] = ssim_std
        
        patient['std']['matching'] = {}
        patient['std']['matching']['mse'] = matching_mse_std
        patient['std']['matching']['psnr'] = matching_psnr_std
        patient['std']['matching']['ssim'] = matching_ssim_std
        
        

        if best_case != {} :
            patient['best_case'] = best_case

        if worst_case != {} :
            patient['worst_case'] = worst_case

        return patient



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
    pre_treatment_file = "..\\..\\Data\\PCV_1120\\ALL\\1\\08707452_L_20161130.png"
    post_treatment_file = "..\\..\\Data\\PCV_1120\\ALL\\1\\08707452_L_20170118.png"
    
    date = '0305'
    disease = 'PCV'
    PATH_DATA = '../../Data/' 
    PATH_BASE = PATH_DATA  + disease + '_' + date + '/'

    label_path = PATH_DATA + 'labeled' + '/'
    PATH_IMAGE = PATH_DATA + 'OCTA/' 
    output_image_path = PATH_BASE + 'inpaint/'
    image_path = PATH_BASE + 'inpaint/MATCH/' 
    output_label_path = output_image_path + 'MATCH_LABEL/' 
    distances = [0.9]
    features = ['FREAK']#,'SIFT','KAZE','AKAZE','ORB','BRISK' 
    matchers = ['BF']# ,'FLANN'
    patient_list = get_data_from_txt_file('PCV.txt')
    setFolder('./record/'+ disease + '_' + date + '/') 
    for distance in distances:
        for feature in features:
            for matcher in matchers:
                # print(image_path + 'crop' + '_' +  feature + '_' + matcher + '_' + str(distance))

                find = finding(label_path,image_path,output_label_path,output_image_path,features,matchers,distance)
                
                find_dict = find.feature(feature,matcher,distance)
                find_dict = convert_float32_to_float(find_dict)
                json_file = './record/'+ disease + '_' + date + '/'+ 'crop_'+feature + '_' + matcher + '_' + str(distance) + '_align.json'
                tools.write_to_json_file(json_file, find_dict)
                
                # eval = find.all_evaluate(image_path + 'crop' + '_' +  feature + '_' + matcher + '_' + str(distance))
                # json_file2 = './record/'+ disease + '_' + date + '/'+ 'crop_'+feature + '_' + matcher + '_' + str(distance) + '_evals.json'
                # tools.write_to_json_file(json_file2, eval)
                eval = find.all_evaluate(image_path + 'crop' + '_' +  feature + '_' + matcher + '_' + str(distance))
                csv_file = './record/'+ disease + '_' + date + '/'+ feature + '_' + matcher + '_' + str(distance) + '_evals.csv'
                cases = 0
                with open(csv_file, 'w' , newline='') as f:
                    csv_writer = csv.writer(f)
                    csv_writer.writerow(['patient', 'eye','post_treatment', 'pre_treatment', 'mse', 'psnr', 'ssim', 'matching_mse', 'matching_psnr', 'matching_ssim'])
                    for patient_eye in eval:
                        # print("patient_eye",patient_eye)
                        if "best_case" not in patient_eye and "worst_case" not in patient_eye and "avg" not in patient_eye and "std" not in patient_eye:
                            patient, eye = patient_eye.split('_')
                            
                            for treatment in eval[patient_eye]:
                                if treatment != "pre_treatment":
                                    # print('date',eval[patient_eye][date]['pre_treatment'])
                                    cases += 1
                                    if 'original' in eval[patient_eye][treatment] and 'matching' in eval[patient_eye][treatment]:
                                        csv_writer.writerow([patient,eye, treatment, 
                                                                eval[patient_eye][treatment]['pre_treatment'],
                                                                eval[patient_eye][treatment]['original'][0],
                                                                eval[patient_eye][treatment]['original'][1],
                                                                eval[patient_eye][treatment]['original'][2],
                                                                eval[patient_eye][treatment]['matching'][0],
                                                                eval[patient_eye][treatment]['matching'][1],
                                                                eval[patient_eye][treatment]['matching'][2]])
                    
                    csv_writer.writerow(['avg', '', '', '', eval["avg"]['original']['mse'], eval["avg"]['original']['psnr'], eval["avg"]['original']['ssim'], eval["avg"]['matching']['mse'], eval["avg"]['matching']['psnr'], eval["avg"]['matching']['ssim']])
                    csv_writer.writerow(['std', '', '', '', eval["std"]['original']['mse'], eval["std"]['original']['psnr'], eval["std"]['original']['ssim'], eval["std"]['matching']['mse'], eval["std"]['matching']['psnr'], eval["std"]['matching']['ssim']])
                        
                aligment_file = './record/'+ disease + '_' + date + '/' + 'evaluations.csv'
                if not os.path.exists(aligment_file):
                    with open(aligment_file, 'w', newline='') as f:
                        csv_writer = csv.writer(f)  
                        csv_writer.writerow(['feature', 'cases','avg_mse', 'avg_psnr', 'avg_ssim','std_mse', 'std_psnr', 'std_ssim','avg_matching_mse', 'avg_matching_psnr', 'avg_matching_ssim','std_matching_mse', 'std_matching_psnr', 'std_matching_ssim'])

                with open(aligment_file, 'a', newline='') as f:
                    csv_writer = csv.writer(f)
                    csv_writer.writerow([feature + '_' + matcher + '_' + str(distance),
                                            cases ,
                                            eval["avg"]['original']['mse'],
                                            eval["avg"]['original']['psnr'],
                                            eval["avg"]['original']['ssim'],
                                            eval["std"]['original']['mse'],
                                            eval["std"]['original']['psnr'],
                                            eval["std"]['original']['ssim'],
                                            eval["avg"]['matching']['mse'],
                                            eval["avg"]['matching']['psnr'],
                                            eval["avg"]['matching']['ssim'],
                                            eval["std"]['matching']['mse'],
                                            eval["std"]['matching']['psnr'],
                                            eval["std"]['matching']['ssim']
                                            ])


