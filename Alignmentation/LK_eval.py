# Lucas-Kanade

import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
import time
import json
import tools.tools as tools
import math
import csv
from skimage.metrics import mean_squared_error
from skimage.metrics import peak_signal_noise_ratio
from skimage.metrics import structural_similarity
def setFolder(path):
    os.makedirs(path, exist_ok=True)
def mean_squared_error_ignore_zeros(img1, img2,img3):
    # 對於兩張影像 計算每個pixel 的差異
    # 找出兩影像中相應像素都為零的位置
    both_zeros_mask  = (img1 == 0) & (img2 == 0) & (img3 == 0)

    
    diff = img1[~both_zeros_mask] - img2[~both_zeros_mask]
    mse = np.mean(diff ** 2)
    return mse
def psnr_ignore_zeros(img1, img2,img3):
    
    both_zeros_mask  = (img1 == 0) & (img2 == 0) & (img3 == 0)
    diff = img1[~both_zeros_mask] - img2[~both_zeros_mask] 
    mse = np.mean(diff**2)
    # 如果 MSE 為 0，則 PSNR 為無窮大
    if mse == 0:
        return float('inf')
    max_pixel_value = 255.0
    # 計算 PSNR
    psnr = 20 * np.log10(max_pixel_value / np.sqrt(mse))
    return psnr

def ssim_ignore_zeros(img1, img2,img3):
    both_zeros_mask  = (img1 == 0) & (img2 == 0) & (img3 == 0)
    # 提取非零区域
    img1_non_zero = img1[~both_zeros_mask]
    img2_non_zero = img2[~both_zeros_mask]
    
    
    # SSIM
    ssim_index, _ = structural_similarity(img1_non_zero, img2_non_zero, full=True)
    return ssim_index

class LK():
    def __init__(self,patient_list,label_path,image_path,output_label_path,output_image_path,methods,matchers ,distance):
        # self.base_path = base_path
        self.patient_list = patient_list
        self.label_path = label_path
        self.image_path = image_path
        self.distances = distance
        self.output_label_path = output_label_path
        self.image_size = (304, 304)

        self.data_list = ['1','2', '3', '4', '1_OCT', '2_OCT', '3_OCT', '4_OCT']
        self.label_list = ['CC', 'OR']
        self.methods = methods 
        self.matchers = matchers
        self.output_image_path = output_image_path

    # distance=0.8, method='SIFT'
    def LK(self,img1, img2,  distance=0.8, method='ORB',matcher='FLANN'):
        # Initiate SIFT detector
        if method == 'SIFT':
            sift = cv2.SIFT_create( )
        elif method == 'KAZE':
            sift = cv2.KAZE_create( )
        elif method == 'AKAZE':
            sift = cv2.AKAZE_create()
        elif method == 'ORB':
            sift = cv2.ORB_create()
        elif method == 'BRISK':
            sift = cv2.BRISK_create()
        elif method == 'BRIEF':
            detector = cv2.FastFeatureDetector_create()
            brief = cv2.xfeatures2d.BriefDescriptorExtractor_create()
        

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


        img = cv2.drawKeypoints(img1, kp1, img1, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        img2_draw = cv2.drawKeypoints(img2, kp2, img2, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        # plt.imshow(img)
        # plt.axis('off')
        # plt.title(method)
        # plt.show()
        # plt.imshow(img2_draw)
        # plt.axis('off')
        # plt.title(method)
        # plt.show()        

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
                search_params = dict(checks=10)

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
                search_params = dict(checks=100)

                flann = cv2.FlannBasedMatcher(index_params, search_params)
                matches = flann.knnMatch(des1, des2, k=2)
        if not matches:
            print('No matches found')
            H = np.array(np.eye(3))
            return H
        if matcher == 'BF': 
            # Need to draw only good matches, so create a mask
            matches = sorted(matches, key=lambda x: x[0].distance)
            min_dist = matches[0][0].distance
            # Apply ratio test
            good = []
            pts1 = []
            pts2 = []
            for i, (m, n) in enumerate(matches):
                if m.distance < distance* n.distance:
                    good.append([m])
                    pts2.append(kp2[m.trainIdx].pt)
                    pts1.append(kp1[m.queryIdx].pt)

                if m.distance > 1.5 *min_dist:
                    break
            # Draw matches
            img3 = cv2.drawMatchesKnn(img1 , kp1, img2, kp2, good, None, flags=2)

            
        elif matcher == 'FLANN':
            # Need to draw only good matches, so create a mask
            matchesMask = [[0, 0] for i in range(len(matches))]
            pts1 = []
            pts2 = []
            if matches:
                # matches = sorted(matches, key=lambda x: x[0].distance)
                min_dist = matches[0][0].distance
                # ratio test as per Lowe's paper
                
                for i, match in enumerate(matches):
                    if len(match) != 2 or match[0] is None or match[1] is None:
                        continue

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

        # Assuming pts1 and pts2 are lists of corresponding points
        pts1 = np.array(pts1, dtype=np.float32)
        pts2 = np.array(pts2, dtype=np.float32)
        # Check for non-empty point sets
        if len(pts1) == 0 or len(pts2) == 0:
            # print("Error: Empty point sets.")
            # print("Translation: ", None)
            # print("Rotation angle: ", None)
            # print("Scale: ", None)
            H = np.array(np.eye(3))
            # print(H)
            return  H


        # Find homography
        if len(pts1) < 4 or len(pts2) < 4:
            # print("Error: Insufficient corresponding points to calculate Homography.")
            H = np.array(np.eye(3))
            # print(H)
            return H

        # Find the transformation matrix using RANSAC
        H, _ = cv2.findHomography(pts2, pts1, cv2.RANSAC, 5.0)

        if H is None:
            H = np.array(np.eye(3))
            return H

        translation = (H[0, 2], H[1, 2])
        rotation_angle = np.arctan2(H[1, 0], H[0, 0])  # angle in radians
        rotation = rotation_angle * 180 / np.pi  # angle in degrees []
        scale = np.sqrt(H[0, 0] ** 2 + H[1, 0] ** 2)
        if scale < 0.5 or scale > 1.5:
            H = np.array(np.eye(3))
            return H
        if translation[0] < -304//2 or translation[0] > 304//2 or translation[1] < -304//2 or translation[1] > 304//2:

            H = np.array(np.eye(3))
            return H

        if rotation < -180 or rotation > 180:
            H = np.array(np.eye(3))
            return H


        # # # show
        # img_out = cv2.warpPerspective(img1, H, (img2.shape[1], img2.shape[0]))
        # fig, ax = plt.subplots(1, 3, figsize=(15, 15))
        # ax[0].imshow(img1,cmap='gray')
        # ax[0].set_title('Input Image')
        # ax[0].axis('off')
        # ax[1].imshow(img2,cmap='gray')
        # ax[1].set_title('Reference Image')
        # ax[1].axis('off')
        # ax[2].imshow(img_out,cmap='gray')
        # ax[2].set_title('Output Image')
        # ax[2].axis('off')
        # plt.show()

        return  H

    def evaluates(self, pre_treatment_img, post_treatment_img,matching_img):
        pre_treatment = cv2.imread(pre_treatment_img , cv2.IMREAD_GRAYSCALE)
        post_treatment = cv2.imread(post_treatment_img, cv2.IMREAD_GRAYSCALE)
        cmp_matching_img = cv2.imread(matching_img, cv2.IMREAD_GRAYSCALE)

        cmp_pre_treatment_img = pre_treatment.copy()  
        cmp_post_treatment_img = post_treatment.copy()          
        cmp_pre_treatment_img [cmp_matching_img == 0] = 0
        cmp_post_treatment_img [cmp_matching_img == 0] = 0
        # fig , ax = plt.subplots(2,3,figsize=(15,15))
        # ax[0][0].imshow(pre_treatment,cmap='gray')
        # ax[0][0].set_title('pre_treatment')
        # ax[0][0].axis('off')
        # ax[0][1].imshow(post_treatment,cmap='gray')
        # ax[0][1].set_title('post_treatment')
        # ax[0][1].axis('off')
        # ax[0][2].imshow(cmp_matching_img,cmap='gray')
        # ax[0][2].set_title('matching_img')
        # ax[0][2].axis('off')
        # ax[1][0].imshow(cmp_pre_treatment_img,cmap='gray')
        # ax[1][0].set_title('cmp_pre_treatment_img')
        # ax[1][0].axis('off')
        # ax[1][1].imshow(cmp_post_treatment_img,cmap='gray')
        # ax[1][1].set_title('cmp_post_treatment_img')
        # ax[1][1].axis('off')
        # ax[1][2].imshow(cmp_matching_img,cmap='gray')
        # ax[1][2].set_title('cmp_matching_img')
        # ax[1][2].axis('off')
        # plt.show()
        
        
                        
        if cmp_pre_treatment_img is None or cmp_post_treatment_img is None or cmp_matching_img is None:
            return -1 , -1 , -1,-1 , -1 , -1

        if cmp_pre_treatment_img.shape != cmp_post_treatment_img.shape and cmp_post_treatment_img.shape != cmp_matching_img.shape : 
            return -1 , -1 , -1,-1 , -1 , -1
        mse = mean_squared_error_ignore_zeros(cmp_pre_treatment_img ,    cmp_post_treatment_img,cmp_matching_img)
        psnr = psnr_ignore_zeros( cmp_pre_treatment_img ,    cmp_post_treatment_img,cmp_matching_img)
        ssim = ssim_ignore_zeros(cmp_pre_treatment_img ,    cmp_post_treatment_img,cmp_matching_img)
        ssim = ssim * 100
        matching_mse = mean_squared_error_ignore_zeros(cmp_pre_treatment_img, cmp_matching_img,cmp_matching_img)
        matching_psnr = psnr_ignore_zeros(cmp_pre_treatment_img, cmp_matching_img,cmp_matching_img)
        matching_ssim = ssim_ignore_zeros(cmp_pre_treatment_img, cmp_matching_img ,cmp_matching_img)
        matching_ssim = matching_ssim * 100

        return mse,psnr,ssim ,matching_mse,matching_psnr,matching_ssim


    def alignment(self, patient_id, eye, pre_treatment, post_treatment,feature, matcher, distance):
        pre_img = cv2.imread(self.output_image_path + '1/' + patient_id + '_' + eye + '_' + pre_treatment + '.png')
        pre_img = cv2.resize(pre_img, self.image_size)
        pre_img = cv2.normalize(pre_img, None, 0, 255, cv2.NORM_MINMAX)
        post_img = cv2.imread(self.output_image_path + '1/' + patient_id + '_' + eye + '_' + post_treatment + '.png')
        post_img = cv2.resize(post_img, self.image_size)
        post_img = cv2.normalize(post_img, None, 0, 255, cv2.NORM_MINMAX)
        gray1 = cv2.cvtColor(pre_img, cv2.COLOR_BGR2GRAY)
        gray1 = cv2.normalize(gray1, None, 0, 255, cv2.NORM_MINMAX)
        gray2 = cv2.cvtColor(post_img, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.normalize(gray2, None, 0, 255, cv2.NORM_MINMAX)

        self.output_path = self.image_path + '/' + feature + '_' + matcher + '_' + str(distance) + '/'
        # print(self.output_path)
        # print(patient_id , eye , pre_treatment , post_treatment)
        
        H = self.LK(gray2, gray1, method=feature, matcher=matcher, distance=distance)
        # Check the shape of H



        for data in self.data_list:

            if os.path.exists(self.output_image_path + data + '/' + patient_id + '_' + eye + '_' + pre_treatment + '.png'):
                pre_image = cv2.imread(self.output_image_path + data + '/' + patient_id + '_' + eye + '_' + pre_treatment + '.png')
                pre_image = cv2.resize(pre_image, self.image_size)
                pre_image = cv2.normalize(pre_image, None, 0, 255, cv2.NORM_MINMAX)
            

                cv2.imwrite(self.output_path + data + '/' + patient_id + '_' + eye + '_' + pre_treatment + '.png', pre_image)
                if os.path.exists(self.output_image_path + data + '/' + patient_id + '_' + eye + '_' + post_treatment + '.png'):
                    post_image = cv2.imread(self.output_image_path + data + '/' + patient_id + '_' + eye + '_' + post_treatment + '.png')
                    post_image = cv2.resize(post_image, self.image_size)
                    post_image = cv2.normalize(post_image, None, 0, 255, cv2.NORM_MINMAX)
                    height, width, channels = post_image.shape
                    if H is None :
                        result = post_image
                    else:
                        # print(H)
                        translation = (H[0, 2], H[1, 2])
                        rotation = math.atan2(H[1, 0], H[0, 0])
                        rotation_angle = rotation * 180 / np.pi  # angle in degrees []
                        scale = H[0, 0] / np.cos(rotation_angle)



                        if scale < 0.5 or scale > 2:
                            H = np.array(np.eye(3))
                            result = post_image
                        elif translation[0] < -304//2 or translation[0] > 304//2 or translation[1] < -304//2 or translation[1] > 304//2:
                            H = np.array(np.eye(3))
                            result = post_image

                        elif rotation_angle < -60 or rotation_angle > 60:
                            H = np.array(np.eye(3))
                            result = post_image
                        elif np.isnan(H).any():
                            H = np.array(np.eye(3))
                            result = post_image

                        else:
                            result = cv2.warpPerspective(post_image, H, (width, height))

                    if not os.path.exists(self.output_path + data + '_move/'):
                        os.makedirs(self.output_path + data + '_move/')
                    cv2.imwrite(self.output_path + data + '_move/' + patient_id + '_' + eye + '_' + post_treatment + '.png', result)
                    result[result == 0] = pre_image[result == 0]

                    cv2.imwrite(self.output_path + data + '/' + patient_id + '_' + eye + '_' + post_treatment + '.png', result)
                    vis_img = result.copy()
                    vis_img[:,:,0] = 0
                    vis_img[:,:,2] = 0

                    # vis_img = cv2.addWeighted(pre_image, 0.5, vis_img, 0.5, 0)
                    # if not os.path.exists(self.output_path + '/' + data + '_vis/'):
                    #     os.makedirs(self.output_path + '/' + data + '_vis/')

                    # if not os.path.exists(self.output_path + '/' + data + '_vis/'):
                    #     os.makedirs(self.output_path + '/' + data + '_vis/')

                    # cv2.imwrite(self.output_path + '/' + data + '_vis/' + patient_id + '_' + eye + '_' + post_treatment + '_vis.png', vis_img)
            
                    # show
                    # fig, ax = plt.subplots(1, 4, figsize=(15, 15))
                    # ax[0].imshow(pre_image)
                    # ax[0].set_title('pre_image')
                    # ax[1].imshow(post_image)
                    # ax[1].set_title('post_image')
                    # ax[2].imshow(result)
                    # ax[2].set_title('result')
                    # ax[3].imshow(vis_img)
                    # ax[3].set_title('vis_img')
                    # plt.show()
        
        # self.output_label = self.output_label_path + '/' + feature + '_' + matcher + '_' + str(distance) 
        match_par = feature + '_' + matcher + '_' + str(distance) 
        for label in self.label_list:
            
            
            if not os.path.exists(self.output_label_path + '/' + match_par +  '/' + label ):
                os.makedirs(self.output_label_path + '/' +match_par  +  '/' + label )
            if not os.path.exists(self.output_label_path + '/' +match_par  +  '/' + label +'_move/'):
                os.makedirs(self.output_label_path + '/' + match_par + '/' + label +'_move/')
                
            mask_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(self.image_path))),label,'masks')
            
            if os.path.exists(mask_path + '/'+ label + '_' + patient_id + '_' + eye + '_' + pre_treatment + '.png'):
                pre_label = cv2.imread(mask_path + '/'+ label + '_' + patient_id + '_' + eye + '_' + pre_treatment + '.png')
                pre_label = cv2.resize(pre_label, self.image_size)
                pre_label = cv2.normalize(pre_label, None, 0, 255, cv2.NORM_MINMAX)
                
                # print('pre_label',self.output_label_path+ match_par+ '/' + label + '/' + patient_id + '_' + eye + '_' + pre_treatment + '.png')
                cv2.imwrite(self.output_label_path+ match_par+ '/' + label + '/' + patient_id + '_' + eye + '_' + pre_treatment + '.png', pre_label)
                
                if os.path.exists(mask_path + '/'+ label + '_' + patient_id + '_' + eye + '_' + post_treatment + '.png'):
                    post_label = cv2.imread(mask_path + '/'+ label + '_' + patient_id + '_' + eye + '_' + post_treatment + '.png')
                    post_label = cv2.resize(post_label, self.image_size)
                    post_label = cv2.normalize(post_label, None, 0, 255, cv2.NORM_MINMAX)
                    height, width, channels = post_label.shape
                    if H is None :
                        result = post_label
                    else:
                        # print(H)
                        translation = (H[0, 2], H[1, 2])
                        rotation = math.atan2(H[1, 0], H[0, 0])
                        rotation_angle = rotation * 180 / np.pi  # angle in degrees []
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


                 
               
            

        
        if H is None:
            return None, None, None
        translation = [H[0, 2], H[1, 2]]
        rotation_angle = np.arctan2(H[1, 0], H[0, 0])  # angle in radians
        scale = H[0, 0] / np.cos(rotation_angle)

        return   translation, rotation_angle, scale

    def registration(self, feature, matcher, distance):

        for data in self.data_list: 
            setFolder(self.image_path + '/' + feature + '_' + matcher + '_' + str(distance)+'/'+ data) 
        patient_dict = {}
        image_folder = self.output_image_path+ '/1/'
        filenames = sorted(os.listdir(image_folder))
        print(feature, matcher, distance)
        print('------------------')

        for filename in filenames: 
            if filename.endswith(".png"):

                patient_id, eye, date = filename.split('.png')[0].split('_')
                if patient_id not in self.patient_list:
                    continue
                if patient_id + '_' + eye not in patient_dict :
                    patient_dict[patient_id + '_' + eye] = {}
                    pre_treatment = date
                    post_treatment = ''
                else :
                    post_treatment =  date 

                    treatment_patient = patient_id
                    treatment_eye = eye

                    # print(treatment_patient, treatment_eye, pre_treatment, post_treatment)

                    if pre_treatment != '' and post_treatment != '':
                        translation, rotation_angle, scale = self.alignment(treatment_patient, treatment_eye, pre_treatment, post_treatment, feature, matcher, distance)
                        
                        patient_dict[patient_id + '_' + eye][post_treatment] = {}
                        patient_dict[patient_id + '_' + eye][post_treatment]['translation']  = [translation[0], translation[1]]
                        patient_dict[patient_id + '_' + eye][post_treatment]['rotation_angle'] = rotation_angle
                        patient_dict[patient_id + '_' + eye][post_treatment]['scale'] = scale

                        pre_image = self.output_image_path + '1/' + patient_id + '_' + eye + '_' + str(pre_treatment) + '.png'
                        post_image = self.output_image_path + '1/' + patient_id + '_' + eye + '_' + str(post_treatment) + '.png'
                        matching_img = self.output_path + '1_move/' + patient_id + '_' + eye + '_' + str(post_treatment) + '.png'
                        mse,psnr,ssim, matching_mse,matching_psnr,matching_ssim = self.evaluates(pre_image,post_image,matching_img)
                        if psnr == float('inf') :
                            continue
                        if ssim == float('inf') : 
                            continue
                        if ssim  < 0 or psnr < 0 or mse <0 :
                            continue    
                        if matching_psnr == float('inf') :
                            continue
                        if matching_ssim == float('inf') :
                            continue
                        if matching_ssim  < 0 or matching_psnr < 0 or matching_mse <0 :
                            continue
                        if  translation[0] !=0 or translation[1]!=0 or rotation_angle !=0 or scale != 1:
                            patient_dict[patient_id + '_' + eye][post_treatment]['original'] = [mse , psnr , ssim]
                            patient_dict[patient_id + '_' + eye][post_treatment]['matching'] = [matching_mse , matching_psnr , matching_ssim]

                            if matching_ssim > ssim :
                                patient_dict[patient_id + '_' + eye][post_treatment]['ssim'] = 'better'
                            elif matching_ssim == ssim :
                                patient_dict[patient_id + '_' + eye][post_treatment]['ssim'] = 'same'
                            else :
                                patient_dict[patient_id + '_' + eye][post_treatment]['ssim'] = 'worse'

                            if matching_psnr > psnr :
                                patient_dict[patient_id + '_' + eye][post_treatment]['psnr'] = 'better'
                            elif matching_psnr == psnr :
                                patient_dict[patient_id + '_' + eye][post_treatment]['psnr'] = 'same'
                            else :
                                patient_dict[patient_id + '_' + eye][post_treatment]['psnr'] = 'worse'

                            if matching_mse < mse :
                                patient_dict[patient_id + '_' + eye][post_treatment]['mse'] = 'better'
                            elif matching_mse == mse :
                                patient_dict[patient_id + '_' + eye][post_treatment]['mse'] = 'same'
                            else :
                                patient_dict[patient_id + '_' + eye][post_treatment]['mse'] = 'worse'                        
        
        # delete empty patient
        for patient in list(patient_dict.keys()):
            if len(patient_dict[patient]) == 0:
                del patient_dict[patient]    
        return patient_dict


    def reg_all(self,disease ,date):
        setFolder('./record/'+ disease + '_' + date + '/') 
        for feature in self.methods:
            for matcher in self.matchers:
                for distance in self.distances:
                    shift_dict= self.registration(feature, matcher, distance)
                    json_file = './record/'+ disease + '_' + date + '/'+feature + '_' + matcher + '_' + str(distance) + '_align.json'
                    tools.write_to_json_file(json_file,shift_dict)
                    eval = self.all_evaluate(self.image_path  +  feature + '_' + matcher + '_' + str(distance))
                    # save eval to csv
                    csv_file = './record/'+ disease + '_' + date + '/'+ feature + '_' + matcher + '_' + str(distance) + '_evals.csv'
                    cases = 0
                    with open(csv_file, 'w') as f:
                        csv_writer = csv.writer(f)
                        csv_writer.writerow(['patient', 'eye','post_treatment', 'pre_treatment', 'mse', 'psnr', 'ssim', 'matching_mse', 'matching_psnr', 'matching_ssim'])
                        for patient_eye in eval:
                            print("patient_eye",patient_eye)
                            if "best_case" not in patient_eye and "worst_case" not in patient_eye and "avg" not in patient_eye and "std" not in patient_eye:
                                patient, eye = patient_eye.split('_')
                                
                                for treatment in eval[patient_eye]:
                                    if treatment != "pre_treatment":
                                        # print('date',eval[patient_eye][date]['pre_treatment'])
                                        cases += 1
                                        csv_writer.writerow([patient,eye, treatment, 
                                                                eval[patient_eye][treatment]['pre_treatment'],
                                                                eval[patient_eye][treatment]['original'][0],
                                                                eval[patient_eye][treatment]['original'][1],
                                                                eval[patient_eye][treatment]['original'][2],
                                                                eval[patient_eye][treatment]['matching'][0],
                                                                eval[patient_eye][treatment]['matching'][1],
                                                                eval[patient_eye][treatment]['matching'][2]])
                            
                    aligment_file = './record/'+ disease + '_' + date + '/' + 'evaluations.csv'
                    if not os.path.exists(aligment_file):
                        with open(aligment_file, 'w') as f:
                            csv_writer = csv.writer(f)  
                            csv_writer.writerow(['feature', 'matcher', 'distance', 'cases','avg_mse', 'avg_psnr', 'avg_ssim','std_mse', 'std_psnr', 'std_ssim','avg_matching_mse', 'avg_matching_psnr', 'avg_matching_ssim','std_matching_mse', 'std_matching_psnr', 'std_matching_ssim'])

                    with open(aligment_file, 'a') as f:
                        csv_writer = csv.writer(f)
                        csv_writer.writerow([feature, matcher, distance, cases ,
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

                                                                
                                                                
                                                                
                                
                                
                                     
                    # json_file2 = './record/'+ disease + '_' + date + '/'+ feature + '_' + matcher + '_' + str(distance) + '_evals.json'
                    # tools.write_to_json_file(json_file2,eval)



    def detect_blood_vessels(self,image,min_area = 4):

        # otsu threshold
        # ret, binary_image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(image, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_TC89_L1)

        # minimum area
        contours_img = np.zeros_like(image)

        for contour in contours:
            if cv2.contourArea(contour) >= min_area:
                cv2.drawContours(contours_img, [contour], -1, 255, -1)
                
        return contours_img


    # def evaluate(self,image ,original_image):
    #     image = cv2.imread(image, cv2.IMREAD_GRAYSCALE)
    #     original_image = cv2.imread(original_image, cv2.IMREAD_GRAYSCALE)
    #     if image is None or original_image is None:
    #         print('Could not open or find the images!')
    #         return -1 , -1 , -1

    #     if image.shape != original_image.shape:
    #         print('The images have different dimensions!')
    #         print(image,image.shape)
    #         print(original_image,original_image.shape)
    #         return -1 , -1 , -1
            
        
    #     mse = mean_squared_error_ignore_zeros(image, original_image,
    #     psnr = psnr_ignore_zeros(image, original_image)
    #     ssim = ssim_ignore_zeros(image, original_image)
    #     ssim = ssim * 100

    #     return mse,psnr,ssim

    def all_evaluate(self,match_path):

        patient = {}
        image_folder = match_path+ '/1/'
        filenames = sorted(os.listdir(image_folder))

        best_case = dict()
        worst_case = dict()
        best_differece_ssim = 0
        worst_differece_ssim = 100000000000000000000
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

                    if pre_treatment != '' and post_treatment != '':
                        patient[patient_id + '_' + eye][date]['pre_treatment'] = pre_treatment
                        pre_treatment_img = self.output_image_path+ '1/' + patient_id + '_' + eye + '_' + pre_treatment + '.png'
                        post_treatment_img = self.output_image_path + '1/' + patient_id + '_' + eye + '_' + post_treatment + '.png'
                        matching_img = match_path  + '/1_move/' + patient_id + '_' + eye + '_' + post_treatment + '.png'


                        mse,psnr,ssim, matching_mse,matching_psnr,matching_ssim = self.evaluates(pre_treatment_img,post_treatment_img,matching_img)

                        if psnr == float('inf') :
                            continue
                        if ssim == float('inf') : 
                            continue
                        if ssim  < 0 or psnr < 0 or mse <0 :
                            continue


                        patient[patient_id + '_' + eye][date]['original'] = [mse,psnr,ssim]

                        if matching_psnr == float('inf') :
                            continue
                        if matching_ssim == float('inf')  :
                            continue
                        if matching_ssim  < 0 or  matching_psnr < 0 or matching_mse <0 :
                            continue

                        mse_list.append(mse)
                        psnr_list.append(psnr)
                        ssim_list.append(ssim)
                        matching_mse_list.append(matching_mse)
                        matching_psnr_list.append(matching_psnr)
                        matching_ssim_list.append(matching_ssim)

                        patient[patient_id + '_' + eye][date]['matching'] = [matching_mse,matching_psnr,matching_ssim]

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

                        if mse - matching_mse > best_differece_ssim :
                            best_differece_ssim = matching_ssim - ssim
                            best_case['patient'] = [ patient_id,eye,post_treatment]
                            best_case ['psnr'] = psnr
                            best_case ['ssim'] = ssim
                            best_case ['mse'] = mse
                            best_case ['matching_psnr'] = matching_psnr
                            best_case ['matching_ssim'] = matching_ssim
                            best_case ['matching_mse'] = matching_mse

                        if mse - matching_mse < worst_differece_ssim :
                            worst_differece_ssim = matching_ssim - ssim
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

def find_darkest_region(image):

    # 計算影像的負片（invert）
    inverted_image = cv2.bitwise_not(image)

    # 使用形態學操作（開運算）來去除小的亮區域
    kernel = np.ones((5, 5), np.uint8)
    opened_image = cv2.morphologyEx(inverted_image, cv2.MORPH_OPEN, kernel)

    # 找到最大的連通區域
    _, labels, stats, _ = cv2.connectedComponentsWithStats(opened_image)

    # 找到最大連通區域的索引
    largest_region_index = np.argmax(stats[:, cv2.CC_STAT_AREA])

    # 使用最大連通區域的索引來提取最黑的區域
    darkest_region = np.zeros_like(inverted_image)
    darkest_region[labels == largest_region_index] = 255

    # 將結果轉換回原始形式
    darkest_region = cv2.bitwise_not(darkest_region)

    return darkest_region       


def convert_to_json_serializable(obj):
    if isinstance(obj, np.float32):
        return float(obj)
    raise TypeError(f'Object of type {obj.__class__.__name__} is not JSON serializable')

def write_to_json_file(data, filename):
    with open(filename, 'w') as json_file:
        json.dump(data, json_file, default=convert_to_json_serializable, indent=4)

def get_data_from_txt_file(filename):
    # get txt
    data = []
    with open(filename, 'r') as file:
        for line in file:
            data.append(line.strip())
    return data

if __name__ == '__main__':
    date = '0205'
    disease = 'PCV'
    PATH_DATA = '../../Data/' 
    PATH_BASE = PATH_DATA  + disease + '_' + date + '/'

    label_path = PATH_DATA + 'labeled' + '/'
    PATH_IMAGE = PATH_DATA + 'OCTA/' 
    output_image_path = PATH_BASE + 'ALL/'
    image_path = output_image_path + 'MATCH/' 
    output_label_path = output_image_path + 'MATCH_LABEL/' 
    features = ['SIFT','KAZE','AKAZE','ORB','BRISK','BRIEF','FREAK']
    matchers = ['BF','FLANN']
    distances = [0.7,0.8]
    patient_list = get_data_from_txt_file('PCV.txt')
    # # distance=0.65, method='ORB',matcher='BF'
    LK = LK(patient_list,label_path,image_path,output_label_path,output_image_path,features, matchers, distances)
    # LK.registration(feature = feature, matcher = matcher, distance = distance[0])
    LK.reg_all(disease , date)
 