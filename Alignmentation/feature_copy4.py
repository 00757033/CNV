import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import math
import json
import shutil
import time
import tools.tools as tools
from skimage.feature import corner_harris
from skimage.metrics import mean_squared_error
from skimage.metrics import structural_similarity
from skimage.metrics import peak_signal_noise_ratio
import pathlib as pl
# pip install opencv-contrib-python

# 尋找OCTA 影像的黃斑中心
class finding():
    def __init__(self,path_image,PATH_LABEL,PATH_MATCH,PATH_MATCH_LABEL,PATH_MATCH_IMAGE,distance):
        self.path_image = path_image
        self.path_label = PATH_LABEL
        self.path_match = PATH_MATCH
        self.distances = distance
        self.path_match_image = PATH_MATCH_IMAGE
        self.data_list = ['1','2', '3', '4', '1_OCT', '2_OCT', '3_OCT', '4_OCT']
        self.label_list = ['label_3', 'label_4']


    def adjust_brightness(self,img1, img2):
        # Convert images to float32 for accurate calculations
        img1 = img1.astype(np.float32)
        img2 = img2.astype(np.float32)

        # Compute the average intensity of each image
        avg_intensity_img1 = np.mean(img1)
        avg_intensity_img2 = np.mean(img2)

        # Compute the scaling factor to make the average intensities equal
        scaling_factor = avg_intensity_img1 / avg_intensity_img2

        # Scale the second image
        img2_scaled = img2 * scaling_factor

        # Clip the values to the valid intensity range (0-255)
        img2_scaled = np.clip(img2_scaled, 0, 255)

        # Convert the images back to uint8 format
        img1 = img1.astype(np.uint8)
        img2_scaled = img2_scaled.astype(np.uint8)

        return img1, img2_scaled

    def preprocess(self, image):
        
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)
        image = cv2.equalizeHist(image)
        # plt.hist(image.ravel(),256,[0,256])
        # plt.title('pre3')
        # plt.show()

        # self.image = cv2.equalizeHist(self.image)
        # 反轉顏色
        # image = cv2.bitwise_not(image)
        # fig, ax = plt.subplots(1, 8, figsize=(20, 4))
        # ax[0].hist(image.ravel(), 256, [0, 256])
        # ax[0].set_title('original')
        image = cv2.equalizeHist(image)
        # ax[1].hist(image.ravel(), 256, [0, 256])
        # ax[1].set_title('equalizeHist')


        image = cv2.GaussianBlur(image, (3, 3), 0)
        # ax[2].hist(image.ravel(), 256, [0, 256])
        # ax[2].set_title('GaussianBlur')

        image = cv2.equalizeHist(image)
        # ax[3].hist(image.ravel(), 256, [0, 256])
        # ax[3].set_title('equalizeHist')
        # plt.hist(image.ravel(),256,[0,256])
        # plt.title('pre3')
        # plt.show()
        image = cv2.GaussianBlur(image, (3, 3), 0)
        # ax[4].hist(image.ravel(), 256, [0, 256])
        # ax[4].set_title('GaussianBlur')
        image = cv2.equalizeHist(image)
        # ax[5].hist(image.ravel(), 256, [0, 256])
        # ax[5].set_title('equalizeHist')

        # plt.hist(image.ravel(),256,[0,256])
        # plt.title('pre3')
        # plt.show()
        image = cv2.GaussianBlur(image, (3, 3), 0)
        # ax[6].hist(image.ravel(), 256, [0, 256])
        # ax[6].set_title('GaussianBlur')

        image = cv2.equalizeHist(image)
        # plt.hist(image.ravel(),256,[0,256])
        # plt.title('pre3')
        # plt.show()

        image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)
        # plt.hist(image.ravel(),256,[0,256])
        # plt.title('pre3333')
        # plt.show()
        # plt

        return image

    def find_center(self,image):
        image2 = image.copy()
        image2 = self.preprocess(image2)
        
        ret1,image2 = cv2.threshold(image2, 32 ,255, cv2.THRESH_BINARY_INV)
        # print(ret1)

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


        # 畫出黃斑中心
        # draw the circle
        cv2.rectangle(draw_image, (int(center[0] - max_radius), int(center[1] - max_radius)), (int(center[0] + max_radius), int(center[1] + max_radius)), (255, 0, 255), 3)
        cv2.circle(draw_image, center, 5, (0, 255, 255), -1)
        t = 40
        cv2.rectangle(draw_image, (int(center[0] - max_radius-t), int(center[1] - max_radius-t)), (int(center[0] + max_radius+ t), int(center[1] + max_radius+ t)), (255, 255, 0), 3)

        cv2.imshow('img',draw_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        if int(center[0] - max_radius) < t :
            t = int(center[0] - max_radius)
        if int(center[1] - max_radius) < t :
            t = int(center[1] - max_radius)
        if int(center[0] + max_radius) > len(image[0]) - t :
            t = len(image[0]) - int(center[0] + max_radius)
        if int(center[1] + max_radius) > len(image) - t :
            t = len(image) - int(center[1] + max_radius)

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


        img1_draw = cv2.drawKeypoints(img1, kp1, img1, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        img2_draw = cv2.drawKeypoints(img2, kp2, img2, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

        plt.imshow(img1_draw)
        plt.axis('off')
        plt.title(method)
        plt.show()
        plt.imshow(img2_draw)
        plt.axis('off')
        plt.title(method)
        plt.show() 
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

                if m.distance > 1.5 * min_dist:
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


        plt.imshow(img3)
        plt.axis('off')
        plt.title(method)
        plt.show()

        # 計算img1 需要旋轉 縮放 平移矩陣 原本img1的實際中心點為center
        pts1 = np.array(pts1)
        pts2 = np.array(pts2)

        if len(pts1) == 0 or len(pts2) == 0:
            # print("Error: Empty point sets.")
            # print("Translation: ", None)
            # print("Rotation angle: ", None)
            # print("Scale: ", None)
            H = np.array(np.eye(3))

            return  None
            # Find homography
        if len(pts1) < 4 or len(pts2) < 4:
            # print("Error: Insufficient corresponding points to calculate Homography.")
            H = np.array(np.eye(3))
            return None

        H, _ = cv2.findHomography(pts2, pts1, cv2.RANSAC, 5.0)



        # # # show
        # translation_matrix = np.array([[1, 0, (center[0] - int(radius)) ], [0, 1, (center[1] - int(radius))], [0, 0, 1]])
        # new_H = np.dot(translation_matrix, H)        
        # img_out = cv2.warpPerspective(img1, new_H, (img2.shape[1], img2.shape[0]))
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

        # transformed_small_img = cv2.warpPerspective(img1, H, (img2.shape[1], img2.shape[0]))

        # fig , ax =  plt.subplots(1,2,figsize=(20,10))
        # ax[0].imshow(img2)
        # ax[1].imshow(transformed_small_img)
        # plt.show()

        return H

    def evaluate(self,image ,original_image):
        image = cv2.imread(image, cv2.IMREAD_GRAYSCALE)
        original_image = cv2.imread(original_image, cv2.IMREAD_GRAYSCALE)
        if image is None or original_image is None:
            print('Could not open or find the images!')
            return 0 , 0 , 1000000

        if image.shape != original_image.shape:
            print('The images have different dimensions!')
            print(image,image.shape)
            print(original_image,original_image.shape)
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
        
    

    
    def feature(self, feature, matcher, distance):
        patient_dict = {}
        image_folder = self.path_match_image+ '/1/'
        filenames = sorted(os.listdir(image_folder))
        self.output_path  = self.path_match + '/' + 'crop_'+ feature + '_' + matcher + '_' + str(distance) + '/'
        for data in self.data_list:
            if not os.path.exists(os.path.join(self.output_path, data)):
                os.makedirs(os.path.join(self.output_path, data))
            if not os.path.exists(os.path.join(self.output_path, data+ '_vis/')):
                os.makedirs(os.path.join(self.output_path, data+ '_vis/'))
            if not os.path.exists(os.path.join(self.output_path, data+ '_move/')):
                os.makedirs(os.path.join(self.output_path, data+ '_move/'))
        print(feature, matcher, distance)
        print('------------------')
        pre_treatment_file = ''
        post_treatment_file = ''
        for filename in filenames:
            if filename.endswith('.png'):

                patient_id, eye, date = filename.split('.png')[0].split('_')

                if patient_id not in patient_dict:
                    patient_dict[patient_id] = {}
                if eye not in patient_dict[patient_id]:
                    patient_dict[patient_id][eye] = {}
                    pre_treatment_file = filename
                    pre_treatment = date

                else:
                    post_treatment_file = filename
                    post_treatment = date
                    patient_dict[patient_id][eye][post_treatment] = {}
                    if pre_treatment_file != '' and post_treatment_file != '':
                        pre_image = cv2.imread(self.path_match_image + '1/' + patient_id + '_' + eye + '_' + str(pre_treatment) + '.png')
                        post_image = cv2.imread(self.path_match_image + '1/' + patient_id + '_' + eye + '_' + str(post_treatment) + '.png')

                        pre_image = cv2.resize(pre_image, (304, 304))
                        post_image = cv2.resize(post_image, (304, 304))

                        pre_image = cv2.normalize(pre_image, None, 0, 255, cv2.NORM_MINMAX)
                        post_image = cv2.normalize(post_image, None, 0, 255, cv2.NORM_MINMAX)



                        # show histogram

                        # fig , ax =  plt.subplots(3,2,figsize=(10,10))
                        # ax[0][0].hist(pre_image2.ravel(), 256, [0, 256])
                        # ax[0][0].set_title('pre_image2')
                        # ax[0][1].hist(post_image2.ravel(), 256, [0, 256])
                        # ax[0][1].set_title('post_image2')
                        # ax[1][0].imshow(pre_image2)
                        # ax[1][1].imshow(post_image2)
                        # ax[2][0].hist(pre_image.ravel(), 256, [0, 256])
                        # ax[2][0].set_title('pre_image')
                        # ax[2][1].hist(post_image.ravel(), 256, [0, 256])
                        # ax[2][1].set_title('post_image')

                        # plt.show()
                        crop_img ,center,radius = self.find_center(post_image)
                        # print(center,radius)
                        # print(center[1] - radius,center[1] + radius,center[0] - radius,center[0] + radius)

                        pre_image2 = cv2.cvtColor(pre_image, cv2.COLOR_BGR2GRAY)
                        crop_img = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)

                        # pre_image2 = cv2.bilateralFilter(pre_image2, 5, 75, 75)
                        # crop_img = cv2.bilateralFilter(crop_img, 5, 75, 75)


                        # pre_image2 = cv2.equalizeHist(pre_image2)
                        # crop_img = cv2.equalizeHist(crop_img)
                

                        H = self.LK(pre_image2,crop_img,center=(center[0],center[1]), radius = int(radius), distance=distance, method=feature, matcher=matcher)
                        

                        if H is not None:
                            translation = (H[0, 2], H[1, 2])
                            rotation_angle = np.arctan2(H[1, 0], H[0, 0])
                            rotation = rotation_angle * 180 / np.pi
                            scale = (np.sqrt(H[0, 0] ** 2 + H[1, 0] ** 2), center)
                            print('translation',translation)
                            print('rotation',rotation)
                            print('scale',scale)
                            print('center',center)

                            # plt 
                            post_image = cv2.warpPerspective(post_image, H, (post_image.shape[1], post_image.shape[0]))
                            fig , ax =  plt.subplots(1,3,figsize=(10,10))
                            ax[0].imshow(pre_image2)
                            ax[1].imshow(crop_img)
                            ax[2].imshow(post_image)
                            plt.show()

                            # print(translation,center,radius,H)
                            translation_matrix = np.array([[1, 0, (center[0] - int(radius)) ], [0, 1, (center[1] - int(radius))], [0, 0, 1]])
                            new_H = np.dot(translation_matrix, H)


                            # 平移
                            translation = (new_H[0, 2], new_H[1, 2])
                      
                            # 旋轉
                            rotation = math.atan2(new_H[1, 0], new_H[0, 0])
                            # 旋轉角度
                            rotation_angle = rotation * 180 / np.pi
                            # 縮放
                            scale = np.sqrt(new_H[0, 0] ** 2 + new_H[1, 0] ** 2)

                            print('translation_matrix',translation_matrix)

                            print('translation2: {}'.format(translation))
                            print('rotation2: {}'.format(rotation))
                            print('rotation_angle2: {}'.format(rotation_angle))
                            print('scale2: {}'.format(scale))  
                            fig , ax =  plt.subplots(1,3,figsize=(10,10))
                            ax[0].imshow(pre_image2)
                            ax[1].imshow(crop_img)
                            ax[2].imshow(post_image)
                            plt.show()

                        for data in self.data_list:


                            if os.path.exists(self.path_match_image + data + '/' + patient_id + '_' + eye + '_' + pre_treatment + '.png'):
                                pre_image = cv2.imread(self.path_match_image + data + '/' + patient_id + '_' + eye + '_' + pre_treatment + '.png')
                                pre_image = cv2.resize(pre_image, (304, 304))
                                pre_image = cv2.normalize(pre_image, None, 0, 255, cv2.NORM_MINMAX)

                        
                                cv2.imwrite(self.output_path + data + '/' + patient_id + '_' + eye + '_' + pre_treatment + '.png', pre_image)
                            if os.path.exists(self.path_match_image + data + '/' + patient_id + '_' + eye + '_' + post_treatment + '.png'):
                                post_image = cv2.imread(self.path_match_image + data + '/' + patient_id + '_' + eye + '_' + post_treatment + '.png')
                                post_image = cv2.resize(post_image, (304, 304))
                                post_image = cv2.normalize(post_image, None, 0, 255, cv2.NORM_MINMAX)

                                # print(self.output_path + data + '/' + patient_id + '_' + eye + '_' + post_treatment + '.png')
                                cv2.imwrite(self.output_path + data + '/' + patient_id + '_' + eye + '_' + post_treatment + '.png', post_image)

                                if H is None:
                                    result = post_image
                                    translation = (0,0)
                                    rotation_angle = 0
                                    scale = 1
                                elif scale > 2 or scale < 0.5:
                                    result = post_image
                                    translation = (0,0)
                                    rotation_angle = 0
                                    scale = 1

                                elif translation[0] > 304 // 2 or translation[0] < -304 // 2:
                                    result = post_image
                                    translation = (0,0)
                                    rotation_angle = 0
                                    scale = 1

                                elif translation[1] > 304 // 2 or translation[1] < -304 // 2:
                                    result = post_image
                                    translation = (0,0)
                                    rotation_angle = 0
                                    scale = 1

                                elif rotation_angle > 90 or rotation_angle < -90:
                                    result = post_image
                                    translation = (0,0)
                                    rotation_angle = 0
                                    scale = 1

                                elif np.isnan(H).any():
                                    result = post_image
                                    translation = (0,0)
                                    rotation_angle = 0
                                    scale = 1


                                        
                                    # elif rotation_angle > 10 or rotation_angle < -10:
                                    #     result = post_image
                                    # elif translation[0] > 100 or translation[0] < -100:
                                    #     result = post_image
                                    # elif translation[1] > 100 or translation[1] < -100:
                                    #     result = post_image
                                else:
                                    # apply the homography to the second image
                                    result = cv2.warpPerspective(post_image, new_H, (pre_image.shape[1], pre_image.shape[0]))
                                    
                                patient_dict[patient_id][eye][post_treatment]['translation'] = translation
                                patient_dict[patient_id][eye][post_treatment]['rotation_angle'] = rotation_angle
                                patient_dict[patient_id][eye][post_treatment]['scale'] = scale
                                patient_dict[patient_id][eye][post_treatment]['center'] = center
                                patient_dict[patient_id][eye][post_treatment]['radius'] = radius

                                # plt
                                fig , ax =  plt.subplots(1,3,figsize=(10,10))
                                ax[0].imshow(pre_image)
                                ax[1].imshow(result)
                                ax[2].imshow(post_image)
                                plt.show()


                                    
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
                                vis = cv2.addWeighted(pre_image, 0.5, vis_img, 0.5, 0)
                                cv2.rectangle(vis, (int(center[0] - radius), int(center[1] - radius)), (int(center[0] + radius), int(center[1] + radius)), (255, 0, 255), 3)
                            
                                cv2.imwrite(self.output_path + data + '_vis/' + patient_id + '_' + eye + '_' + post_treatment + '.png', vis)
                                # fig , ax =  plt.subplots(1,6,figsize=(10,10))
                                # ax[0].imshow(pre_image)
                                # ax[0].set_title(pre_treatment_file.split('_')[2])
                                # ax[1].imshow(post_image)
                                # ax[1].set_title(post_treatment_file.split('_')[2])
                                # ax[2].imshow(crop_img)
                                # ax[3].imshow(transformed_img)
                                # ax[4].imshow(filled)
                                # ax[5].imshow(vis)

        

        return patient_dict

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
                    patient[patient_id + '_' + eye]['pre_treatment'] = date
                else :
                    patient[patient_id + '_' + eye][date] = {}
                    post_treatment =  date 

                    if patient[patient_id + '_' + eye]['pre_treatment'] != '' and patient[patient_id + '_' + eye][date] != '':

                        pre_treatment_img = self.path_match_image+ '1/' + patient_id + '_' + eye + '_' + pre_treatment + '.png'
                        post_treatment_img = self.path_match_image + '1/' + patient_id + '_' + eye + '_' + post_treatment + '.png'
                        matching_img = match_path  + '/1/' + patient_id + '_' + eye + '_' + post_treatment + '.png'
                        mse,psnr,ssim = self.evaluate(pre_treatment_img,post_treatment_img)
                        if psnr == float('inf') :
                            continue
                        if ssim == float('inf') : 
                            continue
                        if ssim  < 0 or psnr < 0 or mse <0 :
                            continue


                        patient[patient_id + '_' + eye][date]['original'] = [mse,psnr,ssim]

                        matching_mse,matching_psnr,matching_ssim = self.evaluate(pre_treatment_img,matching_img)
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

                        if matching_ssim - ssim > best_differece_ssim :
                            best_differece_ssim = matching_ssim - ssim
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


        matching_avg_mse = round(sum(matching_mse_list)/len(matching_mse_list),2)
        matching_avg_psnr = round(sum(matching_psnr_list)/len(matching_psnr_list),2)
        matching_avg_ssim = round(sum(matching_ssim_list)/len(matching_ssim_list),2)
        matching_mse_std = round(np.std(matching_mse_list, ddof=1),2)
        matching_psnr_std = round(np.std(matching_psnr_list, ddof=1),2)
        matching_ssim_std = round(np.std(matching_ssim_list, ddof=1),2)

        patient['avg'] = {}
        patient['avg']['mse'] = [matching_avg_mse,matching_mse_std]
        patient['avg']['psnr'] = [matching_avg_psnr,matching_psnr_std]
        patient['avg']['ssim'] = [matching_avg_ssim,matching_ssim_std]


        if best_case != {} :
            patient['best_case'] = best_case

        if worst_case != {} :
            patient['worst_case'] = worst_case

            
            avg_mse = round(sum(mse_list)/len(mse_list),3)
            avg_psnr = round(sum(psnr_list)/len(psnr_list),3)
            avg_ssim = round(sum(ssim_list)/len(ssim_list),3)
            mse_std = round(np.std(mse_list, ddof=1),3)
            psnr_std = round(np.std(psnr_list, ddof=1),3)
            ssim_std = round(np.std(ssim_list, ddof=1),3)

            patient['original'] = {}
            patient['original']['mse'] = [avg_mse,mse_std]
            patient['original']['psnr'] = [avg_psnr,psnr_std]
            patient['original']['ssim'] = [avg_ssim,ssim_std]
       
        # save patient to csv

        return patient








            




if __name__ == '__main__':
    pre_treatment_file = "..\\..\\Data\\PCV_1120\\ALL\\1\\08707452_L_20161130.png"
    post_treatment_file = "..\\..\\Data\\PCV_1120\\ALL\\1\\08707452_L_20170118.png"
    
    date = '1120'
    disease = 'PCV'
    PATH_DATA = '../../Data/' 
    PATH_BASE = PATH_DATA  + disease + '_' + date + '/'

    PATH_LABEL = PATH_DATA + 'labeled' + '/'
    PATH_IMAGE = PATH_DATA + 'OCTA/' 
    PATH_MATCH_IMAGE = PATH_BASE + 'ALL/'
    PATH_MATCH = PATH_MATCH_IMAGE + 'MATCH/' 
    PATH_MATCH_LABEL = PATH_MATCH_IMAGE + 'MATCH_LABEL/' 
    distances = [0.8]
    features = ['KAZE']
    matchers = ['BF']
    for distance in distances:
        for feature in features:
            for matcher in matchers:
                # print(PATH_MATCH + 'crop' + '_' +  feature + '_' + matcher + '_' + str(distance))

                find = finding(PATH_IMAGE,PATH_LABEL,PATH_MATCH,PATH_MATCH_LABEL,PATH_MATCH_IMAGE,distance)
                find_dict = find.feature(feature,matcher,distance)
                json_file = './record/'+ disease + '_' + date + '_'  +feature + '_' + matcher + '_' + str(distance) + '_cropAlign.json'
                tools.write_to_json_file(json_file, find_dict)
                eval = find.all_evaluate(PATH_MATCH + 'crop' + '_' +  feature + '_' + matcher + '_' + str(distance))
                json_file2 = './record/'+ disease + '_' + date + '_'+ feature + '_' + matcher + '_' + str(distance) + '_cropEval.json'
                tools.write_to_json_file(json_file2, eval)

    # for distance in distances:
    #     for feature in features:
    #         for matcher in matchers:
    #             find = finding(PATH_IMAGE,PATH_LABEL,PATH_MATCH,PATH_MATCH_LABEL,PATH_MATCH_IMAGE,distance)
    #             eval = find.all_evaluate(PATH_MATCH + 'crop' + '_' +  feature + '_' + matcher + '_' + str(distance))
    #             json_file = './record/'+ feature + '_' + matcher + '_' + str(distance) + '_eval.json'
    #             tools.write_to_json_file(json_file, eval)


