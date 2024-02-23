import cv2
from skimage import morphology


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
        image = cv2.GaussianBlur(image, (3, 3), 0)
        # ax[4].hist(image.ravel(), 256, [0, 256])
        # ax[4].set_title('GaussianBlur')
        image = cv2.equalizeHist(image)
        # ax[5].hist(image.ravel(), 256, [0, 256])
        # ax[5].set_title('equalizeHist')
        image = cv2.GaussianBlur(image, (3, 3), 0)
        # ax[6].hist(image.ravel(), 256, [0, 256])
        # ax[6].set_title('GaussianBlur')

        # image = cv2.equalizeHist(image)
        # ax[7].hist(image.ravel(), 256, [0, 256])
        # ax[7].set_title('equalizeHist')
        # plt.show()
        image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)
        
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
    
        if int(center[0] - max_radius) < t :
            t = int(center[0] - max_radius)
        if int(center[1] - max_radius) < t :
            t = int(center[1] - max_radius)
        if int(center[0] + max_radius) > len(image[0]) - t :
            t = len(image[0]) - int(center[0] + max_radius)
        if int(center[1] + max_radius) > len(image[1]) - t :
            t = len(image) - int(center[1] + max_radius)

        crop_img = image[int(center[1] - max_radius)-t:int(center[1] + max_radius)+t, int(center[0] - max_radius)-t:int(center[0] + max_radius)+t]
        # show the image wait 3 seconds and destroy
        # cv2.imshow('image', image)
        # cv2.imshow('image2', image2)
        # cv2.imshow('image3', draw_image)
        # cv2.imshow('image4', crop_img)

        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        return  crop_img , center , max_radius + t

    def LK(self,img1, img2,center,  distance=0.9, method='KAZE',matcher='BF'):
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
        fig , ax =  plt.subplots(2,2,figsize=(10,10))
        ax[0,0].imshow(img1_draw)
        ax[0,1].imshow(img2_draw)
        ax[1,0].imshow(img1)
        ax[1,1].imshow(img2)
        plt.show()
        # Convert to numpy arrays


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

        # print('translation: {}'.format(translation))
        # print('rotation: {}'.format(rotation))
        # print('scale: {}'.format(scale))


        # transformed_small_img = cv2.warpPerspective(img1, H, (img2.shape[1], img2.shape[0]))

        # fig , ax =  plt.subplots(1,2,figsize=(20,10))
        # ax[0].imshow(img2)
        # ax[1].imshow(transformed_small_img)
        # plt.show()

        return H



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
                        pre_image = cv2.imread(self.path_match_image + '1/' + patient_id + '_' + eye + '_' + pre_treatment + '.png')
                        post_image = cv2.imread(self.path_match_image + '1/' + patient_id + '_' + eye + '_' + post_treatment + '.png')


                        pre_image2 = cv2.normalize(pre_image, None, 0, 255, cv2.NORM_MINMAX)
                        post_image2 = cv2.normalize(post_image, None, 0, 255, cv2.NORM_MINMAX)



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



                        crop_img ,center,radius = self.find_center(post_image2)

                        crop_pre_img , center2 , radius2 = self.find_center(pre_image2)

                        # show image
                        fig , ax =  plt.subplots(1,2,figsize=(20,10))
                        ax[0].imshow(crop_img)
                        ax[0].set_title(str(center) + ' ' + str(radius))
                        ax[1].imshow(crop_pre_img )
                        ax[1].set_title(str(center2)+ ' ' + str(radius2))

                        plt.show()


                        # print(center,radius)
                        # print(center[1] - radius,center[1] + radius,center[0] - radius,center[0] + radius)

                        crop_pre_img = cv2.cvtColor(crop_pre_img, cv2.COLOR_BGR2GRAY)
                        crop_img = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)

                        # crop_pre_img = cv2.threshold(crop_pre_img, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
                        # crop_img = cv2.threshold(crop_img, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

                        # crop_pre_img[crop_pre_img == 255] = 1
                        # crop_img[crop_img == 255] = 1




                        # crop_pre_img = morphology.skeletonize(crop_pre_img)
                        # crop_img = morphology.skeletonize(crop_img)

                        # crop_pre_img = cv2.cvtColor(crop_pre_img,  cv2.COLOR_GRAY2BGR)
                        # crop_img = cv2.cvtColor(crop_img,  cv2.COLOR_GRAY2BGR)

                        # show image
                        fig , ax =  plt.subplots(1,2,figsize=(20,10))
                        ax[0].imshow(crop_img)
                        ax[0].set_title(str(center) + ' ' + str(radius))
                        ax[1].imshow(crop_pre_img )
                        ax[1].set_title(str(center2)+ ' ' + str(radius2))
                        plt.show()
                

                        H = self.LK(crop_img,crop_pre_img,center=(center[0],center[1]), distance=distance, method=feature, matcher=matcher)

                        # if H is not None:
                        #     translation_matrix = np.array([[1, 0, 0-(center[0] - int(radius)) ], [0, 1, 0-(center[1] - int(radius))], [0, 0, 1]])
                        #     H = np.dot(translation_matrix, H)
                        #     # 平移
                        #     translation = (H[0, 2], H[1, 2])
                        #     # 旋轉
                        #     rotation = math.atan2(H[1, 0], H[0, 0])
                        #     # 旋轉角度
                        #     rotation_angle = math.degrees(rotation)
                        #     # 縮放
                        #     scale = H[0, 0] / math.cos(rotation)

                        #     print('translation: {}'.format(translation))
                        #     print('rotation: {}'.format(rotation))
                        #     print('rotation_angle: {}'.format(rotation_angle))
                        #     print('scale: {}'.format(scale))  

                        # for data in self.data_list:


                        #     if os.path.exists(self.path_match_image + data + '/' + patient_id + '_' + eye + '_' + pre_treatment + '.png'):
                        #         pre_image = cv2.imread(self.path_match_image + data + '/' + patient_id + '_' + eye + '_' + pre_treatment + '.png')

                        #         # print(self.output_path + data + '/' + patient_id + '_' + eye + '_' + pre_treatment + '.png')
                        #         cv2.imwrite(self.output_path + data + '/' + patient_id + '_' + eye + '_' + pre_treatment + '.png', pre_image)
                        #     if os.path.exists(self.path_match_image + data + '/' + patient_id + '_' + eye + '_' + post_treatment + '.png'):
                        #         post_image = cv2.imread(self.path_match_image + data + '/' + patient_id + '_' + eye + '_' + post_treatment + '.png')

                        #         # print(self.output_path + data + '/' + patient_id + '_' + eye + '_' + post_treatment + '.png')
                        #         cv2.imwrite(self.output_path + data + '/' + patient_id + '_' + eye + '_' + post_treatment + '.png', post_image)

                        #         if H is None:
                        #             result = post_image
                        #             result = post_image
                        #             translation = (0,0)
                        #             rotation_angle = 0
                        #             scale = 1
                        #         elif scale > 1.5 or scale < 0.5:
                        #             result = post_image
                        #             translation = (0,0)
                        #             rotation_angle = 0
                        #             scale = 1

                                        
                        #             # elif rotation_angle > 10 or rotation_angle < -10:
                        #             #     result = post_image
                        #             # elif translation[0] > 100 or translation[0] < -100:
                        #             #     result = post_image
                        #             # elif translation[1] > 100 or translation[1] < -100:
                        #             #     result = post_image
                        #         else:

                        #             # apply the homography to the second image
                        #             result = cv2.warpPerspective(post_image, H, (pre_image.shape[1], pre_image.shape[0]))
                                    
                        #         patient_dict[patient_id][eye][post_treatment]['translation'] = translation
                        #         patient_dict[patient_id][eye][post_treatment]['rotation_angle'] = rotation_angle
                        #         patient_dict[patient_id][eye][post_treatment]['scale'] = scale
                        #         patient_dict[patient_id][eye][post_treatment]['center'] = center
                        #         patient_dict[patient_id][eye][post_treatment]['radius'] = radius

                                    
                        #         filled = result.copy()
                        #         # cv2.imwrite(self.output_path + data + '_move/' + patient_id + '_' + eye + '_' + post_treatment + '.png', filled)

                        #         filled[filled == 0] = pre_image [filled == 0]


                        #         # cv2.imwrite(self.output_path + data + '/' + patient_id + '_' + eye + '_' + post_treatment + '.png', filled)

                        #         # add the two images together
                        #         vis_img = filled.copy()
                        #         vis_img[:,:,0] = 0
                        #         vis_img[:,:,2] = 0
                        #         vis = cv2.addWeighted(pre_image, 0.5, vis_img, 0.5, 0)
                        #         cv2.rectangle(vis, (int(center[0] - radius), int(center[1] - radius)), (int(center[0] + radius), int(center[1] + radius)), (255, 0, 255), 3)
                            
                        #         # cv2.imwrite(self.output_path + data + '_vis/' + patient_id + '_' + eye + '_' + post_treatment + '.png', vis)
                        #         fig , ax =  plt.subplots(1,6,figsize=(10,10))
                        #         ax[0].imshow(pre_image)
                        #         ax[0].set_title(pre_treatment_file.split('_')[2])
                        #         ax[1].imshow(post_image)
                        #         ax[1].set_title(post_treatment_file.split('_')[2])
                        #         ax[2].imshow(crop_img)
                        #         ax[2].set_title('crop')
                        #         ax[3].imshow(result)
                        #         ax[3].set_title('result')
                        #         ax[4].imshow(filled)
                        #         ax[4].set_title('filled')
                        #         ax[5].imshow(vis)
                        #         ax[5].set_title('vis')
                        #         plt.show()

        

        return patient_dict










                    




    

    







if __name__ == '__main__':
    pre_treatment_file = "..\\..\\Data\\PCV_1120\\ALL\\1\\08707452_L_20161130.png"
    post_treatment_file = "..\\..\\Data\\PCV_1120\\ALL\\1\\08707452_L_20170118.png"
    
    date = '1120'
    disease = 'PCV'
    PATH_DATA = '../../Data/' 
    PATH_BASE = PATH_DATA  + disease + '_' + date + '/'

    PATH_LABEL = PATH_DATA + 'labeled' + '/'
    PATH_IMAGE = PATH_DATA + 'OCTA/' 
    PATH_MATCH_IMAGE = PATH_DATA + 'PCV_1120/ALL/'
    PATH_MATCH = PATH_MATCH_IMAGE + 'MATCH/' 
    PATH_MATCH_LABEL = PATH_MATCH_IMAGE + 'MATCH_LABEL/' 
    distances = [0.75]
    features = ['FREAK'] # ,'KAZE','AKAZE','ORB','BRISK','FREAK'
    matchers = ['BF']
    for distance in distances:
        for feature in features:
            for matcher in matchers:

                find = finding(PATH_IMAGE,PATH_LABEL,PATH_MATCH,PATH_MATCH_LABEL,PATH_MATCH_IMAGE,distance)
                find_dict = find.feature(feature,matcher,distance)
                json_file = './'+ feature + '_' + matcher + '_' + str(distance) + '_crop.json'
                tools.write_to_json_file(json_file, find_dict)


