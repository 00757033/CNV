import cv2



import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import math
import json
import shutil
import time

# 尋找OCTA 影像的黃斑中心
class finding():
    def __init__(self,path):
        self.path = path


    def preprocess(self, image):
        
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # self.image = cv2.equalizeHist(self.image)
        # # 反轉顏色
        # self.image = cv2.bitwise_not(self.image)
        
        # image = cv2.equalizeHist(image)
        image = cv2.GaussianBlur(image, (3, 3), 0)
        image = cv2.equalizeHist(image)
        # image = cv2.GaussianBlur(image, (3, 3), 0)
        # image = cv2.equalizeHist(image)

        return image

    def find_center(self):
        for file in os.listdir(self.path):
            if file.endswith('.png'):
                pre_treatment = self.path+ file
                print(pre_treatment)
                image2 = cv2.imread(pre_treatment)
                image = self.preprocess(image2)

                ret1,image = cv2.threshold(image, 32 ,255, cv2.THRESH_BINARY_INV)
                print(ret1)

                # 找到影像中 最白的最大面積的圓
                _, labels, stats, centroids = cv2.connectedComponentsWithStats(image, connectivity=8)

                # 找到最大的區域（排除背景區域）
                label_area = dict()
                for i in range(1, labels.max() + 1):
                    label_area[i] = stats[i][4]

                # 依照面積從大到小排序
                label_area = sorted(label_area.items(), key=lambda kv: kv[1], reverse=True)

                # 找到最大的面積
                max_label = label_area[0][0]
                # 上色
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
                image[labels == max_label] = (0, 0, 255)

                # 找到最大的面積的中心點
                center = centroids[max_label]
                center = (int(center[0]), int(center[1]))
                # 找到最大的面積的半徑和直徑
                radius = stats[max_label][2]

                # 畫出黃斑中心
                # draw the circle
                cv2.circle(image, center, int(radius), (0, 0, 255), 10)
                cv2.circle(image, center, 5, (0, 255, 255), -1)



                # show the image wait 3 seconds and destroy
                cv2.imshow('image', image)
                cv2.imshow('image2', image2)

                cv2.waitKey(1000)
                cv2.destroyAllWindows()

        
                # save the image


        return 



    

    







if __name__ == '__main__':
    pre_treatment = "..\\..\\Data\\PCV_1120\\ALL\\1\\08707452_L_20161130.png"
    post_treatment = "..\\..\\Data\\PCV_1120\\ALL\\1\\08707452_L_20170118.png"
    
    date = '1120'
    disease = 'PCV'
    PATH_DATA = '../../Data/' 
    PATH_BASE = PATH_DATA  + disease + '_' + date + '/'

    PATH_LABEL = PATH_DATA + 'labeled' + '/'
    PATH_IMAGE = PATH_DATA + 'OCTA/' 
    PATH_MATCH_IMAGE = PATH_DATA + 'PCV_1120/ALL/'
    PATH_MATCH = PATH_MATCH_IMAGE + 'MATCH/' 
    PATH_MATCH_LABEL = PATH_MATCH_IMAGE + 'MATCH_LABEL/' 

    find = finding(PATH_MATCH_IMAGE + '1/')
    find.find_center()  
