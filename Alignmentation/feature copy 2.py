import cv2



import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import math
import json
import shutil

# 尋找OCTA 影像的黃斑中心
class finding():
    def __init__(self, image):
        self.image = image

        self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        self.image = cv2.equalizeHist(self.image)
        # # 反轉顏色
        self.image = cv2.bitwise_not(self.image)
        self.image = cv2.equalizeHist(self.image)
        self.image = cv2.GaussianBlur(self.image, (9, 9), 0)
        # self.image = cv2.equalizeHist(self.image)
        self.image = cv2.GaussianBlur(self.image, (9, 9), 0)
        # self.image = cv2.equalizeHist(self.image)
        self.image = cv2.bilateralFilter(self.image, 9, 150, 150)
        # self.image = cv2.equalizeHist(self.image)
        self.image = cv2.bilateralFilter(self.image, 9, 150, 150)
        self.image = cv2.equalizeHist(self.image)



        # self.image = cv2.bitwise_not(self.image)

    def find_center(self):

        # 找到影像中 最白的最大面積的圓
        # find the center
        minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc(self.image)

        # 最大的白色半徑
        max_radius = 0
        for i in range(len(self.image)):
            for j in range(len(self.image[0])):
                if self.image[i][j] == maxVal:
                    radius = math.sqrt((i - maxLoc[1]) ** 2 + (j - maxLoc[0]) ** 2)
                    if radius > max_radius:
                        max_radius = radius
                    break


        # 畫出黃斑中心
        self.image = cv2.cvtColor(self.image, cv2.COLOR_GRAY2BGR)
        # draw the circle
        cv2.circle(self.image, maxLoc, int(radius), (0, 0, 255), 10)
        cv2.circle(self.image, maxLoc, 20, (0, 255, 255), 10)


        # show the image
        cv2.imshow('image', self.image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        return maxLoc, radius



    

    







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
    for file in os.listdir(PATH_MATCH_IMAGE + '1/'):
        if file.endswith('.png'):
            pre_treatment = PATH_MATCH_IMAGE + '1/' + file
            find = finding(cv2.imread(pre_treatment))
            center, radius = find.find_center()