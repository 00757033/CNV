from sklearn.metrics import jaccard_score, precision_score, recall_score, accuracy_score
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
import tools.tools as tools
import pandas as pd
import pathlib as pl

class metrics():
    def __init__(self,path,layers = {"3":"OR","4":"CC"}):
        self.path = path
        self.layers = layers
    def metrics(self):
        pass
    def groundTruth(self,img1,img2):
        JI =  jaccard_score(img1,img2, average='micro')
        iou_dc = self.iou_dc(img1,img2)
        return JI,iou_dc
    
    def iou_dc(self,img1,img2):
        intersection = np.logical_and(img1, img2)
        union = np.logical_or(img1, img2)
        iou_score = np.sum(intersection) / np.sum(union)
        return iou_score
    def getMetrics(self,path_name1,path_name2):
        for layer in self.layers:
            iou_dc = 0
            JI = 0
            length = 0
            print(self.layers[layer])
            for images in pl.Path(self.path + '/'+ path_name1 + '/' + path_name1 + '_otsu_' +self.layers[layer] + '/masks').iterdir():
                if images.suffix == '.png':
                    length += 1
                    image_path = str(images)
                    image_name = images.name
                    image2_path = image_path.replace(path_name1,path_name2)
                    image2 = cv2.imread(image2_path, cv2.IMREAD_GRAYSCALE)
                    image1 = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
                    image1[image1 < 128] = 0
                    image1[image1 >= 128] = 1
                    image1 = np.array(image1).ravel()
                    image2[image2 < 128] = 0
                    image2[image2 >= 128] = 1
                    image2 = np.array(image2).ravel()

                    j, iou = self.groundTruth(image1,image2)
                    JI += j
                    iou_dc += iou

            print('iou_dc',iou_dc/length)
            print('JI',JI/length)

def find_unlabel():
    for images in os.listdir("..\\..\\Data\\PCV_0918\\CC\\images"):
        if images not in os.listdir("..\\..\\Data\\PCV_0819\\CC\\images") :
            print(images)

def mask_difference():
    for mask in pl.Path("..\..\Data\PCV_0918\PCV_0918_otsu_CC\masks").iterdir():
    #     if images not in os.listdir("..\\..\\Data\\PCV_0819\\CC\\images") :
        print(mask) 
        mask = str(mask)
        path0 = mask.replace('PCV_0918','PCV_0918_0')
        path2 = mask.replace('PCV_0918','PCV_0918_25')
        path5 = mask.replace('PCV_0918','PCV_0918_5')
        image_path = mask.replace('masks','images')
        image0_path = path0.replace('masks','images')
        image2_path = path2.replace('masks','images')
        image5_path = path5.replace('masks','images')

        image = cv2.imread(image_path)
        image0 = cv2.imread(image0_path)
        image2 = cv2.imread(image2_path)
        image5 = cv2.imread(image5_path)

        mask1 = cv2.imread(mask)
        mask5 = cv2.imread(path5)
        mask0 = cv2.imread(path0)
        mask2 = cv2.imread(path2)

        mask5d = mask1.copy()
        mask0d = mask1.copy()
        mask2d = mask1.copy()
        mask5d[mask5 > 0]  =0
        mask0d[mask0 > 0]  =0
        mask2d[mask2 > 0]  =0
        # plt the images
        # 影像 間距縮小
        plt.figure()
        plt.subplot(531)
        plt.imshow(mask1)
        plt.axis('off')
        plt.title('mask')
        plt.subplot(532)
        plt.imshow(image)
        plt.axis('off')
        plt.title('image')
        plt.subplot(534)
        plt.imshow(mask0)
        plt.axis('off')
        plt.title('mask 0 : 10')
        plt.subplot(535)
        plt.imshow(image0)
        plt.axis('off')
        plt.title('image 0 : 10')
        plt.subplot(536)
        plt.imshow(mask0d)
        plt.axis('off')
        plt.title('mask 0 : 10 difference')
        plt.subplot(537)
        plt.imshow(mask2)
        plt.axis('off')
        plt.title('mask 25 : 75')
        plt.subplot(538)
        plt.imshow(image2)
        plt.axis('off')
        plt.title('image 25 : 75')
        plt.subplot(539)
        plt.imshow(mask2d)
        plt.axis('off')
        plt.title('mask 25 : 75 difference')
        plt.subplot(5,3,10)
        plt.imshow(mask5)
        plt.axis('off')
        plt.title('mask 50 : 50')
        plt.subplot(5,3,11)
        plt.imshow(image5)
        plt.axis('off')
        plt.title('image 50 : 50')
        plt.subplot(5,3,12)
        plt.imshow(mask5d)
        plt.axis('off')
        plt.title('mask 50 : 50 difference')


        plt.show()
        
        


        # cv2.imwrite(images,image5d)
        # cv2.imshow('image1',image1)
        # cv2.imshow('image5',image5)
        # cv2.imshow('image5d',image5d)
        # cv2.imshow('image0',image0)
        # cv2.imshow('image0d',image0d)
        # cv2.imshow('image2',image2)
        # cv2.imshow('image2d',image2d)
        # cv2.waitKey(0)
        # cv2.destroyWindow()

        
if __name__ == '__main__' :
    date = '0918'
    date1 = '0918_5'
    disease = 'PCV'
    PATH = "../../Data/"
    PATH_BASE =  disease + "_"+ date
    PATH_BASE1 = disease + "_"+ date1
    PATH_LABEL = PATH + "/" + "labeled"
    PATH_IMAGE = PATH + "/" + "OCTA"
    layers = ['CC','OR']
    mask_difference()
    # math = metrics(PATH)
    # math.getMetrics(PATH_BASE,PATH_BASE1)

