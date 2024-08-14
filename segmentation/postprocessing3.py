import numpy as np
import cv2
import os
import glob
from pathlib import Path
import json
import tools.tools as tools
import tensorflow as tf

import sklearn.metrics as metrics
from sklearn.metrics import confusion_matrix 
import pathlib as pl

import pydensecrf.densecrf as dcrf
from pydensecrf.utils import unary_from_softmax, create_pairwise_bilateral, create_pairwise_gaussian, unary_from_labels

from skimage import io
from skimage.color import gray2rgb
from skimage import img_as_float
from skimage import img_as_uint


def jaccard_index(y_true, y_pred):
    score = metrics.jaccard_score(y_true, y_pred)
    return score

def dice_coefficient(y_true, y_pred):
    y_true = np.array(y_true).ravel()
    y_pred = np.array(y_pred).ravel()
    dice_score = np.sum(y_pred[y_true==1])*2.0 / (np.sum(y_pred) + np.sum(y_true))
    return dice_score

class postprocessing():
    def __init__(self,image_path,result_path,post_path):
        self.image_path = image_path
        self.result_path = result_path
        self.post_path = post_path

    def postprocessing(self):
        # all image in file to crf
        for i in os.listdir(self.image_path):
            image = cv2.imread(self.image_path + i)
            annotated_image = cv2.imread(self.result_path + i)
            output_image = self.post_path + i
            crf(image, annotated_image,output_image, use_2d = True)


def crf(original_image, predict_image,output_image, use_2d = True,min_area = 10):
    # 用在血管分割上
    # 將分割結果轉成rgb
    original_image = original_image[:,:,0]
    original_image = gray2rgb(original_image)
    annotated_image = predict_image.copy()
    if(len(annotated_image.shape)>2):
        annotated_image = annotated_image[:,:,0]
    if(len(annotated_image.shape)<3):
        annotated_image = gray2rgb(annotated_image)

    # 轉成uint32
    annotated_image = annotated_image.astype(np.uint32)
    annotated_label = annotated_image[:,:,0] + (annotated_image[:,:,1]<<8) + (annotated_image[:,:,2]<<16)

    # Convert the 32bit integer color to 0,1, 2, ... labels.
    colors, labels = np.unique(annotated_label, return_inverse=True)

    #Creating a mapping back to 32 bit colors
    colorize = np.empty((len(colors), 3), np.uint8)
    colorize[:,0] = (colors & 0x0000FF)
    colorize[:,1] = (colors & 0x00FF00) >> 8
    colorize[:,2] = (colors & 0xFF0000) >> 16
    
    #Gives no of class labels in the annotated image
    n_labels = len(set(labels.flat)) 
    # print(annotated_image > 0)
    # print("--------------------")

    if use_2d :
        if n_labels > 1:
            # Example using the DenseCRF2D code
            d = dcrf.DenseCRF2D(original_image.shape[1], original_image.shape[0], n_labels)

            # 得到一元勢函數 unary potential
            U = unary_from_labels(labels, n_labels, gt_prob=0.8, zero_unsure=False)
            # U = unary_from_softmax(annotated_image, scale=1, clip=0.0001, zero_unsure=False)
            d.setUnaryEnergy(U)
            # 建立二元勢函數 pairwise potential 二元势就引入了邻域像素对当前像素的影响，所以需要同时考虑像素的位置和其观测值
            # sdims = (3, 3)  # 位置特征的scaling参数，决定位置对二元势的影响
            # schan = (0.01,)  # 颜色通道的平滑



            pairwise_energy = create_pairwise_bilateral(sdims=(3,3), schan=(0.1,), img=original_image, chdim=2)
            d.addPairwiseEnergy(pairwise_energy, compat=10)


            Q = d.inference(5)
            final = np.argmax(Q, axis=0).reshape((original_image.shape[0], original_image.shape[1]))

            final = colorize[final,:]
            cv2.imwrite(output_image, final)
        elif predict_image is not None:
            final = predict_image
            cv2.imwrite(output_image, predict_image)
            print("only one class")
        else:
            print(predict_image)
            final = predict_image
            print("no class")

        return final


# def crf(original_image, predict_image,output_image, use_2d = True):
#     # 用在血管分割上
#     annotated_image = predict_image.copy()
#     # Converting annotated image to RGB if it is Gray scale
#     if(len(annotated_image.shape)<3):
#         annotated_image = gray2rgb(annotated_image)
    
#     # cv2.imwrite("testing2.png",annotated_image)
#     annotated_image = annotated_image.astype(np.uint32)
#     #Converting the annotations RGB color to single 32 bit integer
#     annotated_label = annotated_image[:,:,0].astype(np.uint32) + (annotated_image[:,:,1]<<8).astype(np.uint32) + (annotated_image[:,:,2]<<16).astype(np.uint32)
#     # Convert the 32bit integer color to 0,1, 2, ... labels.
#     colors, labels = np.unique(annotated_label, return_inverse=True)
    
#     #Creating a mapping back to 32 bit colors
#     colorize = np.empty((len(colors), 3), np.uint8)
#     colorize[:,0] = (colors & 0x0000FF)
#     colorize[:,1] = (colors & 0x00FF00) >> 8
#     colorize[:,2] = (colors & 0xFF0000) >> 16
    
#     #Gives no of class labels in the annotated image
#     n_labels = len(set(labels.flat)) 
    
#     # print("No of labels in the Image are ")
#     # print(n_labels)
    
    
#     #Setting up the CRF model
#     if use_2d :
#         d = dcrf.DenseCRF2D(original_image.shape[1], original_image.shape[0], n_labels)

#         # get unary potentials (neg log probability) 
#         if n_labels > 1:
#             U = unary_from_labels(labels, n_labels, gt_prob=0.9, zero_unsure=False)
#             d.setUnaryEnergy(U)
#         else :
#             print("No labels")
#             cv2.imwrite(output_image,predict_image)
#             return

#         # This adds the color-independent term, features are the locations only.
#         d.addPairwiseGaussian(sxy=(3, 3), compat=3, kernel=dcrf.DIAG_KERNEL,
#                           normalization=dcrf.NORMALIZE_SYMMETRIC)

#         # This adds the color-dependent term, i.e. features are (x,y,r,g,b).
#         d.addPairwiseBilateral(sxy=(80, 80), srgb=(13, 13, 13), rgbim=original_image,
#                            compat=10,
#                            kernel=dcrf.DIAG_KERNEL,
#                            normalization=dcrf.NORMALIZE_SYMMETRIC)
        
#     #Run Inference for 5 steps 
#     Q = d.inference(5)

#     # Find out the most probable class for each pixel.
#     MAP = np.argmax(Q, axis=0)

#     # Convert the MAP (labels) back to the corresponding colors and save the image.
#     # Note that there is no "unknown" here anymore, no matter what we had at first.
#     MAP = colorize[MAP,:]
#     cv2.imwrite(output_image,MAP.reshape(original_image.shape))
#     return MAP.reshape(original_image.shape)


if __name__ == '__main__':
    list = ['PCV_1011_otsu_bil_clahe_42_CC','PCV_1011_otsu_bil_clahe_42_OR','PCV_1011_otsu_bil_clahe_42_aug_CC']# ,'PCV_1011_otsu_bil_clahe_42_aug_OR'
    
    file ='./Result/PCV_0205/' 
    for i in pl.Path(file).iterdir():
        print(i)
    for j in pl.Path(file+'PCV_0205_bil510_clahe7_concate_42_aug2_OR').iterdir():
        print(j)
        for k in j.iterdir():
            print(k)
            model_name = k.name
            if not model_name.endswith('_2') and not model_name.endswith('_3') :
                images_path = str(k)+'/images/'
                results_path =  str(k)+'/predict/'
                postcrf_path = str(k)+'/postcrf/'
                print(images_path,results_path,postcrf_path)
                if not os.path.exists(postcrf_path):
                    os.makedirs(postcrf_path)
                print(images_path)
                postprocess= postprocessing(images_path,results_path,postcrf_path)
                postprocess.postprocessing()
                print("finish")


    # postprocessing = postprocessing('./Result/PCV_1011/trainset/images/','./Result/PCV_1011/trainset/results/','./Result/PCV_1011/trainset/post/')