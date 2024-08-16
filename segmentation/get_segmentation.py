import os
import cv2
import numpy as np
import csv
from sklearn.metrics import jaccard_score, recall_score
from segmentation.UNetModel import *
import tensorflow as tf
from Deeplab import *
model_classes = {
    'UNet': UNet,
    'FRUNet': FRUNet,
    'UNetPlusPlus': UNetPlusPlus,
    'AttUNet': AttentionUNet,
    'BCDUNet': BCDUNet,
    'RecurrentUNet': RecurrentUNet,
    'ResUNet': ResUNet,
    'R2UNet': R2UNet,
    # 'R2AttentionUNet': R2AttentionUNet,
    'DenseUNet': DenseUNet,
    'MultiResUNet': MultiResUNet,
    'DCUNet': DCUNet,
    'SDUNet': SDUNet,
    'CARUNet' : CARUNet,
    'DeepLabV3Plus101': DeepLabV3Plus101,
    'DeepLabV3Plus50': DeepLabV3Plus50,
}

def get(dataset_name, model, output_path,model_path, model_name,predict_threshold,postprocess_signal):
    # get segmentation result
    img_path = os.path.join(dataset_name, "images")
    mask_path = os.path.join(dataset_name, "masks")
    print("img_path",img_path)
    print("mask_path",mask_path)
    # load model
    if model_name in model_classes:
        getModels = model_classes[model_name](
    
    model.load_weights(os.path.join(model_path, model_name))
    # make folder
    predict_path = os.path.join(output_path, "predict")
    list = [predict_path]
    if postprocess_signal:
        postprocess = os.path.join(output_path, "postprocess")
        list.append(postprocess)
    for path in list:
        if not os.path.isdir(path):
            os.makedirs(path)
            
        
        
if  __name__ == '__main__':
    dataset_name = '../Data/PCV_20240418/PCV_20240418_connectedComponent_bil51010_clah1016_concate34OCT_CC'
    model = DCUNet
    output_path = '../Data/PCV_20240418/get_segmentation'
    model_path = './Model/PCV_20240418/PCV_20240418_connectedComponent_bil51010_clah1016_concate34OCT_42_CC/train'
    model_name = 'DCUNet_150_2_0.01_32_1.h5'
    predict_threshold = 0.5
    postprocess_signal = False
    get(dataset_name, model, output_path,model_path, model_name,predict_threshold,postprocess_signal)