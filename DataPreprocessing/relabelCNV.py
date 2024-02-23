import cv2
import numpy as np
from matplotlib import pyplot as plt
from pathlib import Path
from tools.function  import setFolder , otsu

#從原始的label資料中提取血管的部分當作訓練的label
def relabeler (path):
    path = Path(path)
    image_path = str(path / 'origin')
    label_path = str(path / 'label')
    output_path = str(path / 'label_otsu')
    output_path = Path(output_path)
    setFolder(output_path)
    output_path = str(output_path)
    for image_name in Path(image_path).glob("*.png"):
        image_name = str(image_name.name)
        image_path = str(image_path)
        label_path = str(label_path)
        image = cv2.imread(image_path + '/' + image_name, 0)
        image_mask = otsu(image)
        image_label = cv2.imread(label_path + '/' + image_name, 0)
        image_label[image_mask == 0] = 0
        #print(image_result)
        cv2.imwrite(output_path + '/' + image_name, image_label)
        

if __name__ == '__main__':
    image_label_path = Path('..\\..\\Data\\new_label\\00006202\\20220328\\R\\label_3.png')
    image_path = Path('..\\..\\Data\\OCTA\\00006202\\20220328\\R\\3.png')
    
    image = cv2.imread(str(image_path), 0)
    if image is None:
        print("Error")
    else :
        print(image.shape)

    image_mask = otsu(image)
    image_label = cv2.imread(str(image_label_path), 0)
    image_label[image_mask == 0] = 0
    cv2.imshow('image', image_label)
    cv2.waitKey(0)
    cv2.imwrite('..\\..\\Data\\new_label\\00006202\\20220328\\R\\label_otsu_3.png', image_label)
