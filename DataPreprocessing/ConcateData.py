import cv2
import os
import numpy as np
import shutil
import matplotlib
import preprocessing 
matplotlib.use('TkAgg')  # 使用TkAgg后端
import tools.tools as tools
import pathlib as pl
import matplotlib.pyplot as plt
from skimage.filters import threshold_otsu, threshold_niblack, threshold_sauvola 
class Concatenation() :
    def __init__(self, PATH, image_path,layers = {"4":"CC"}):
        self.PATH = PATH
        self.image_path = image_path
        self.layers = layers   
    def getConcate(self,layer,concateLayer ,mask_file, output_file):
        if 'ALL' in concateLayer[1]:
            input_path=[os.path.join(self.image_path, path) for path in concateLayer]
            input_path[0] = os.path.join(input_path[0],'images')
        else:
            input_path=[os.path.join(self.image_path, path,'images') for path in concateLayer]
        print('input_path',input_path)
        tools.makefolder(os.path.join(self.image_path,output_file,'images'))
        tools.makefolder(os.path.join(self.image_path,output_file,'images_original'))
        tools.makefolder(os.path.join(self.image_path,output_file,'masks'))
        for image_path in pl.Path( input_path[0]).iterdir():
            image_name = image_path.name
            image_stem = image_path.stem
            split_image_name = image_stem.split("_")
            merged_image = np.zeros((304,304,2))
            check = True
            for i in range(len(input_path)):
                # concate 
                image_path = os.path.join(input_path[i],image_name)
                image   = cv2.imread(image_path)
                if image is None:
                    check = False
                    continue
                image   = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                image   = cv2.resize(image, (304,304))
                image   = np.expand_dims(image, axis = 2)
                if i == 0:
                    cv2.imwrite(os.path.join(self.image_path,output_file,'images_original',image_name),image)

                if i > 0:

                    if '_OCT' in image_path:
                        merged_image = np.dstack([merged_image, image])
                        # plt.imshow(image, cmap='gray')
                        # plt.title(image_name)
                        # plt.axis('off')
                        # plt.show()
                    else:
                        # blur = cv2.bilateralFilter(image, 3, 10, 10)
                        blur = cv2.GaussianBlur(image, (3, 3), 0)
                        # blur = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8,8)).apply(blur)
                        otsu_image = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                        # niblack = threshold_niblack (blur, window_size=45, k=0.3)
                        # binary_image = np.where(blur > niblack, 255, 0).astype(np.uint8)
                        
                        # 刪除小面積
                        # 連通域的數目 連通域的圖像 連通域的信息 矩形框的左上角坐標 矩形框的寬高 面積
                        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(otsu_image[1], connectivity=8)
                        areas = stats[:, cv2.CC_STAT_AREA]
                        without_background = otsu_image[1].copy()
                        output = np.zeros((labels.shape[0], labels.shape[1], 3), dtype=np.uint8)
                        for i in range(1, num_labels):
                            if areas[i] < 50:
                                without_background[labels == i] = 0
                        without_background = np.expand_dims(without_background, axis = 2)
                        # fig, ax = plt.subplots(1, 2)
                        # ax[0].imshow(image, cmap='gray')
                        # ax[0].set_title('original')
                        # ax[0].axis('off')
                        # ax[1].imshow(without_background, cmap='gray')
                        # ax[1].set_title('without_background')
                        # ax[1].axis('off')
                        # plt.show()
                        
                        # num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_image, connectivity=4)
                        # areas = stats[:, cv2.CC_STAT_AREA]
                        # without_background = binary_image.copy()
                        # output = np.zeros((labels.shape[0], labels.shape[1], 3), dtype=np.uint8)
                        # for i in range(1, num_labels):
                        #     if areas[i] < 50:
                        #         without_background[labels == i] = 0
                        # without_background = np.expand_dims(without_background, axis = 2)
                        # fig, ax = plt.subplots(1, 2)
                        # ax[0].imshow(image, cmap='gray')
                        # ax[0].set_title('original')
                        # ax[0].axis('off')
                        # ax[1].imshow(without_background, cmap='gray')
                        # ax[1].set_title('without_background')
                        # ax[1].axis('off')
                        # plt.show()
                        

                        
                        merged_image = np.dstack([merged_image, without_background])
                        # merged_image = np.dstack([merged_image, image])


                    # cv2.imshow('otsu_image' + str(i),without_background)
                    # cv2.waitKey(0)
                    # cv2.destroyAllWindows()
                else:
                    merged_image = np.dstack([image])
                    # cv2.imshow('image',image)
                    # cv2.waitKey(0)
                    # cv2.destroyAllWindows()
                   
            # print('merged_image',merged_image.shape)   
            output_path = os.path.join(self.image_path,output_file,'images',image_name)
            if check:
                cv2.imwrite(output_path,merged_image)
                mask_path = os.path.join(self.image_path, mask_file +'_'+ layer,'masks',image_name)
                output_mask_image = os.path.join(self.image_path,output_file,'masks',image_name)
                # print(mask_path,output_mask_image)
                shutil.copyfile(mask_path,output_mask_image)
                
                



        return output_file


if __name__ == "__main__":
    import cv2
    date = '20240814'
    disease = 'PCV'
    PATH = "../../Data/"
    FILE = disease + "_"+ date
    image_path = PATH + FILE
    data_groups = [ "CC"]
    filters = "_connectedComponent_bil31010_clah0712"
    dict_concate = {'OR': [FILE  + filters+"_OR", FILE  + "_CC",FILE + "_OR"] , 'CC': [FILE  + filters+"_CC", "ALL/4/", "ALL/3/" ,"ALL/4_OCT/"]}# ,"ALL/4_OCT/"
    
    for data_group in data_groups:
        path = FILE + '/'+ data_group
        label= Concatenation(path,image_path)
        label.getConcate(data_group,dict_concate[data_group] ,FILE  +filters,FILE  +filters + '_concate34OCT_' + data_group)