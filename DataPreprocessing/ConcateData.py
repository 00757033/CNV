import cv2
import os
import numpy as np
import shutil
import matplotlib
import preprocessing 
matplotlib.use('TkAgg')  # 使用TkAgg后端
import tools.tools as tools
import pathlib as pl
class Concatenation() :
    def __init__(self, PATH, image_path,layers = {"3":"OR","4":"CC"}):
        self.PATH = PATH
        self.image_path = image_path
        self.layers = layers
    #concateLayer需要合併的影像集
    # def getConcate(self,input_name,ratio = 0.5):
    #     for layer in self.layers:
    #         input_path = os.path.join(self.image_path,  self.layers[layer],'images')
    #         print('input_path',input_path)
    #         tools.makefolder(os.path.join(self.image_path,input_name + '_concate_' + self.layers[layer],'images'))
    #         for image_path in pl.Path(input_path).iterdir():
    #                 image_name = image_path.name
    #                 image_stem = image_path.stem
    #                 split_image_name = image_stem.split("_")
                    
    #                 image_path = os.path.join(self.PATH,split_image_name[1],split_image_name[3],split_image_name[2])
    #                 image_path1 = os.path.join(image_path,'3.png')
    #                 image1   = cv2.imread(image_path1)
    #                 gray1   = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    #                 gray1 = np.expand_dims(gray1, axis = 2)
    #                 image_path2 = os.path.join(image_path,'4.png')
    #                 image2   = cv2.imread(image_path2)
    #                 gray2   = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
    #                 gray2 = np.expand_dims(gray2, axis = 2)
    #                 parm = { "3": [5,10,10,0.7,3],"4":[5,10,10,0.7,12]}

    #                 if split_image_name[0] == 'OR':
    #                     merged_image = np.dstack([gray1, gray2, np.zeros_like(gray1)])
    #                 else:
    #                     merged_image = np.dstack([gray2, gray1, np.zeros_like(gray1)])

    #                 output_path = os.path.join(self.image_path,input_name + '_concate_' + self.layers[layer],'images',image_name)
    #                 # cv2.imwrite(output_path,merged_image)
    #         mask_path = os.path.join(self.image_path, 'otsu_'+ self.layers[layer],'masks')
    #         print(mask_path,os.path.join(self.image_path,input_name + '_concate_' + self.layers[layer]))
    #         # shutil.copytree(mask_path, os.path.join(self.image_path,input_name + '_concate_' + self.layers[layer],'masks'))
            
    def getConcate(self,layer,concateLayer ,mask_file, output_file, bilateral = True, Unsharp = True, CLAHE = True):
        input_path1 = os.path.join(self.image_path, concateLayer[0],'images')
        input_path2 = os.path.join(self.image_path, concateLayer[1],'images')
        input_path3 = os.path.join(self.image_path, concateLayer[2],'images')
        print('input_path',input_path1)
        print('input_path',input_path2)
        print('input_path',input_path3)
        tools.makefolder(os.path.join(self.image_path,output_file,'images'))
        for image_path in pl.Path(input_path1).iterdir():
            image_name = image_path.name
            image_stem = image_path.stem
            split_image_name = image_stem.split("_")

            image_path1_image_name = concateLayer[0] + '_' + split_image_name[1] + '_' + split_image_name[2] + '_' + split_image_name[3] + '.png'
            image_path1 = os.path.join(input_path1,image_path1_image_name)
            image1   = cv2.imread(image_path1, cv2.IMREAD_GRAYSCALE)
            image_path2_image_name = concateLayer[1] + '_' + split_image_name[1] + '_' + split_image_name[2] + '_' + split_image_name[3] + '.png'
            image_path2 = os.path.join(input_path2,image_path2_image_name)
            image2   = cv2.imread(image_path2, cv2.IMREAD_GRAYSCALE)
            if image1 is None or image2 is None:
                continue

            pram = { "CC": [5,10,10,0.7,3],"OR":[5,10,10,0.7,12]}
            d,sigmaColor,sigmaSpace,clip,kernel = pram[layer]
            preprocessing_image = image1.copy()
            if bilateral:
                preprocessing_image = cv2.bilateralFilter(image1,d,sigmaColor,sigmaSpace)
            if Unsharp :
                preprocessing_image = cv2.GaussianBlur(preprocessing_image,(5,5),0)
                preprocessing_image = cv2.addWeighted(preprocessing_image, 1.5, image1, -0.5, 0)
            if CLAHE:
                clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(kernel,kernel))
                preprocessing_image = clahe.apply(preprocessing_image)

            # concate
            merged_image = np.dstack([preprocessing_image, image2, np.zeros_like(preprocessing_image)])
            output_path = os.path.join(self.image_path,output_file,'images',image_name)
            cv2.imwrite(output_path,merged_image)
        mask_path = os.path.join(self.image_path, mask_file +'_'+ layer,'masks')
        shutil.copytree(mask_path, os.path.join(self.image_path,output_file,'masks'))



        return output_file





if __name__ == "__main__":
    import cv2
    date = '1201'
    disease = 'PCV'
    PATH = "../../Data/"
    FILE = disease + "_"+ date
    image_path = PATH + FILE
    data_groups = ["OR", "CC"]
    dict_concate = {'OR': ["OR", "CC", "_bil510_Unsharp_clahe7"] , 'CC': ["CC", "OR", "_bil510_Unsharp_clahe7"]}
    
    for data_group in data_groups:
        path = FILE + '/'+ data_group
        label= Concatenation(path,image_path)
        label.getConcate(data_group,dict_concate[data_group] ,FILE +"_bil510_Unsharp_clahe7",FILE +'_bil510_Unsharp_clahe7' + '_concate_' + data_group, bilateral = True, Unsharp = True, CLAHE = True)