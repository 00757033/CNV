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
    def getConcate(self,layer,concateLayer ,mask_file, output_file):
        if 'ALL' in concateLayer[1]:
            input_path=[os.path.join(self.image_path, path) for path in concateLayer]
            input_path[0] = os.path.join(input_path[0],'images')
        else:
            input_path=[os.path.join(self.image_path, path,'images') for path in concateLayer]
        print('input_path',input_path)
        tools.makefolder(os.path.join(self.image_path,output_file,'images'))
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


                if i > 0:
                    # blur = cv2.GaussianBlur(image,(5,5),0)
                    otsu_image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                    merged_image = np.dstack([merged_image, otsu_image[1]])
                    # cv2.imshow('otsu_image',otsu_image[1])
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
    date = '0306'
    disease = 'PCV'
    PATH = "../../Data/"
    FILE = disease + "_"+ date
    image_path = PATH + FILE
    data_groups = [ "CC"]
    filters = "_connectedComponent_bil510_clahe7"
    dict_concate = {'OR': [FILE  + filters+"_OR", FILE  + "_CC",FILE + "_OR"] , 'CC': [FILE  + filters+"_CC", "ALL/3/" ,"ALL/4_OCT/"]}
    
    for data_group in data_groups:
        path = FILE + '/'+ data_group
        label= Concatenation(path,image_path)
        label.getConcate(data_group,dict_concate[data_group] ,FILE  +filters,FILE  +filters + '_concateOCT_' + data_group)