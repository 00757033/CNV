import tools.tools as tools
import pathlib as pl
import os
import cv2
import shutil
path = "../../Data"
labeled = path + '/' + 'labeled2'
date = '0831'
disease = 'CNV'
path_base =  path + "/" + disease + "_"+ date
image_path  = path + "/" + "OCTA"
layers = ['CC','OR']
OR = 0
CC = 0
for layer in layers :
    path = os.path.join(path_base,  layer,'label')
    print(path)
    for image_path in pl.Path(path).iterdir():
        if image_path.suffix == '.png':
            image_name = image_path.name
            image_stem = image_path.stem
            # print(image_stem)
            split_image_name = image_stem.split("_")
            
            sort_path = sorted(pl.Path(os.path.join(labeled,  split_image_name[1])).iterdir())
            
            print(split_image_name)
            img_path = str(image_path)
            # image = cv2.imread(img_path)
            # output_path = os.path.join(labeled,  split_image_name[1],split_image_name[3],split_image_name[2])
            # tools.makefolder(output_path)
            if layer =='CC':
                i = 4
                CC += 1
            else:
                i = 3
                OR += 1
            # cv2.imwrite(os.path.join(output_path,'label_'+ str(i) + '.png'),image)
            # cv2.imshow("combined", image)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()
print(CC)
print(OR)