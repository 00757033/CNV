import albumentations as A
import cv2
import tools.tools as tools
import pathlib as pl
import shutil
import os
from matplotlib import pyplot as plt
class Augment():
    def __init__(self,path,layers = {"3":"OR","4":"CC"}):
        self.path = path
        self.layers = layers
        self.train_test = ["train","test","valid"]

    def albumentation(self, path,output,image_name):
        image = cv2.imread(path)
        mask = cv2.imread(path.replace('images','masks'))
        
        # Define the augmentation pipeline
        transform = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.Transpose(p=0.5),
            A.GridDistortion(p=0.5),
            A.OpticalDistortion(p=0.5),
            A.RandomBrightnessContrast(p=0.5),
            A.GridDistortion(p=0.5),
            A.OneOf([
                A.MotionBlur(p=0.2),
                A.MedianBlur(blur_limit=3, p=0.1),
                A.Blur(blur_limit=3, p=0.1),
            ], p=0.2), 
        ],is_check_shapes=False)
        
        # Augment an image
        augmented = transform(image=image, mask=mask)
        aug_image = augmented['image']
        aug_mask = augmented['mask']
        # # show the augmented image
        # fig, ax = plt.subplots(2, 2, figsize=(10, 10))
        # ax[0, 0].imshow(image)
        # ax[0, 0].set_title('Original image')
        # ax[0, 1].imshow(mask)
        # ax[0, 1].set_title('Original mask')
        # ax[1, 0].imshow(aug_image)
        # ax[1, 0].set_title('Augmented image')
        # ax[1, 1].imshow(aug_mask)
        # ax[1, 1].set_title('Augmented mask')
        # plt.show()
        
        
        
        cv2.imwrite(os.path.join(output,'images',image_name + '.png'), aug_image)
        cv2.imwrite(os.path.join(output,'masks',image_name + '.png'), aug_mask)
        
        

    def augumentation(self,input_path,output_name,augment_time = 5):
        data_dir = ['images','masks']
        for layer in self.layers:
            layer_input_path = tools.get_label_path(input_path, self.layers[layer])
            layer_output_path = tools.get_label_path(input_path + '_' + output_name, self.layers[layer])
            input_dir = os.path.join(self.path,  layer_input_path)
            output_dir= os.path.join(self.path,  layer_output_path)
            print('input_dir',input_dir)
            print('output_dir',output_dir)
            for train_test in self.train_test:     
                if train_test == 'train' : 
                    input = os.path.join(input_dir,'train')
                    output = os.path.join(output_dir,'train')
                    for  dir in data_dir:
                        tools.makefolder(os.path.join(output,  dir))
                    for image in pl.Path(input + '/'+ 'images').iterdir():
                        if image.suffix == ".png" :
                            image_name = image.stem
                            img_path = str(image)    
                            cv2.imwrite(os.path.join(output,'images',image_name + '.png') ,cv2.imread(img_path))
                            cv2.imwrite(os.path.join(output,'masks',image_name + '.png') ,cv2.imread(img_path.replace('images','masks')))
                            
                            for time in range(augment_time):     
                                self.albumentation(img_path,output,image_name + '_' + str(time))

                else :
                    print('copytree')
                    print(input_dir+ '/' +train_test)
                    shutil.copytree(input_dir+ '/' +train_test, output_dir+'/' +train_test)

def make_output_path(path ,output_name, layer_name):
        output_image = os.path.join(path, f"{output_name}_{layer_name}/images")
        output_label = os.path.join(path, f"{output_name}_{layer_name}/masks")
        for output_dir in [output_image, output_label]:
            os.makedirs(output_dir, exist_ok=True)

        return output_image, output_label

if __name__ == "__main__":
    date = '0205'
    disease = 'PCV'
    PATH = "../../Data/"
    PATH_BASE =  PATH + "/" + disease + "_"+ date
    PATH_LABEL = PATH + "/" + "new_label"
    PATH_IMAGE = PATH + "/" + "OCTA"
    preprocess = Augment(PATH_BASE + '/' + 'trainset')
    augment_time = [3,4,5]
    for time in augment_time:
        preprocess.augumentation(disease + "_"+ date +  '_bil510_clahe7_concate_42','aug' + str(time),time)    